# coding: utf-8
import argparse
import torch
import torch.multiprocessing as _mp
import os
import time
import yaml
from torch.utils.tensorboard import SummaryWriter
from .trans_env import Translate_Env
from .adversarial_utils import *
from .agent import Agent
from joeynmt.model import Model
from joeynmt.data import load_data
from plm_mt.utils import load_and_process_data, BartVocab
from joeynmt.builders import build_optimizer, build_gradient_clipper, build_scheduler

from sacrebleu.metrics import BLEU
os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"
os.environ["OMP_NUM_THREADS"]="1"
os.environ["CUDA_LAUNCH_BLOCKING"]="1"
mp = _mp.get_context("spawn")

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=1, 
    help="number of parallel training thread")
parser.add_argument("--config_path", type=str, 
    default="./adversarials/adversarial_configs/attack_cwmt_zhen.yaml",
    help="path to adversarial config")
parser.add_argument("--save_to", type=str, 
    default="./adversarials/attack_cwmt_log",
    help="save adversarial model and logging")
parser.add_argument("--max_episode", type=int, default=1000000,
    help="maximum training episodes")
parser.add_argument("--reload", action="store_true",default=False, 
    help="whether reload the policy ckpt.")
parser.add_argument("--use_gpu", action="store_true", default=False,
    help="whether to use GPU (default as False).")

def train_thread(
        rank, device, args, global_update_step:mp.Value, lock:mp.Lock, 
        attack_configs:dict, discriminator_configs:dict, 
        victim_model:Model, word2near_vocab:dict,
        global_agent:Agent, agent_configs:dict, 
        global_saver:Saver=None,
    ):
    """
    an on-policy A3C training thread, starts by sync from global models, 
    updates local parameters and soft-updates global model episodically.
    training minimizes the bellman residual (TD learning) on a max-step rollout trajectory. 
    global_update_step 
    """
    # build and load victim translator locally and initiate to the env
    with open(attack_configs["victim_configs"], "r", encoding="utf-8") as victim_config_f:
        victim_configs = yaml.safe_load(victim_config_f)
    victim_model_path=attack_configs["victim_model"]
    # local saver saves the environments
    local_summary_writer=SummaryWriter(log_dir=os.path.join(args.save_to, "train_env%d"%rank))
    
    local_victim_model = build_translate_model(
        victim_configs=victim_configs, victim_model_path=victim_model_path,
        src_vocab=victim_model.src_vocab, trg_vocab=victim_model.trg_vocab,
        device=device
    )
    local_victim_model.eval()

    # build local agent
    local_agent = Agent(
        local_victim_model,
        **agent_configs["attacker_model_configs"]
    )
    if "plm_path" in attack_configs:
        # will process by PLM data preprocessing
        train_data, _, _, _, _ = load_and_process_data(
            data_cfg=victim_configs["data"], datasets=["train"]
        )
    else:
        train_data, _, _, _, _ = load_data(
            data_cfg=victim_configs["data"], datasets=["train"])
    # build local environments with local victim model and saver
    local_env = Translate_Env(            
        attack_configs, discriminator_configs, 
        local_victim_model, train_data, word2near_vocab,
        save_to=os.path.join(args.save_to, "train_env%d"%rank), 
        device=device)

    # build optimizers for the agent.
    local_optim = build_optimizer(
        config=agent_configs["attacker_optimizer_configs"],
        parameters=local_agent.parameters())
    local_clip_fn = build_gradient_clipper(
        config=agent_configs["attacker_optimizer_configs"]
    )
    local_scheduler, local_schedule_at = build_scheduler(
        config=agent_configs["attacker_optimizer_configs"],
        optimizer=local_optim,scheduler_mode="min")

    # main training loop with infinite batch initialization.
    local_step=0   # update step count for agent
    if args.reload and os.path.exists(os.path.join(local_env.save_to, "D.final")):
        local_step=local_env.reload()+1
    patience_t=discriminator_configs["patience"]
    discriminator_base_step = local_step  # step count for D
    episode_count=0   # each episode completes perturbation on a batch of full sequence
    # warmup_step = agent_configs["attacker_optimizer_configs"]["learning_rate_warmup"]
    trust_acc = 0.8
    acc_bound = discriminator_configs["acc_bound"]
    converged_bound = discriminator_configs["converged_bound"]
    while True:   # infinite training episodes (batch of sentences)
        # sync from the global agent
        try:
            print("%s init"%device, end=".")
            local_agent.sync_from(global_agent)
            local_env.reset_state()
            episode_length = 0
            while not local_env.terminated() or local_env.index<local_env.padded_src.shape[1]-1:
                # update discriminator by local adversarial policy
                if local_step % agent_configs["attacker_update_steps"]==0:
                    discriminator_base_step, trust_acc = local_env.update_discriminator(
                        local_agent, 
                        discriminator_base_step,
                        min_update_step=discriminator_configs["acc_valid_freq"],
                        max_update_step=discriminator_configs["discriminator_update_steps"],
                        random_inject_ratio=trust_acc-acc_bound if trust_acc>acc_bound else 0.0,
                        summary_writer=local_summary_writer
                    )
                    discriminator_base_step += 1
                    if trust_acc < discriminator_configs["converged_bound"]:  # GAN target reached
                        patience_t -= 1
                    else:
                        patience_t = discriminator_configs["patience"] 
                # saves the local env and the current global model
                if global_saver and (local_step+rank) % attack_configs["save_freq"]==0:
                    local_env.save(
                        global_step=local_step, model=local_env.discriminator,
                        optim=local_optim, scheduler=local_scheduler
                    )
                    # with lock:
                    #     global_saver.save(global_step=global_update_step.value, model=global_agent)
                    # if trust_acc < discriminator_configs["converged_bound"]:
                    #     global_saver.save(global_step=local_step, model=global_agent)
                if patience_t == 0:
                    print("training thread%d converged"%rank)# thread converged
                    return
                
                # ============= episode counts from 0, ignores BOS and EOS ====================
                # loop for a section of sequential policy for one update step
                local_agent.train()
                local_agent.to(device)
                values = []
                log_probs = []
                entropies = []
                rewards = []
                for rollout_count in range(local_agent.action_roll_steps):
                    episode_length+=1
                    assert local_env.padded_src[:, local_env.index-1:local_env.index+2].shape[1]==3, "index %d, sliced embedding must be 3"%local_env.index
                    state, terminal_flags = local_env.get_state()
                    attack_out, critic_out = local_agent(
                        state[:, 1:], local_env.src_mask, local_env.src_length,
                        state[:, local_env.index-1:local_env.index+2]
                    )
                    # prematurely finish trajectory accumulation
                    if local_env.terminated():
                        break  # breaks with an empty trojectory

                    logits_attack_out = torch.log(attack_out)
                    entropy = -(attack_out*logits_attack_out).sum(dim=-1)
                    local_summary_writer.add_scalar("reinforce/H", scalar_value=entropy.mean(),global_step=local_step)
                    actions = attack_out.multinomial(num_samples=1).detach()

                    # only extract log probs of chosen action and update env with action
                    log_attack_out = logits_attack_out.gather(-1, actions)
                    reward = local_env.step(actions.squeeze())
                    # print("rewards:", reward)
                    # mask out invalid part
                    critic_out *= (~terminal_flags.bool()).unsqueeze(dim=-1).int()
                    values.append(critic_out)  # [batch,1]
                    log_attack_out *=(~terminal_flags.bool()).unsqueeze(dim=-1).int()
                    log_probs.append(log_attack_out)  # [batch,1]
                    entropy *= (~terminal_flags.bool()).int()
                    entropies.append(entropy.unsqueeze(dim=-1))  # for entropy loss [batch,1]
                    rewards.append(reward.unsqueeze(dim=-1))  # [batch,1]
                
                if len(rewards)==0:
                    break  # terminated with an empty trojectory, refresh

                R = torch.zeros_like(critic_out)  # temrinated rewards are also padded 
                if not local_env.terminated():  
                    # in the middle of the trajectory
                    state, terminal_flags = local_env.get_state()
                    _, c_out = local_agent(
                        state[:, 1:], local_env.src_mask, local_env.src_length,
                        state[:, local_env.index-1:local_env.index+2])
                    R = c_out.detach()
                else:
                    print("%d samples end at %d/%d"%(state.shape[0], local_env.index, state.shape[1]))
                values.append(R)
                gae = torch.zeros_like(critic_out)
                policy_loss = torch.zeros_like(critic_out)
                value_loss = torch.zeros_like(critic_out)
                
                # if len(rewards)>0:
                for i in reversed(range(len(rewards))):
                    R = attack_configs["gamma"] * R + rewards[i]
                    advantage = R-values[i]
                    value_loss = value_loss + 0.5 * advantage.pow(2)
                    # generalize advantage
                    delta_t = rewards[i] + attack_configs["gamma"]*values[i+1] - values[i]
                    gae = gae*attack_configs["gamma"]+ delta_t
                    policy_loss = policy_loss - log_probs[i]*gae.detach() - attack_configs["entropy_coef"]*entropies[i]
                # discount by environments' trust region (D validation)
                total_loss = trust_acc*(policy_loss+attack_configs["value_coef"]*value_loss)
                valid_loss_count = total_loss.ne(0).sum().detach()
                mean_loss = total_loss.sum()/valid_loss_count
                mean_loss.backward()
                
                if local_clip_fn is not None:
                    local_clip_fn(params=local_agent.parameters())
                local_optim.step()
                if local_scheduler is not None and local_schedule_at=="step":
                    local_scheduler.step()
                local_optim.zero_grad()
                local_step += 1 
                local_summary_writer.add_scalar("lr/lr-A", scalar_value=local_optim.param_groups[0]["lr"], global_step=local_step)
                local_summary_writer.add_scalar("reinforce/policy_loss", scalar_value=trust_acc*policy_loss.sum(), global_step=local_step)
                local_summary_writer.add_scalar("reinforce/value_loss", scalar_value=trust_acc*value_loss.sum(), global_step=local_step)
                local_summary_writer.add_scalar("mean_loss", scalar_value=mean_loss, global_step=local_step)

                with lock:
                    # soft update global model (exponential moving average parameters)
                    global_update_step.value += 1
                    local_agent.to("cpu")
                    for param, target_param in zip(local_agent.parameters(), global_agent.parameters()):
                        target_param.data.copy_(attack_configs["tau"] * target_param.data+(1-attack_configs["tau"])* param.data)
                    local_agent.to(device)

            if local_scheduler is not None and local_schedule_at=="epoch":
                local_scheduler.step()
            local_optim.zero_grad()
            
            episode_count += 1
            if episode_count>args.max_episode:
                print("reach maximnum update episode. finish")
                return
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("OOM, skip batch...")
                local_optim.zero_grad()
                continue
            elif "`inf`, `nan`" in str(e):
                print("policy NAN, reset D", end="...")
                acc = local_env.acc_validation(
                    local_agent, 
                    random_inject_ratio=trust_acc-acc_bound if trust_acc>acc_bound else 0.0,
                    showNAN=True)
                print("nan env valid: ", acc)
                # reset environment's Discriminator by an initial random agent
                local_env.reset_D()

                if local_step<1000: # the policy is completely silenced
                    # reset the policy
                    global_agent.reset()

                local_agent = Agent(
                    local_victim_model,
                    **agent_configs["attacker_model_configs"]
                    )
                discriminator_base_step, trust_acc = local_env.update_discriminator(
                    local_agent, 
                    discriminator_base_step,
                    min_update_step=discriminator_configs["acc_valid_freq"],
                    max_update_step=discriminator_configs["discriminator_update_steps"],
                    random_inject_ratio=trust_acc-acc_bound if trust_acc>acc_bound else 0.0,
                    summary_writer=local_summary_writer
                )
                discriminator_base_step += 1

                # print(local_env.origin_padded_src, local_env.origin_bleu)
            else:
                raise e


def valid_thread(
        device, args, global_update_step:mp.Value, lock:mp.Lock, 
        attack_configs:dict, discriminator_configs:dict, 
        victim_model:Model, word2near_vocab:dict,
        global_agent:Agent, agent_configs:dict, 
        global_saver:Saver=None,
    ):
    # build and load victim translator locally and initiate to the env
    with open(attack_configs["victim_configs"], "r", encoding="utf-8") as victim_config_f:
        victim_configs = yaml.safe_load(victim_config_f)
    # local saver saves the environments
    local_summary_writer=SummaryWriter(log_dir=os.path.join(args.save_to, "dev_env"))

    victim_model_path=attack_configs["victim_model"]
    local_victim_model = build_translate_model(
        victim_configs=victim_configs, victim_model_path=victim_model_path,
        src_vocab=victim_model.src_vocab, trg_vocab=victim_model.trg_vocab,
        device=device
    )
    local_victim_model.eval()

    # build local agent
    local_agent = Agent(
        local_victim_model,
        **agent_configs["attacker_model_configs"]
    )
    local_agent.eval()  # evaluation model pulls from global
    if "plm_path" in attack_configs:
        # will process by PLM data preprocessing
        train_data, _, _, _, _ = load_and_process_data(
            data_cfg=victim_configs["data"], datasets=["train"]
        )
    else:
        train_data, _, _, _, _ = load_data(
            data_cfg=victim_configs["data"], datasets=["train"]
        )
    # build local environments with local victim model and saver
    local_env = Translate_Env(            
        attack_configs, discriminator_configs, 
        local_victim_model, train_data, word2near_vocab,
        save_to=os.path.join(args.save_to, "dev_env"), 
        device=device)

    local_step = 0
    if args.reload:
        local_step = local_env.reload()+1
    episode_count = 0
    while True:  # infinite episodes 
        try:
            print("valid thread init")
            local_env.reset_state()
            with lock:
                local_agent.sync_from(global_agent)
            episode_length = 0
            while not local_env.terminated() or local_env.index<=local_env.padded_src.shape[1]-2:
                # validate environments and check for discriminator update
                discriminator_base_step = local_step  # step count for D
                if local_step % (agent_configs["attacker_update_steps"]*args.n)==0:
                    discriminator_base_step, _ = local_env.update_discriminator(
                        local_agent, 
                        discriminator_base_step,
                        min_update_step=discriminator_configs["acc_valid_freq"],
                        max_update_step=discriminator_configs["discriminator_update_steps"],
                        random_inject_ratio=0.2,
                        summary_writer=local_summary_writer
                    )
                    discriminator_base_step += 1

                # # save the current global model
                # if global_saver and local_step % attack_configs["save_freq"]==0:
                #     with lock:
                #         global_saver.save(global_step=global_update_step.value, model=global_agent)

                assert local_env.padded_src[:, local_env.index-1:local_env.index+2].shape[1]==3, "index %d, sliced embedding must be 3"%local_env.index
                state, _ = local_env.get_state()
                attack_out, critic_out = local_agent(
                    state[:, 1:], local_env.src_mask, local_env.src_length,
                    state[:, local_env.index-1:local_env.index+2]
                ) 

                entropy = -(attack_out*torch.log(attack_out)).sum(dim=-1)
                local_summary_writer.add_scalar("reinforce/H", scalar_value=entropy.mean(),global_step=local_step)
                actions = attack_out.argmax(dim=-1).detach()  # [batch]
                # critic filters the action
                action_mask = critic_out.gt(0).long().squeeze()  # [batch]
                actions *= action_mask
                local_summary_writer.add_scalar("env/dev_act_rate", 
                    scalar_value=actions.sum().float()/actions.shape[0], global_step=local_step)

                # # only extract log probs of chosen action and update env with action
                # rewards = local_env.step(actions)
                # local_summary_writer.add_scalar("dev_r", 
                #     scalar_value=rewards.mean(), global_step=local_step)
                episode_length += 1
                local_step += 1
            current_bleu = local_env.get_current_bleu()
            success_count = 0.
            for i in range(len(current_bleu)):
                if current_bleu[i]< local_env.origin_bleu[i]:
                    success_count+=1
            print("success_rate:%.2f"%(success_count/len(current_bleu)))
            local_summary_writer.add_scalar("env/success_rate", 
                scalar_value=success_count/len(current_bleu), global_step=episode_count)
            episode_count += 1
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("OOM, skip batch...")
                continue
            else:
                raise e
               
def logging_thread(
        device, args, global_update_step:mp.Value, lock:mp.Lock, 
        attack_configs:dict, discriminator_configs:dict, 
        victim_model:Model, word2near_vocab:dict,
        global_agent:Agent, agent_configs:dict, 
        global_saver:Saver=None,
    ):
    # build and load victim translator locally and initiate to the env
    with open(attack_configs["victim_configs"], "r", encoding="utf-8") as victim_config_f:
        victim_configs = yaml.safe_load(victim_config_f)

    victim_model_path=attack_configs["victim_model"]
    local_victim_model = build_translate_model(
        victim_configs=victim_configs, victim_model_path=victim_model_path,
        src_vocab=victim_model.src_vocab, trg_vocab=victim_model.trg_vocab,
        device=device
    )
    local_victim_model.eval()

    # build local agent
    local_agent = Agent(
        local_victim_model,
        **agent_configs["attacker_model_configs"]
    )
    local_agent.eval()  # evaluation model pulls from global

    local_step = 0
    print("logging thread init")
    while True:  # infinite episodes 
        try:
            with lock:
                local_agent.sync_from(global_agent)
                # save the current global model
                if global_saver and local_step % (attack_configs["save_freq"])==0:
                    global_saver.save(global_step=global_update_step.value, model=global_agent)
                    print("save at %d"%global_update_step.value)
                    global_update_step.value+=1
            time.sleep(10)
            local_step += 1
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("OOM, skip batch...")
                continue
            else:
                raise e



def run():
    """
    multi-agent reinforcement learning with distributed optimization.
    local updated models are episodically soft-updated to the global_model.
    """
    # multiprocess preparation: default agent training thread as 1
    process = []
    global_update_step = mp.Value("i", 0)  # global update step 
    lock = mp.Lock()

    # initiate configs files and saver
    args = parser.parse_args()
    if not os.path.exists(args.save_to):
        args.reload = False
        os.mkdir(args.save_to)
    # load configs
    if args.reload:  # from ckpt if there is any, else default
        if os.path.exists(os.path.join(args.save_to, "current_attack_configs.yaml")):
            print("reload configs from ckpt")
            with open(os.path.join(args.save_to, "current_attack_configs.yaml"), "r") as f:
                configs = yaml.safe_load(f)
        else:
            with open(args.config_path, "r") as f:
                configs = yaml.safe_load(f)
    else:  
        with open(args.config_path, "r") as f:
            configs = yaml.safe_load(f)

    attack_configs = configs["attack_configs"]
    with open(attack_configs["victim_configs"], "r", encoding="utf-8") as victim_config_f:
        victim_configs = yaml.safe_load(victim_config_f)
    global_saver = Saver(save_prefix="{0}.ckpt".format(os.path.join(args.save_to, "ACmodel")),
        num_max_keeping=attack_configs["num_kept_checkpoints"])
    # dump the current configs as ckpt
    with open(os.path.join(args.save_to, "current_attack_configs.yaml"), "w") as current_configs_f:
        yaml.safe_dump(configs, current_configs_f)

    # load vocab by the victim trianing config
    # initiate iterator padded with initiated length
    src_vocab_file=os.path.join(victim_configs["training"]["model_dir"], "src_vocab.txt")
    trg_vocab_file=os.path.join(victim_configs["training"]["model_dir"], "trg_vocab.txt")
    if "code_file_path" in attack_configs:  # default joeynmt vocab 
        src_vocab = Vocabulary(file=src_vocab_file)
        trg_vocab = Vocabulary(file=trg_vocab_file)
    elif "plm_path" in attack_configs:  # plm vocab (BART)
        src_vocab = BartVocab(file=src_vocab_file)
        trg_vocab = BartVocab(file=trg_vocab_file)

    # build and load victim translator
    with open(attack_configs["victim_configs"], "r", encoding="utf-8") as victim_config_f:
        victim_configs = yaml.safe_load(victim_config_f)
    victim_model_path=attack_configs["victim_model"]
    victim_translator = build_translate_model(
        victim_configs=victim_configs, victim_model_path=victim_model_path,
        src_vocab=src_vocab, trg_vocab=trg_vocab,
        device="cpu"
    )
    victim_translator.eval()

    # load and extract nearest candidates by given victim NMT
    word2near_vocab = load_or_extract_near_vocab(
        victim_translator, 
        save_to=os.path.join(args.save_to, "near_vocab"), 
        batch_size=100, top_reserve=attack_configs["knn"],
        reload=True, emit_as_id=True, all_with_unk=True
    )

    # build a global adversarial model
    agent_configs = configs["agent_configs"]
    global_agent = Agent(
        victim_translator,
        **agent_configs["attacker_model_configs"])
    if args.reload:  # reload from the save_to path if there is any
        if os.path.exists(os.path.join(args.save_to, "ACmodel.ckpt")):
            with open(os.path.join(args.save_to, "ACmodel.ckpt")) as ckpt_file:
                ckpt_file_name = ckpt_file.readlines()[0]
                print("load ckpt from ", ckpt_file_name)
            agent_ckpt = torch.load(os.path.join(args.save_to, ckpt_file_name), map_location="cpu")
            global_agent.load_state_dict(agent_ckpt["model"])
            global_update_step.value=int(ckpt_file_name.split(".")[-1])+1
    global_agent.share_memory()
    
    # # test thread initialization with local agents and environments
    # train_thread(0, "cuda:0", args, global_update_step, lock, 
    #     attack_configs, configs["discriminator_configs"], 
    #     victim_translator, word2near_vocab=word2near_vocab,
    #     global_agent=global_agent, agent_configs=configs["agent_configs"], 
    #     global_saver=global_saver)
    # valid_thread("cuda:0", args, global_update_step, lock, 
    #     attack_configs, configs["discriminator_configs"], 
    #     victim_translator, word2near_vocab=word2near_vocab,
    #     global_agent=global_agent, agent_configs=configs["agent_configs"], 
    #     global_saver=global_saver)

    # build multi thread training and validation
    process = []
    for rank in range(args.n):
        print("initialize training thread on cuda:%d" % (rank+1))
        p=mp.Process(
            target=train_thread,
            args=(rank, "cuda:%d"%(rank+1), args, global_update_step, lock,
                attack_configs, configs["discriminator_configs"], 
                victim_translator, word2near_vocab,
                global_agent, configs["agent_configs"], 
                global_saver)
        )
        p.start()
        process.append(p)
    # run the dev or logging thread for initiation
    print("initialize logging thread on cuda:0")
    p = mp.Process(
        target=logging_thread,
        args=("cuda:0", args, global_update_step, lock, 
            attack_configs, configs["discriminator_configs"], 
            victim_translator, word2near_vocab,
            global_agent, configs["agent_configs"], 
            global_saver)
    )
    p.start()
    process.append(p)
    for p in process:
        p.join()

if __name__=="__main__":
    run()
    # APItest()
