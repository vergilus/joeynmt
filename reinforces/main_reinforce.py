# coding: utf-8
import argparse
import torch
import torch.multiprocessing as _mp
import os
import yaml
import time
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Normal
from reinforces.trans_env import Translate_Env
from reinforces.reinforce_utils import *
from reinforces.agent import Agent
from joeynmt.model import Model

from joeynmt.data import load_data
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
    default="./reinforces/reinforce_configs/reinforce_cwmt_zhen.yaml",
    help="path to reinforcement config")
parser.add_argument("--save_to", type=str, 
    default="./reinforces/reinforce_cwmt_log",
    help="save model and logging")
parser.add_argument("--max_episode", type=int, default=500000,
    help="maximum training episodes")
parser.add_argument("--reload", action="store_true", default=False,
    help="Whether to reload ckpt from the --save_to path")
parser.add_argument("--use_gpu", action="store_true", default=False,
    help="whether to use GPU (default as False).")

def training_thread(
        rank, device, args, global_update_step:mp.Value, lock:mp.Lock, 
        reinforce_configs:dict, agent_configs:dict, 
        src_vocab:Vocabulary, trg_vocab:Vocabulary,
        global_agent:Agent,
        global_saver:Saver=None,
    ):
    """
    an on-policy A3C thread, collects the noising exploration for the global noiser, 
    updates local parameters and soft-updates the global episodically.
    training minimizes the bellman residual (TD learning) on a max-step rollout trajectory. 
    """
    # build and load victim translator locally and initiate to the env
    print("initiate train thread %d"%rank)
    with open(reinforce_configs["victim_configs"], "r", encoding="utf-8") as victim_config_f:
        victim_configs = yaml.safe_load(victim_config_f)
    victim_model_path=reinforce_configs["victim_model"]
    # local saver saves the environments
    local_summary_writer=SummaryWriter(log_dir=os.path.join(args.save_to, "noiser_env%d"%rank))
    # build local models: separate translator and token embeddings
    local_src_embed, local_victim_translator = build_embedding_and_NMT(
        victim_configs=victim_configs["model"], victim_model_path=victim_model_path,
        src_vocab=src_vocab, trg_vocab=trg_vocab,
        device=device
    )
    local_src_embed.eval()
    local_victim_translator.eval()

    print("building dataset")
    dataset, _, _, _, _ = load_data(
        data_cfg=victim_configs["data"], datasets=["train"])

    # build local noiser
    print("building local noiser:")
    # policy action(std) range
    if "action_range" in agent_configs["agent_model_configs"]:
        action_range = agent_configs["agent_model_configs"]["action_range"]
        local_agent = Agent(
            embedding_dim=victim_configs["model"]["encoder"]["embeddings"]["embedding_dim"],
            **agent_configs["agent_model_configs"]).to(device)
    else:  # determined by src embeddings (embedding layer outputs)
        action_range = global_agent.action_range
        local_agent = Agent(
            embedding_dim=victim_configs["model"]["encoder"]["embeddings"]["embedding_dim"],
            action_range=action_range,
            **agent_configs["agent_model_configs"]).to(device)

    # build local environments with local victim model and saver
    local_env = Translate_Env(
        reinforce_configs, local_src_embed, 
        local_victim_translator, data_set=dataset,
        save_to=os.path.join(args.save_to, "noiser_env%d"%rank), 
        device=device)

    # build optimizers for the agent.
    local_agent_optim = build_optimizer(
        config=agent_configs["agent_optimizer_configs"],
        parameters=local_agent.parameters())
    
    local_clip_fn = build_gradient_clipper(
        config=agent_configs["agent_optimizer_configs"]
    )
    local_scheduler, local_schedule_at = build_scheduler(
        config=agent_configs["agent_optimizer_configs"],
        optimizer=local_agent_optim, scheduler_mode="min")

    # main training loop with infinite env (batch) initialization.
    local_step=0   # update step count for the noiser
    if args.reload:
        local_step=global_update_step.value+1
    episode_count=0   # each episode completes perturbation on a batch of full sequence
    while True:   # infinite training batch of sentences
        try:
            # sync from the global noiser 
            print("new batch init..")
            local_env.reset_state()
            s_0, _= local_env.get_state()
            cumprod_a_t = torch.ones_like(s_0)  # noise scales for consistency training, stores \cumprod(a_t) where a_t = 1-std_t^2
            with lock:
                local_agent.sync_from(global_agent)
            perturbation_round = 0 
            sample_round = np.random.randint(1,10)
            while not local_env.terminated():
                ## save checkpoint, saving timed by rank
                # if global_saver and (local_step+rank) % reinforce_configs["save_freq"]==0:
                #     with lock:
                #         global_saver.save(
                #             global_step=global_update_step.value, 
                #             agent=global_agent,
                #             )
                # ============= train local noiser & perturbation rounds  ====================
                # loop for a section of sequential policy for on-policy update
                local_agent.train()
                local_agent.to(device)
                V_estimates = []  # records the on-policy V-function (critic)
                log_probs = []
                entropies = []
                rewards = []
                reconstruct_loss = 0  # collect reconstruct loss along the rollout if necessary
                for _ in range(local_agent.action_roll_steps):
                    perturbation_round+=1  # noising round starts from 1
                    s_t, _ = local_env.get_state()
                    next_s_t, alpha_t, act_log_prob, act_entropies = local_agent.sample_noise(
                        s_t, local_env.src_mask, local_env.src_length,
                    )  # train by exploration
                    current_cumprod_a_t = cumprod_a_t
                    next_cumprod_a_t = cumprod_a_t * alpha_t.detach()
                    V = local_agent.get_V(
                        s_t, local_env.src_mask, local_env.src_length,
                    )  # [batch, len, 1] the state value function
                    step_reward = local_env.update_to(next_s_t, with_bleu=False, show_info=True).unsqueeze(dim=-1)
                    print(perturbation_round, "V %.5g"%V[0][0].item(), "r_t %.5g"%step_reward[0].item())

                    """ 
                    collect consistent-reconstruction loss bootstrapped by the global agent
                    on this transition, the one with more V value is the supervision
                    """
                    st_noise_scale = torch.sqrt(1-current_cumprod_a_t)
                    next_st_noise_scale = torch.sqrt(1-next_cumprod_a_t)
                    if perturbation_round%50 == sample_round:
                        x_t = local_agent.noise_by_scale(
                            s_0, local_env.src_mask, local_env.src_length, st_noise_scale
                        ).detach()
                        next_x_t = local_agent.noise_by_scale(
                            s_0, local_env.src_mask, local_env.src_length, next_st_noise_scale
                        ).detach()
                        reconstruct_by_next_st = local_agent.denoise(
                            next_x_t, local_env.src_mask, local_env.src_length,
                            next_st_noise_scale)         
                        global_agent.to(device)
                        
                        supervise_by_st = global_agent.denoise(
                            x_t, local_env.src_mask, local_env.src_length,
                            st_noise_scale).detach()
                        local_std= local_agent.get_std(s_t, local_env.src_mask, local_env.src_length)
                        global_std = global_agent.get_std(s_t, local_env.src_mask, local_env.src_length)
                        ratio1=local_std/global_std
                        ratio2=global_std/local_std
                        varrho = torch.cat([ratio1.unsqueeze(dim=-1),ratio2.unsqueeze(dim=-1)],dim=-1).min(dim=-1)[0].detach()
                        global_agent.to("cpu")
                        supervised_by_s0 = s_0+local_agent.min_std*torch.randn_like(s_0)
                        reconstruct_loss += 0.5*((varrho * (reconstruct_by_next_st- supervise_by_st.detach()))**2 \
                                        ).sum(dim=-1, keepdims=True)  # + (reconstruct_by_next_st-supervised_by_s0.detach())**2 

                    cumprod_a_t = next_cumprod_a_t
                    # record rolled trajectory
                    V_estimates.append(V)  # [batch, len, 1]
                    log_probs.append(act_log_prob.sum(dim=-1, keepdims=True))  # [batch, len, 1]
                    entropies.append(act_entropies)  # [batch, len, 1]
                    rewards.append(step_reward)  # [batch, 1]
                    # print(V.shape, act_log_prob.shape, act_entropies.shape, step_reward.shape)
                    if local_env.terminated():               
                        # if prematurely finish trajectory accumulation
                        break  
                # print(local_env.view_token(src_vocab=src_vocab))
                R = torch.zeros_like(V)  # accumulated rewards on trajectory (training trg for critic)
                if not local_env.terminated():  
                    # bootstrap by V(s_t) as reward
                    s_t, terminal_flags = local_env.get_state()
                    boot_strap_V = local_agent.get_V(
                        s_t, local_env.src_mask, local_env.src_length,
                        # timestep=timestep,
                    ) # [batch,len,1] state value function
                    R = boot_strap_V
                V_estimates.append(R)  # [batch]
                gae = torch.zeros_like(V)  # advantage value
                policy_loss = torch.zeros_like(V)
                value_loss = torch.zeros_like(V)  # all in shape: [batch, len, 1]
                
                # accumulate policy and value loss
                for i in reversed(range(len(rewards))):
                    R = reinforce_configs["gamma"] * R + rewards[i].unsqueeze(dim=1)  # rewards shared along sen_len
                    reward_mse = R - V_estimates[i]
                    # accumulated value loss
                    value_loss = value_loss + 0.5 * reward_mse.pow(2)  # [batch, len, 1]
                    # accumulated advantage value with discounted future rewards on trajectory
                    advantage = rewards[i].unsqueeze(dim=1) + reinforce_configs["gamma"]*V_estimates[i+1] - V_estimates[i]
                    gae = gae*reinforce_configs["gamma"]+ advantage  # general advantage [batch, len, 1]
                    # accumualted policy loss on trajectory with maximum entropy regularization
                    policy_loss = policy_loss - \
                        log_probs[i]*gae.detach() - \
                            reinforce_configs["entropy_coef"]*entropies[i]
                
                # mean loss updates over this batch of sentences and length dimension
                # print(policy_loss.shape, value_loss.shape, reconstruct_loss.shape)
                rl_loss = policy_loss+reinforce_configs["value_coef"]*value_loss 
                total_loss = rl_loss + reconstruct_loss
                # mean over valid parts
                total_loss = total_loss.mean()
                total_loss.backward()
                
                if local_clip_fn is not None:
                    local_clip_fn(params=local_agent.parameters())
                local_agent_optim.step()
                if local_scheduler is not None and local_schedule_at=="step":
                    local_scheduler.step()
                local_agent_optim.zero_grad()
                local_step += 1 
                local_summary_writer.add_scalar("lr", scalar_value=local_agent_optim.param_groups[0]["lr"], global_step=local_step)
                local_summary_writer.add_scalar("rl/L_noise", scalar_value=rl_loss.mean(), global_step=local_step)
                local_summary_writer.add_scalar("rl/L_noise_p", scalar_value=policy_loss.mean(), global_step=local_step)
                local_summary_writer.add_scalar("rl/L_noise_v", scalar_value=value_loss.mean(), global_step=local_step)
                if "torch" in str(type(reconstruct_loss)):   # can be zero or a tensor!
                    local_summary_writer.add_scalar("diffusion/L_reconst", scalar_value=reconstruct_loss.mean(), global_step=local_step)
                    reconstruct_v= 0.5*((reconstruct_by_next_st -s_0.detach())**2).sum(dim=-1).mean()
                    local_summary_writer.add_scalar("diffusion/deviate_x_0", scalar_value=reconstruct_v, global_step=local_step)
                with lock:
                    # soft update global model (exponential moving average parameters)
                    print("global updated by thread %d"%rank)
                    global_update_step.value += 1
                    local_agent.to("cpu")
                    for (name,param), target_param in zip(local_agent.named_parameters(), global_agent.parameters()):
                        target_param.data.copy_(reinforce_configs["tau"] * target_param.data+(1-reinforce_configs["tau"])* param.data)
                    local_agent.to(device)

            if local_scheduler is not None and local_schedule_at=="epoch":
                local_scheduler.step()
            local_agent_optim.zero_grad()
            episode_count += 1
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("OOM, skip batch...")
                local_agent_optim.zero_grad()
                continue
            else:
                raise e

        if episode_count>args.max_episode:
            return


def denoise_thread(
        device, args, global_update_step:mp.Value, lock:mp.Lock, 
        reinforce_configs:dict, agent_configs:dict, 
        src_vocab:Vocabulary, trg_vocab:Vocabulary,  
        global_agent:Agent,
        global_saver:Saver=None,
    ):
    """
    training consistency function (one-step denoising) given a x_t and denoising scale
    """
    print("initiate denoiser thread")
    with open(reinforce_configs["victim_configs"], "r", encoding="utf-8") as victim_config_f:
        victim_configs = yaml.safe_load(victim_config_f)
    victim_model_path=reinforce_configs["victim_model"]
    # local saver saves the environments
    local_summary_writer=SummaryWriter(log_dir=os.path.join(args.save_to, "denoiser_env"))
    # build local models: separate translator and token embeddings
    local_src_embed, local_victim_translator = build_embedding_and_NMT(
        victim_configs=victim_configs["model"], victim_model_path=victim_model_path,
        src_vocab=src_vocab, trg_vocab=trg_vocab,
        device=device
    )
    local_src_embed.eval()
    local_victim_translator.eval()

    # print("building dataset")
    dataset, _, _, _, _ = load_data(
        data_cfg=victim_configs["data"], datasets=["train"])

    # build local denoiser
    print("building valid agent:")
    if "action_range" in agent_configs["agent_model_configs"]:
        action_range = agent_configs["agent_model_configs"]["action_range"]
        local_agent = Agent(
             embedding_dim=victim_configs["model"]["encoder"]["embeddings"]["embedding_dim"],
            **agent_configs["agent_model_configs"]).to(device)
    else:  # determined by src embeddings
        action_range = global_agent.action_range
        local_agent = Agent(
            embedding_dim=victim_configs["model"]["encoder"]["embeddings"]["embedding_dim"],
            action_range=action_range,
            **agent_configs["agent_model_configs"]).to(device)
    
    # build local environments with local victim model and saver
    local_env = Translate_Env(
        reinforce_configs, local_src_embed, 
        local_victim_translator, data_set=dataset,
        save_to=os.path.join(args.save_to, "denoiser_env"), 
        device=device)
    
    # build optimizers for the global agent's denoising.
    local_optim = build_optimizer(
        config=agent_configs["agent_optimizer_configs"],
        parameters=local_agent.parameters())
    local_clip_fn = build_gradient_clipper(
        config=agent_configs["agent_optimizer_configs"]
    )
    local_scheduler, local_schedule_at = build_scheduler(
        config=agent_configs["agent_optimizer_configs"],
        optimizer=local_optim, scheduler_mode="min")

    # the learnt global transition (noising) distribution q(next_x_t|x_t)
    # trains the reverse transition (denoising) distribution p(x_t|next_x_t) 
    local_agent.train()
    local_step=0
    if args.reload:
        local_step=global_update_step.value+1
    while True:  # infinite batch of sentences 
        """
        loop the batch, training consistency reconstruction once every batch
        """
        try:
            local_env.reset_state()
            with lock:
                local_agent.sync_from(global_agent)
            # save the current global model
            if global_saver and local_step % (reinforce_configs["save_freq"]*args.n)==0:
                with lock:
                    global_saver.save(
                        global_step=global_update_step.value, 
                        agent=global_agent,
                    )
                    
            s_0, _ = local_env.get_state()
            next_s_t, alpha_t, act_log_prob, act_entropies = local_agent.sample_noise(
                s_0, local_env.src_mask, local_env.src_length
            )
            s_0 = s_0.detach()
            next_s_t = next_s_t.detach()
            noise_scale = torch.sqrt(1-alpha_t)  # the std of noising
            # diffusion increments with reverse distribution KL divergence
            s_minor = local_agent.noise_by_scale(
                s_0, local_env.src_mask, local_env.src_length, 
                local_agent.min_std*torch.ones_like(alpha_t)
            ).detach()

            reconstruct_by_next_st = local_agent.denoise(
                next_s_t, local_env.src_mask, local_env.src_length,
                noise_scale
            )
            supervised_by_s_minor = local_agent.denoise(
                s_minor, local_env.src_mask, local_env.src_length,
                local_agent.min_std*torch.ones_like(alpha_t)
            ).detach()
            
            reconstruct_loss = (0.5*(reconstruct_by_next_st - supervised_by_s_minor)**2).sum(dim=-1).mean()
            reconstruct_loss.backward()
            
            if local_clip_fn is not None:
                local_clip_fn(params=local_agent.parameters()) 
            local_optim.step()
            if local_scheduler is not None and local_schedule_at=="step":
                local_scheduler.step()
            local_optim.zero_grad()
            local_step += 1    
            local_summary_writer.add_scalar("lr", scalar_value=local_optim.param_groups[0]["lr"], global_step=local_step)
            local_summary_writer.add_scalar("diffusion/L_reconst", scalar_value=reconstruct_loss,global_step=local_step)
            time.sleep(10)
            # perturbed_bleu_list = local_env.get_current_bleu()
            # # check the noise rate of the current perturbation policy:
            # noising_count=0.
            # for i in range(len(start_bleu_list)):
            #     if perturbed_bleu_list[i]<start_bleu_list[i]:
            #         noising_count+=1
            # noising_rate = noising_count/len(perturbed_bleu_list)
            # print("noising rate: %.2f"%noising_rate)
            # local_summary_writer.add_scalar("noising rate", 
            #     scalar_value=noising_rate, global_step=local_step)
            # if noising_rate==0:
            #     zero_bleu_count+=1
            #     if zero_bleu_count>5:
            #         break
            
            with lock:
                # soft update denoiser to global model (exponential moving average parameters)
                print("global denoiser updates")
                local_agent.to("cpu")
                for (name,param), target_param in zip(local_agent.named_parameters(), global_agent.parameters()):
                    if name in local_agent.denoise_param_names:
                        target_param.data.copy_(reinforce_configs["tau"] * target_param.data+(1-reinforce_configs["tau"])* param.data)
                
                if global_saver and local_step % (reinforce_configs["save_freq"])==0:
                    global_saver.save(
                        global_step=global_update_step.value, 
                        agent=global_agent)
                    print("save at %d"%global_update_step.value)
                global_update_step.value += 1
                local_agent.to(device)

        except RuntimeError as e:
            if "out of memory" in str(e):
                print("OOM, skip batch...")
                continue
            else:
                raise e

def logging_thread(
        device, args, global_update_step:mp.Value, lock:mp.Lock, 
        reinforce_configs:dict, agent_configs:dict, 
        src_vocab:Vocabulary, trg_vocab: Vocabulary,  
        global_agent:Agent, 
        global_saver:Saver=None,
    ):
    print("init logging thread")
    with open(reinforce_configs["victim_configs"], "r", encoding="utf-8") as victim_config_f:
        victim_configs = yaml.safe_load(victim_config_f)

    # build local agent
    if "action_range" in agent_configs["agent_model_configs"]:
        action_range = agent_configs["agent_model_configs"]["action_range"]
        local_agent = Agent(
            embedding_dim=victim_configs["model"]["encoder"]["embeddings"]["embedding_dim"],
            **agent_configs["agent_model_configs"]).to(device)
    else:  # determined by src embeddings
        action_range = global_agent.action_range
        local_agent = Agent(
            embedding_dim=victim_configs["model"]["encoder"]["embeddings"]["embedding_dim"],
            action_range=action_range,
            **agent_configs["agent_model_configs"]).to(device)
    local_agent.eval()  # evaluation model pulls from global
    local_step = 0
    while True:  # infinite batch of sentences
        try:
            with lock:
                local_agent.sync_from(global_agent)
                # save the current global model
            if global_saver and local_step % (reinforce_configs["save_freq"])==0:
                with lock:
                    global_saver.save(
                        global_step=global_update_step.value, 
                        agent=global_agent)
                    print("save at %d"%global_update_step.value)
                    global_update_step.value+=1
            time.sleep(60)
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

    # initiate ckpt paths for files and saver
    args = parser.parse_args()
    if not os.path.exists(args.save_to):
        args.reload=False  
        os.mkdir(args.save_to)

    if args.reload and os.path.exists(os.path.join(args.save_to, "current_reinforce_configs.yaml")): # load from ckpt configs
        with open(os.path.join(args.save_to, "current_reinforce_configs.yaml"), "r") as f:
            configs = yaml.safe_load(f)
    else:
        with open(args.config_path, "r") as f:  # load RL configs
            configs = yaml.safe_load(f)

    reinforce_configs = configs["reinforce_configs"]
    with open(reinforce_configs["victim_configs"], "r", encoding="utf-8") as victim_config_f:
        victim_configs = yaml.safe_load(victim_config_f)
    global_saver = Saver(save_prefix="{0}.ckpt".format(os.path.join(args.save_to, "ACmodel")),
        num_max_keeping=reinforce_configs["num_kept_checkpoints"])

    # load vocab by the victim trianing config
    # initiate iterator padded with initiated length
    src_vocab_file=os.path.join(victim_configs["training"]["model_dir"], "src_vocab.txt")
    trg_vocab_file=os.path.join(victim_configs["training"]["model_dir"], "trg_vocab.txt")
    src_vocab = Vocabulary(file=src_vocab_file)
    trg_vocab = Vocabulary(file=trg_vocab_file)

    # build and load victim translator with separate embeddings
    with open(reinforce_configs["victim_configs"], "r", encoding="utf-8") as victim_config_f:
        victim_configs = yaml.safe_load(victim_config_f)
    victim_model_path=reinforce_configs["victim_model"]
    if args.use_gpu:
        device="cuda:0"
    else:
        device="cpu"
    src_embed, victim_translator = build_embedding_and_NMT(
        victim_configs=victim_configs["model"], victim_model_path=victim_model_path,
        src_vocab=src_vocab, trg_vocab=trg_vocab,
        device=device
    )
    src_embed.eval()
    victim_translator.eval()

    # load and extract nearest candidates of given victim NMT for action_space and norm ball limits
    # build global agent
    agent_configs = configs["agent_configs"]
    if "action_range" in agent_configs["agent_model_configs"]:
        action_range = agent_configs["agent_model_configs"]["action_range"]
    else:  # extract action range from src embeddings
        action_range = extract_action_space(src_embed, device=device)
        agent_configs["agent_model_configs"]["action_range"] = action_range
        
    global_agent = Agent(
        embedding_dim=victim_configs["model"]["encoder"]["embeddings"]["embedding_dim"],
        **agent_configs["agent_model_configs"])
    
    if args.reload:  # reload from the save_to path
        if os.path.exists(os.path.join(args.save_to, "ACmodel.ckpt")):
            with open(os.path.join(args.save_to, "ACmodel.ckpt")) as ckpt_file:
                ckpt_file_name = ckpt_file.readlines()[0]
                print("load ckpt from ", ckpt_file_name)
            agent_ckpt = torch.load(os.path.join(args.save_to, ckpt_file_name), map_location="cpu")
            global_agent.load_state_dict(agent_ckpt["agent"])
            global_update_step.value=int(ckpt_file_name.split(".")[-1])+1

    global_agent.share_memory()
    print("perturb range against victim model:",action_range)
    # log RL configs for ckpt
    with open(os.path.join(args.save_to, "current_reinforce_configs.yaml"), "w") as current_configs:
        yaml.safe_dump(configs, current_configs)

    # """ test thread initialization with local agents and environments
    # initiate training and valid threads of on-policy A3C """
    training_thread(0, "cuda:0", args, global_update_step, lock, 
        reinforce_configs, agent_configs=configs["agent_configs"],
        src_vocab=src_vocab, trg_vocab=trg_vocab,
        global_agent=global_agent, 
        global_saver=global_saver)

    # denoise_thread("cuda:0", args, global_update_step, lock, 
    #     reinforce_configs,  agent_configs=configs["agent_configs"],  
    #     src_vocab=src_vocab, trg_vocab=trg_vocab,
    #     global_agent=global_agent,
    #     global_saver=global_saver)

    # # build multi thread training and validation
    # process = []
    # for rank in range(args.n):
    #     print("initialize noising thread on cuda:%d" % (rank+1))
    #     p=mp.Process(
    #         target=training_thread,
    #         args=(rank, "cuda:%d"%(rank+1), args, global_update_step, lock, 
    #             reinforce_configs, configs["agent_configs"],
    #             src_vocab, trg_vocab,
    #             global_agent,
    #             global_saver)
    #     )
    #     p.start()
    #     process.append(p)
    # # run the dev thread for initiation
    # print("initialize denoising thread on cuda:0")
    # p = mp.Process(
    #     target=denoise_thread,
    #     args=("cuda:0", args, global_update_step, lock, 
    #     reinforce_configs,  configs["agent_configs"],  
    #     src_vocab, trg_vocab,
    #     global_agent,
    #     global_saver)
    # )
    # p.start()
    # process.append(p)
    # for p in process:
    #     p.join()

 
if __name__=="__main__":
    run()
    # APItest()
