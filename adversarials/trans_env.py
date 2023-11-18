import os
import torch
import torch.nn as nn 
import numpy as np
import yaml
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torchtext.legacy.data import Dataset, Iterator
from adversarials.adversarial_utils import *
from adversarials.agent import Agent 
from adversarials.discriminator import TransDiscriminator
import codecs
from joeynmt.model import Model
from joeynmt.constants import UNK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN
from joeynmt.builders import build_optimizer, build_gradient_clipper, build_scheduler
from joeynmt.search import beam_search
from joeynmt.helpers import bpe_postprocess
from sacrebleu.metrics import BLEU, CHRF, TER
from subword_nmt import apply_bpe
from transformers import MBart50Tokenizer

class Translate_Env(object):
    """
    wrap the translator as environment for reinforcement learing
    provides state transition dynamics and step rewards 
    translator and discriminator
    
    :return: states, and corresponding rewards on batched sequences.
    """
    def __init__(self,
            attack_configs:dict, discriminator_configs:dict, 
            victim_model:Model, data_set:Dataset, word2near_vocab:dict,
            save_to:str, device:str
        ) -> None:
        """
        :param victim_model: receives a locally-built victim model in thread
        :param save_to: the discriminator model saving
        """
        self.victim_model = victim_model
        assert self.victim_model.training==False, "victim must be validation mode" 
        self.PAD_id = self.victim_model.src_vocab.stoi[PAD_TOKEN]
        self.word2near_vocab = word2near_vocab
        self.save_to = save_to
        self.device = device
        self.BPE_dropout = 0.2

        # rewards
        self.adversarial = attack_configs["adversarial"]  # adversarial rewards
        self.r_s_weight = attack_configs["r_s_weight"]
        self.r_d_weight = attack_configs["r_d_weight"]
        self.metric = BLEU()  # episodic rewards by BLEU
        self.batch_size = attack_configs["batch_size"]  # num of sentences 
        # retokenization and translation, tokenization skips the reserved tokens
        if "code_file_path" in attack_configs:
            self.toker = apply_bpe.BPE(
                codes=codecs.open(attack_configs["code_file_path"], encoding='utf-8'),
                glossaries=[BOS_TOKEN,EOS_TOKEN,PAD_TOKEN,UNK_TOKEN],
            )
        elif "plm_path" in attack_configs:
            self.toker = MBart50Tokenizer.from_pretrained(attack_configs["plm_path"])
        else:
            print("lacks tokenizer models, require code_file_path or plm_path")
            return
        
        if "pinyin_data" in attack_configs:
            # recommend saving in local paths
            with open(attack_configs["victim_configs"], "r") as v_f:
                victim_configs = yaml.safe_load(v_f)
                if os.path.exists(os.path.join(self.save_to, "../char2py.dict")) and \
                        os.path.exists(os.path.join(self.save_to, "../py2char.dict")):
                    print("loading pinyin:")
                    self.char2pyDict = yaml.safe_load(os.path.join(self.save_to, "../char2py.dict"))
                    self.py2charDict = yaml.safe_load(os.path.join(self.save_to, "../py2char.dict"))
                else:
                    print("collect pinyin data for gen_UNK, this would take a while..")
                    self.char2pyDict, self.py2charDict = collect_pinyin(
                        pinyin_path=attack_configs["pinyin_data"],
                        src_path=victim_configs["data"]["train"] + ".zh" ) 
                    with open(os.path.join(self.save_to, "../char2py.dict"), "w", encoding="utf-8") as charfile:
                        yaml.safe_dump(self.char2pyDict, charfile)
                    with open(os.path.join(self.save_to, "../py2char.dict"), "w", encoding="utf-8") as pyfile:
                        yaml.safe_dump(self.py2charDict, pyfile)    
        else:
            self.char2pyDict, self.py2charDict = None, None

        # build local iterator (with repeat as true for infinite iteration and shuffle for epoch)
        self.local_iter = iter(Iterator(
            repeat=True, sort=False, dataset=data_set,
            batch_size=self.batch_size, batch_size_fn=None, 
            train=True, sort_within_batch=True,
            sort_key=lambda x: len(x.src), shuffle=True
        ))
        # build local saver which saves the environment D
        self.local_saver = Saver(
            save_prefix="{0}.final".format(os.path.join(self.save_to, "D")),
            num_max_keeping=1
        )

        # build discriminator (sentence matching with hard-binary Xentloss)
        D_model_configs = discriminator_configs["discriminator_model_configs"]
        self.acc_bound = discriminator_configs["acc_bound"]
        self.discriminator = TransDiscriminator(
            self.victim_model,
            **D_model_configs
        ).to(self.device)
        self.loss_D = nn.CrossEntropyLoss()
        
        # build discriminator optimizer, minimizing the 
        D_optim_configs = discriminator_configs["discriminator_optimizer_configs"]
        self.optim_D = build_optimizer(
            config=D_optim_configs, 
            parameters=self.discriminator.parameters()
        )
        self.clip_grad_fn = build_gradient_clipper(config=D_optim_configs)
        self.scheduler_D, self.schedule_step_at = build_scheduler(
            config=D_optim_configs,
            scheduler_mode="min",
            optimizer=self.optim_D
        )                 

        self.index = 1  # records the position to be perturbed
        self.reset_state()
    
    def reset_D(self,):
        self.discriminator._reset_parameters()

    def reset_state(self,):
        """
        initiate environment with bathced sentences and a BLEU scorer
        including original BLEU score.
        returns the batch, and index
        """
        self.index = 1  # reset the position indicator for perturbation to the first
        while True:
            batch = next(self.local_iter)
            batch = MyBatch(batch, 
                pad_index=self.PAD_id, device=self.device)
            if batch.nseqs>1 and check_unk_ratio(self.victim_model.src_vocab,batch.src)<0.1: # too many unk or not enough sampels
                break
       
        self.src_length = batch.src_length   # Tensor object, count w/o BOS, w EOS
        self.terminal_signals = torch.zeros_like(batch.src_length)  
        # the generated batch is padded, append BOS for environment
        self.BOS_id = self.victim_model.src_vocab.stoi[BOS_TOKEN]
        BOS_vec = self.BOS_id * batch.src.new_ones(batch.src.shape[0]).unsqueeze(dim=-1)
        
        self.origin_padded_src = torch.cat([BOS_vec, batch.src], dim=-1)  # padded with BOS token
        self.padded_src = torch.cat([BOS_vec, batch.src], dim=-1)
        # print(self.padded_src[0])  # note that the joeynmt translator (default) does not take BOS as inputs!
        self.src_mask = batch.src_mask 
        self.trg = batch.trg
        self.trg_length = batch.trg_length
        self.trg_mask = batch.trg_mask
        self.origin_bleu=[]
        decode_result = self.translate()  # translate current batch init origin bleu
        # transit from id to bpe tokens
        decoded_valid = self.victim_model.trg_vocab.arrays_to_sentences(decode_result, cut_at_eos=True)
        reference_valid = self.victim_model.trg_vocab.arrays_to_sentences(self.trg, cut_at_eos=True)
        decoded_valid =[" ".join(t) for t in decoded_valid] 
        reference_valid = [" ".join(t) for t in reference_valid] 
        # transit from bpe tokens to seqs for BLEU
        hyps = [bpe_postprocess(s, "subword-nmt") for s in decoded_valid]
        refs = [bpe_postprocess(s, "subword-nmt") for s in reference_valid]
        for i, hyp in enumerate(hyps):
            bleu_t=0.01*self.metric.corpus_score([hyp], [[refs[i]]]).score
            self.origin_bleu.append(bleu_t)
        return

    def segment_tokens(self, line, dropout=0.0):
        """
        return a list of segmented subwords 
        """
        if "BPE" in self.toker.__class__.__name__:
            return self.toker.segment_tokens(line, dropout=dropout)
        elif "MBart" in self.toker.__class__.__name__ :
            seg_ids = self.toker.encode(line)[1:-1]
            return self.toker.convert_ids_to_tokens(seg_ids)

    def retokenize(self,origin_src:Tensor, perturbed_src:Tensor, dropout:float=0):
        """
        retokenize the given src, if none, retokenize env src
        ignores the src BOS-id and valididate UNK tokens during the process
        ignores the original UNK token
        :param origin_src: input ids [batch, len] , usually does not include BOS in translation
        :param perturbed_src: must be with BOS (might be longer than origin_src)
        :param dropout: BPE-dropout rate during segmentation
        :return: retokenized padded src_ids, src_length, src_mask in 
            the same style of the initial batching process (default without BOS)
        """
        if perturbed_src.shape==origin_src.shape:
            perturbed_src_list = perturbed_src.detach().cpu().numpy().tolist()
        else:
            perturbed_src_list = perturbed_src[:, 1:].detach().cpu().numpy().tolist() # remove BOS
        origin_src_list = origin_src.detach().cpu().numpy().tolist()
        # valid UNK tokens and detokenized sequence for another tokenization
        retokenized_lines = []
        for l_index in range(len(perturbed_src_list)):  # for each line of input token ids  
            raw_line= []
            for id_index in range(len(perturbed_src_list[l_index])):
                id = perturbed_src_list[l_index][id_index]
                origin_id = origin_src_list[l_index][id_index]
                if self.victim_model.src_vocab.itos[origin_id] not in [BOS_TOKEN, EOS_TOKEN, PAD_TOKEN]:
                    # ignore BOS EOS and PAD, transfer to valid token
                    if id ==self.victim_model.src_vocab.stoi[UNK_TOKEN] and origin_id!=self.victim_model.src_vocab.stoi[UNK_TOKEN]:
                        raw_line.append(
                            gen_UNK(src_token=self.victim_model.src_vocab.itos[origin_id],
                            vocab=self.victim_model.src_vocab, 
                            char2pyDict=self.char2pyDict, py2charDict=self.py2charDict)
                        )
                    else:
                        raw_line.append(self.victim_model.src_vocab.itos[id])
            # cleanse and tokenize the new_line, exclude UNK token
            # env tokenization implements BPE-dropout for segmentation robustness. 
            if "BPE" in self.toker.__class__.__name__:
                raw_line = " ".join(raw_line)
                new_line = bpe_postprocess(raw_line, "subword-nmt").split(" ")  # remove the separaters and re-split
            elif "MBart" in self.toker.__class__.__name__:
                new_line = "".join(raw_line).replace("▁", " ").strip()

            retokenized_line = self.segment_tokens(new_line, dropout=dropout)
            # transform to ids and append with EOS
            retokenized_ids = [self.victim_model.src_vocab.stoi[w] for w in retokenized_line] 
            retokenized_ids.append(self.victim_model.src_vocab.stoi[EOS_TOKEN])
            retokenized_lines.append(retokenized_ids)
        # batch process: generate src_length
        novel_src_length = [len(line) for line in retokenized_lines]  # with additional EOS
        novel_src_length = self.src_length.new_tensor(novel_src_length)

        # pad the retokenized src
        pad_len = int(max(novel_src_length.cpu().numpy()))
        batch_size = self.padded_src.shape[0]
        PAD_id = self.victim_model.src_vocab.stoi[PAD_TOKEN]
        novel_src = np.full((batch_size, pad_len), fill_value=PAD_id, dtype='int64')
        for k in range(batch_size):
            for kk in range(len(retokenized_lines[k])):
                novel_src[k][kk] = retokenized_lines[k][kk]
        novel_src = self.padded_src.new_tensor(novel_src)
        novel_src_mask = (novel_src != PAD_id).unsqueeze(1)
        return novel_src, novel_src_mask, novel_src_length

    def translate(self, src:Tensor=None, 
        src_mask:Tensor=None, src_length:Tensor=None, retokenize:bool=True):
        """
        fast translate the given tokenized inputs, if None, translate batch in the env
        returns the padded decoder ids
        :param retokenize: whether to retokenize before the translation, default as false
        """
        if src is None:
            src = self.padded_src[:, 1:]  # remove the BOS
            src_mask = self.src_mask
            src_length = self.src_length
        else:
            assert src[0][0] is not self.victim_model.src_vocab.stoi[BOS_TOKEN], \
            "default joeynmt victim does not take BOS for input"

        if retokenize:  # retokenize the input ids, without BPE-dropout
            src, src_mask, src_length = self.retokenize(src, src)  # does not include BOS

        with torch.no_grad():
            # print("translate on: ",src.device)
            max_output_length = int(max(src_length.cpu().numpy())*1.5)
            # encode
            encoder_output, encoder_hidden, _, _ = self.victim_model(
                return_type="encode", 
                src=src, src_length=src_length,src_mask=src_mask
            )
            # beam-search for the best results (stacked_out as np.array)
            stacked_out, _ = beam_search(
                model=self.victim_model, size=5, 
                encoder_output=encoder_output, encoder_hidden=encoder_hidden,
                src_mask=src_mask, max_output_length=max_output_length,
                alpha=1.0, n_best=1
            )
        return stacked_out

    def get_current_bleu(self):
        """
        translate current states and return current bleu score
        the BLEU is calculated by the orignal reference.
        return: a list of BLEU
        """
        decode_result = self.translate(retokenize=True)  # translate current batch init origin bleu
        # transit from id to bpe tokens
        decoded_valid = self.victim_model.trg_vocab.arrays_to_sentences(decode_result, cut_at_eos=True)
        reference_valid = self.victim_model.trg_vocab.arrays_to_sentences(self.trg, cut_at_eos=True)
        decoded_valid =[" ".join(t) for t in decoded_valid] 
        reference_valid = [" ".join(t) for t in reference_valid] 
        # transit from bpe tokens to seqs for BLEU
        hyps = [bpe_postprocess(s, "subword-nmt") for s in decoded_valid]
        refs = [bpe_postprocess(s, "subword-nmt") for s in reference_valid]
        current_bleu=[]
        for i, hyp in enumerate(hyps):
            bleu_t = 0.01 * self.metric.corpus_score([hyp], [[refs[i]]]).score
            current_bleu.append(bleu_t)
        return current_bleu
    
    def acc_validation(self, agent:Agent, random_inject_ratio:float, showNAN=False):
        """ validate environment's D by given agent """ 
        self.discriminator.eval()
        acc=0.0
        sample_count=0
        for i in range(5):
            # extract batch
            while True: # extract a valid batch
                batch = next(self.local_iter)
                batch = MyBatch(
                    batch, pad_index=self.PAD_id, device=self.device)
                if batch.nseqs>1 and check_unk_ratio(self.victim_model.src_vocab,batch.src)<0.1:
                    break
            perturbed_src, flags = agent.seq_attack(
                src=batch.src, src_mask=batch.src_mask, src_length=batch.src_length, 
                word2near_vocab=self.word2near_vocab,
                src_vocab=self.victim_model.src_vocab,
                random_inject_ratio=random_inject_ratio,
                half_mode=True)
            # validate UNK and retokenize perturbed src with BPE-dropout
            novel_src, _, novel_length = self.retokenize(batch.src, perturbed_src, dropout=self.BPE_dropout)
            if showNAN:
                print("flag:", flags[0].item(), random_inject_ratio)
                print("origin:", [self.victim_model.src_vocab.itos[i.item()] for i in batch.src[0] ])
                print("perturb:", [self.victim_model.src_vocab.itos[i.item()] for i in novel_src[0]])
            retok_src = torch.cat([self.BOS_id * novel_src.new_ones(novel_src.shape[0], 1), novel_src], dim=-1)            # perturbed_src follows the BOS padding style with the src 
            retok_mask = (retok_src != self.PAD_id).unsqueeze(1)
            retok_length = novel_length+1
            with torch.no_grad():
                preds = self.discriminator(
                    retok_src, retok_mask, retok_length,
                    batch.trg, batch.trg_mask, batch.trg_length).argmax(dim=-1)
                acc+=torch.eq(preds, flags).sum()
                sample_count+=preds.size(0)

        acc = acc.float()/sample_count
        return acc.item()

    def save(self, **kwargs):
        #  global_step=local_step, model=local_env.discriminator,
        #  optim=local_optim, scheduler=local_scheduler
        self.local_saver.save(**kwargs)

    def reload(self ):
        # reload trans states
        local_step=0
        if os.path.exists(os.path.join(self.save_to, "D.final")):
            with open(os.path.join(self.save_to, "D.final"), "r") as ckp_file:
                ckpt_file_name = ckp_file.readlines()[0]
                print("load env from ", ckpt_file_name)
            env_ckpt = torch.load(os.path.join(self.save_to, ckpt_file_name), map_location=self.device)
            
            self.discriminator.load_state_dict(env_ckpt["model"]) 
            # self.optim_D.load_state_dict(env_ckpt["optim"])
            # self.scheduler_D.load_state_dict(env_ckpt["scheduler"])
            local_step = int(ckpt_file_name.split(".")[-1])
        return local_step

    def update_discriminator(self, 
        agent:Agent, base_step:int=0, 
        min_update_step:int=20, max_update_step:int=300,random_inject_ratio=0.2,
        summary_writer:SummaryWriter=None
    ):
        """ generate perturbed src and paired with trg for discrimination
        note that discriminator takes in src with BOS ids
        :param agent: provides perturbation policy
        :param base_steps: used for saving
        :param min_update_step: minimum updates between each validation
        :param max_update_step: maximum updates 
        :param summary_writer: for loging
        :return: steps for the summary logging and acc
        """
        assert max_update_step>min_update_step, "at least update min_update step!"
        print("update D:")
        agent = agent.to(self.device)
        acc = 0.
        update_count = 0
        while update_count<max_update_step :
            while True: # extract a valid batch
                batch = next(self.local_iter)
                batch = MyBatch(
                    batch, pad_index=self.PAD_id, device=self.device)
                if batch.nseqs>1 and check_unk_ratio(self.victim_model.src_vocab,batch.src)<0.1:
                    break
            # generate training data with half the batch perturbed.
            self.discriminator.train()
            perturbed_src, flags = agent.seq_attack(
                src=batch.src, src_mask=batch.src_mask, src_length=batch.src_length, 
                word2near_vocab=self.word2near_vocab, 
                src_vocab=self.victim_model.src_vocab,
                half_mode=True, random_inject_ratio=random_inject_ratio
            )            
            # the perturbed src is retokenized
            novel_src, _, novel_length = self.retokenize(batch.src, perturbed_src, dropout=self.BPE_dropout)
            retok_src = torch.cat([self.BOS_id * novel_src.new_ones(novel_src.shape[0], 1), novel_src], dim=-1)            # perturbed_src follows the BOS padding style with the src 
            retok_mask = (retok_src != self.PAD_id).unsqueeze(1)
            retok_length = novel_length + 1 

            binary_probs = self.discriminator(
                retok_src, retok_mask, retok_length,
                batch.trg, batch.trg_mask, batch.trg_length
            )
            loss = self.loss_D(binary_probs, flags)
            summary_writer.add_scalar("env/loss_D",loss, global_step=base_step+update_count)
            loss.backward()
            # update gradient: for pytorch>1.1, 
            # backward -> clip -> optim step -> scheduler step -> zero_grad
            if self.clip_grad_fn is not None:
                self.clip_grad_fn(params=agent.parameters())
            self.optim_D.step()
            summary_writer.add_scalar("lr/lr-D",self.optim_D.param_groups[0]["lr"], global_step=base_step+update_count)
            if self.scheduler_D is not None and self.schedule_step_at == "step":
                self.scheduler_D.step()
            self.optim_D.zero_grad()

            update_count += 1
            if update_count%min_update_step==0:
                # validate D's accuracy, random inject ratio must be the same!
                acc = self.acc_validation(agent, random_inject_ratio)
                print("acc_D:%.2f"%acc)
                summary_writer.add_scalar("env/acc_D",
                    scalar_value=acc, global_step=base_step+update_count)
                if acc>self.acc_bound:
                    print("Reach training acc bound: %.2f/%.2f"%(acc, self.acc_bound))
                    break
        
        # scheduler updates once at D's training epoch (learning rate annealing)
        if self.schedule_step_at == "epoch":
            self.scheduler_D.step()

        return base_step+update_count, acc

    def terminated(self)->bool:
        """
        if current environment is terminated.
        """
        if 0 in self.terminal_signals:
            return False
        return True

    def get_state(self):
        """
        return copys of padded src and survival singals for learning
        (in place ops triggers gradient error)
        """
        return self.padded_src.detach().clone(), self.terminal_signals.detach().clone()

    def step(self, actions:Tensor)->Tensor:
        """
        update the environments by given actions (perturbations)[batch_size, dim]
        perturb the current rephrase_positions
        update states, increase index for next position to be perturbed
        yield rewards by local env's D(D is trained by src and trg padding setting by victim model)
        
        :param actions: whether perturb current index position [batch, 1]
        :param BLEU_reward: whether return BLEU score at current step (False -> None as BLEU score)
        :return: tensor step rewards (D's classification probs)
        """
        with torch.no_grad():
            batch_size = actions.shape[0]
            inputs = self.padded_src[:, self.index]  # position to be perturbed 
            inputs_mask = ~inputs.eq(self.PAD_id)  # position 
            # exact modificaiton on sequences (state)
            target_of_step = []
            for batch_index in range(batch_size):
                word_id = inputs[batch_index]
                target_word_id = self.word2near_vocab[word_id.item()][np.random.choice(len(self.word2near_vocab[word_id.item()]), 1)[0]]
                target_of_step += [target_word_id]
            if self.device != "cpu" and not actions.is_cuda:
                actions = actions.to(self.device)
                actions *= inputs_mask  # PAD position is neglected

            self.padded_src[:, self.index] *= (1-actions)  # erase 
            adjustification_ = self.padded_src.new_tensor(target_of_step)
            self.padded_src[:, self.index] += adjustification_ * actions
            
            # get local discriminator probs as instance rewards and update terminal signals
            D_probs = self.discriminator(
                self.padded_src[:, 1:], self.src_mask, self.src_length,
                self.trg, self.trg_mask, self.trg_length
            )

            # penaltys: currently surviving but temrinated by D and not the last step, -1 rewards
            penalty_flag = torch.logical_and(
                ~self.terminal_signals.bool(),
                D_probs.detach().argmax(dim=-1).bool(),
            ).int()

            death_penalty = -1.0 * penalty_flag

            # D probs as survival rewards [batch]
            dis, D_index = D_probs.max(dim=-1)  #[batch, ]
            dis = (dis - dis.mean())/(dis.std()+1e-6)  # norm the survival rewards
            D_index = (1-D_index) 
            # mask and rescale the survival rewards
            survival_rewards = dis * D_index * (1-self.terminal_signals)

            # update terminal signals by D
            self.terminal_signals = torch.logical_or(
                self.terminal_signals.bool(), 
                D_probs.detach().argmax(dim=-1).bool()
            ).int()
            self.index += 1
            # there are final BLEU variation to calculate. The variants are normed within [0-1]
            bleu_rewards = torch.zeros_like(survival_rewards)
            if self.index>=min(self.src_length).item():
                current_bleu=self.get_current_bleu()
                for i in range(len(current_bleu)):
                    if self.index==self.src_length[i].item()-1 and not self.terminal_signals[i].item():
                        # survives all perturbation, reward with relative bleu degradation
                        if self.origin_bleu[i] == 0:
                            relative_degradation = 0
                        else:
                            relative_degradation = (self.origin_bleu[i]-current_bleu[i])/self.origin_bleu[i]
                        bleu_rewards[i]+=relative_degradation * self.r_d_weight

            # terminal signals by length checking
            len_terminals = torch.le(self.src_length-1 - self.index, 0)
            self.terminal_signals = torch.logical_or(
                self.terminal_signals.bool(),
                len_terminals.bool()
            ).int()
            reward = survival_rewards * self.r_s_weight + bleu_rewards*self.r_d_weight + death_penalty
            normed_reward = (reward - reward.mean())/reward.std()
        return normed_reward
