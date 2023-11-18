import torch
import numpy as np
from torch import Tensor

from torchtext.legacy.data import Dataset, Iterator
from reinforces.reinforce_utils import *
import codecs
from joeynmt.model import Model
from joeynmt.loss import XentLoss
from joeynmt.constants import UNK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, DEFAULT_UNK_ID
from joeynmt.embeddings import Embeddings
from joeynmt.helpers import bpe_postprocess
from joeynmt.search import beam_search
from joeynmt.data import token_batch_size_fn
from joeynmt.batch import Batch
from sacrebleu.metrics import BLEU
from subword_nmt import apply_bpe

class Translate_Env(object):
    """
    wrap the translator as environment for reinforcement learing
    provides state transition dynamics and step rewards 
    translator and discriminator
    
    :return: states, and corresponding rewards on batched sequences.
    """
    def __init__(self,
            reinforce_configs:dict,
            victim_src_embed: Embeddings, 
            victim_model:Model, data_set:Dataset,
            save_to:str, device:str
        ) -> None:
        """
        :param victim_model: receives a locally-built victim model in thread
        :param save_to: in case of some accessive saving
        """
        self.victim_src_embed =victim_src_embed  # initial representation
        self.victim_model = victim_model  # the model body that provides rewards
        assert self.victim_model.training==False, "victim must be validation mode" 
        self.PAD_id = self.victim_model.trg_vocab.stoi[PAD_TOKEN]
        self.save_to = save_to
        self.device = device

        # normalized embedding space as states for agent's policy
        self.total_mean, self.total_std = self._get_norm_stat() 

        # rewards
        self.zero_bleu_bound = reinforce_configs["zero_bleu_bound"]
        self.zero_bleu_patience = reinforce_configs["zero_bleu_patience"]
        self.r_s_weight = reinforce_configs["r_s_weight"]
        self.r_d_weight = reinforce_configs["r_d_weight"]
        self.metric = BLEU()  # episodic rewards by BLEU
        self.batch_size = reinforce_configs["batch_size"]  # batch_size
        batch_fn = token_batch_size_fn if reinforce_configs["batching_key"]=="token" else None
        # retokenization and translation
        self.bpe = apply_bpe.BPE(codes=codecs.open(reinforce_configs["code_file_path"], encoding='utf-8')) 

        # build local iterator (bilingual, sort by src_length)
        # (with "repeat as true" for infinite iteration and shuffle for epoch)
        self.local_iter = iter(Iterator(
            repeat=True, sort=False, dataset=data_set,
            batch_size=self.batch_size, batch_size_fn=batch_fn, 
            train=True, sort_within_batch=True,
            sort_key=lambda x: len(x.src), shuffle=True
        ))
        self.victim_model.loss_function = XentLoss(
            pad_index=self.victim_model.pad_index, smoothing=0.1,
            reduction="none")  # smooth whatever u like

    def _get_norm_stat(self):
        # return mean and variance for the victim embedding space (over vocabulary)
        with torch.no_grad():
            weight = self.victim_src_embed.lut.weight.detach().clone()
            if self.victim_src_embed.scale:
                weight*=math.sqrt(weight.shape[1])
            return weight.mean(), weight.std()

    def reset_state(self, show_info=True):
        """
        initiate environment with bathced sentences and a BLEU scorer
        including original BLEU score.
        returns the batch, and index
        """
        while True:
            batch = next(self.local_iter)
            batch = Batch(batch, 
                pad_index=self.PAD_id, use_cuda=True)
            batch.to(self.device)
            if batch.nseqs>1:
                if show_info:
                    print("initiate environment..."),
                break
        """
        Tensor object, for terminal signals, a sample is terminated when its test
        bleu reaches and stays zero for self.zero_bleu_patience
        """ 
        self.src_length = batch.src_length  # [batch]
        self.bleu_patience = self.zero_bleu_patience * torch.ones_like(batch.src_length)
        self.terminal_signals = self.bleu_patience.le(0)  # [batch], there are still patience left
        
        self.origin_src = batch.src
        self.origin_src_emb = self.victim_src_embed(batch.src) # log the original representation
        self.perturbed_src_emb = self.origin_src_emb.detach().clone() # the perturbed representation
        self.src_mask = batch.src_mask  # [batch, 1, len]
        # this is used for loss variations as rewards(BLEU is not sensitive enough)
        self.trg = batch.trg  # loss computation
        self.trg_input = batch.trg_input  # teacher-forcing inputs for loss computation
        self.trg_length = batch.trg_length
        self.trg_mask = batch.trg_mask

        # for bleu rewards
        self.origin_bleu=[]
        decode_result = self.translate()  # translate current batch src_embeddings origin bleu
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
    
    def get_state(self):
        """return normalized current embedding as states for policy
        """
        s_t = (self.perturbed_src_emb.detach().clone()-self.total_mean) / self.total_std
        return s_t, self.terminal_signals.detach().clone()
    
    def translate(self, src_emb:Tensor=None, 
        src_mask:Tensor=None, src_length:Tensor=None):
        """
        fast translate the given src_emb, translate current batch in the env if none is given
        returns the padded decoder ids
        """
        if src_emb is None:
            src_emb = self.perturbed_src_emb  # the perturbed src
            src_mask = self.src_mask
            src_length = self.src_length
        
        with torch.no_grad():
            # print("translate on: ",src.device)
            max_output_length = int(max(src_length.cpu().numpy())*1.5)
            # encode
            encoder_output, encoder_hidden, _, _ = self.victim_model(
                return_type="encode", 
                src_emb=src_emb, src_length=src_length,src_mask=src_mask
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
        tranlate current states and return current bleu score
        the BLEU is calculated by the orignal reference.
        return: a list of BLEU in range [0-100]
        """
        decode_result = self.translate()  # translate current batch init origin bleu
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
            bleu_t = self.metric.corpus_score([hyp], [[refs[i]]]).score
            current_bleu.append(bleu_t)
        return current_bleu
    
    def get_current_loss(self):
        """
        calculate the training loss for current representation as rewards.
        return the sample-wise loss(sum over embedding dim) [batch]
        """
        loss, _, _, _ = self.victim_model(
            return_type="loss", 
            src_emb=self.perturbed_src_emb, trg=self.trg, trg_input=self.trg_input, src_mask=self.src_mask,
            src_length=self.src_length, trg_mask=self.trg_mask)
        loss = loss.view(self.trg.shape[0], -1, loss.shape[-1]).sum(dim=-1)
        return loss

    def terminated(self)->bool:
        """
        if current environment is terminated.
        """
        if 0 in self.terminal_signals:
            return False
        return True

    def view_token(self, src_vocab:Vocabulary):
        """
        only the last sentence embedding of self.perturbed_src_emb as default
        projected nearest token for perturbed token embedding. 
        traverse the self.victim_src_embed and extract nearest candidate (EucliDist)
        provide the src_vocab for token views.

        note that the embedding might be scaled according to victim settings.
        """
        with torch.no_grad():
            emb = self.victim_src_embed.lut.weight  # extract embedding
            if self.victim_src_embed.scale:
                emb *= math.sqrt(self.victim_src_embed.embedding_dim)
            # batch_size = self.perturbed_src_emb.shape[0] 
            perturbed_emb = self.perturbed_src_emb[0]  # [sent_len, dim]
            E_dist =((perturbed_emb.unsqueeze(dim=1) - emb.unsqueeze(dim=0))**2).sum(dim=-1) # [sent_len, vocab]
            _, nearest_id = E_dist.topk(dim=-1, k=2, largest=False)  # [sent_len] of tensor index
            sentence = []
            for id_list in nearest_id:
                id = id_list[0]
                s=src_vocab.itos[id.item()]
                if s in [EOS_TOKEN]:
                    break
                if s in [PAD_TOKEN]:
                    continue
                sentence.append(s)
        return sentence
    
    def step_to(self, new_state:Tensor, with_bleu=False, show_info=False):
        batch_size, src_len, _ = new_state.shape
        with torch.no_grad():
            current_bleu_list = self.get_current_bleu()
            self.perturbed_src_emb = new_state*self.total_std + self.total_mean
            perturbed_bleu_list = self.get_current_bleu()
            bleu_variants = []
            for i in range(batch_size):
                bleu_variants.append(perturbed_bleu_list[i]-current_bleu_list[i])
            bleu_variants = - np.array(bleu_variants)
        return

    def update_to(self, new_state:Tensor, with_bleu=False, show_info=False):
        """
        update environments's current embedding by given new states, 
        yield rewards on each token: [batch]
        the batched step rewards must be normed
        
        :param new_state: the perturbed s_t (normalized embeddings)
        :param with_bleu: bool, whether return bleu variations as part of rewards
        :return: sample-wise rewards Tensor (loss variation and BLEU score variation)
         [batch]
         reward is ovrally normed
        """
        batch_size, src_len, _ = new_state.shape
        with torch.no_grad():
            # masked action:
            current_bleu_list = self.get_current_bleu()
            current_loss = self.get_current_loss()
            perturbed_src_emb = new_state*self.total_std + self.total_mean
            if show_info:
                print("perturbation:%.5g"%abs(self.perturbed_src_emb- perturbed_src_emb).mean().item())

            # collect loss variants
            self.perturbed_src_emb = perturbed_src_emb
            perturbed_bleu_list = self.get_current_bleu()
            perturbed_loss = self.get_current_loss()
            # mean over the length
            loss_variants = abs(perturbed_loss-current_loss).mean(dim=-1)
            loss_variants = loss_variants.clamp(max=5)
            if show_info:
                # print("loss:", loss_variants[0])
                print("bleu count down:%d, from %.4g~%.4g"%(self.bleu_patience[0].item(), current_bleu_list[0], perturbed_bleu_list[0])),
            # normed step rewards, norm over all samples the loss varies on different tokens
            # normed_loss_variants = (loss_variants-loss_variants.mean())/(loss_variants.std()+1e-6)
            reward = self.r_s_weight * loss_variants
            
            # collect bleu variants without normed bleu variants
            if with_bleu:
                bleu_variants = []
                for i in range(batch_size):
                    bleu_variants.append(perturbed_bleu_list[i]-current_bleu_list[i])
                bleu_variants = - np.array(bleu_variants)  # noiser advocates degradation
                # normed_bleu_variants = (bleu_variants-bleu_variants.mean())/(bleu_variants.std()+1e-6)
                reward += self.r_d_weight * bleu_variants
            # norm the overall rewards to make tuning easier
            reward = (reward-reward.mean())/(reward.std() + 1e-6)

            # exploration bounded within certain range of BLEU variation
            # update zero-BLEU patience for surviving sentence
            lb = self.zero_bleu_bound  # below lower-bound are considered bleu-zero and are additionally punished
            for i in range(batch_size):  # bleu changing from 0 to 0, reduce the patience
                if self.bleu_patience[i]>=0:
                    if current_bleu_list[i]<lb and perturbed_bleu_list[i]<lb:  # stays zero
                        self.bleu_patience[i] -= 1
                        reward[i] -= self.r_d_weight*1  # punished for exceed exploration
                    elif current_bleu_list[i]<lb and perturbed_bleu_list[i]>=lb: 
                        self.bleu_patience[i] = self.zero_bleu_patience  # restore patience
            # update terminal signals for exploration
            self.terminal_signals = self.bleu_patience.le(0)  # [batch], there are still patience left

        return reward
