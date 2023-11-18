import torch
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np
import re
from torch import Tensor
from joeynmt.model import Model
from joeynmt.encoders import RecurrentEncoder
from joeynmt.constants import UNK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN
from joeynmt.embeddings import Embeddings
from joeynmt.vocabulary import Vocabulary
from adversarials.adversarial_utils import *

class Agent(nn.Module):
    """
    the base agent policy is the vanilla MLP by given the exact src_vocab

    states: the current representation of the sentence
    actions: a perturbation bool or vector on the given position
    rewards: BLEU scores or discriminator values
    """
    def __init__(self, 
            victim_model:Model,
            action_dim:int=2, action_roll_steps:int=3, 
            d_model:int=256, dropout:float=0.0, 
            **kwargs):
        super(Agent, self).__init__()
        self.input_dim = victim_model.src_embed.embedding_dim
        self.action_roll_steps = action_roll_steps  # for on policy value estimation training
        self.action_dim = action_dim
        self.d_model = d_model
        self.BOS_id = victim_model.src_vocab.stoi[BOS_TOKEN]
        self.UNK_id = victim_model.src_vocab.stoi[UNK_TOKEN]
        # agent contains an embedding representation 
        self.src_embedding = Embeddings(
            embedding_dim=self.input_dim, 
            scale=victim_model.src_embed.scale, 
            vocab_size=len(victim_model.src_vocab),
            padding_idx=victim_model.src_vocab.stoi[PAD_TOKEN],
            freeze=kwargs["freeze_embedding"])

        # representaton encoding layer: bi-GRU
        self.src_gru = RecurrentEncoder(
            rnn_type="gru",
            hidden_size=d_model, emb_size=self.input_dim,dropout=dropout,
            bidirectional=True)

        self.ctx_linear = nn.Linear(2*d_model, d_model)
        self.input_linear = nn.Linear(self.input_dim, d_model)
        self.feature_layer_norm = nn.LayerNorm(self.d_model, elementwise_affine=True)

        # classifier and value net
        self.attacker_linear = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.LeakyReLU(),
            nn.Linear(self.d_model, self.action_dim),
        )
        self.critic_linear = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.LeakyReLU(),
            nn.Linear(self.d_model, 1)
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for weight in self.ctx_linear.parameters():
            default_init(weight)
        for weight in self.input_linear.parameters():
            default_init(weight)
        for weight in self.attacker_linear.parameters():
            default_init(weight)
        for weight in self.critic_linear.parameters():
            default_init(weight)
        for weight in self.src_gru.parameters():
            rnn_init(weight)
        # the embedding is initialized by the victim model

    def reset(self):
        self._reset_parameters()

    def sync_from(self, agent):
        self.load_state_dict(agent.state_dict())

    def preprocess(self, 
        src:Tensor, src_mask:Tensor, src_length:Tensor, 
        label:Tensor) -> Tensor:
        """
        :param src: batched padded src tokens [batch, len]
        :param src_mask: indicate valid token [batch, 1, len]
        :param src_length: [batch]
        encode the current token sequence and the position to be attacked
        """
        label_emb = self.src_embedding(label) # batch, 3, dim
        label_emb = label_emb.sum(dim=1) # batch, dim
        
        src_emb = self.src_embedding(src)
        ctx_x, _ = self.src_gru(src_emb, src_length, src_mask)
        # print("ctx",ctx_x.shape,"mask", src_mask.shape,"length", src_length.shape)
        ctx_mean = (ctx_x * src_mask.squeeze(dim=1).unsqueeze(2).float()).sum(1)/ src_length.unsqueeze(1)
        
        perturb_feature = self.input_linear(label_emb)+self.ctx_linear(ctx_mean)
        perturb_feature = self.feature_layer_norm(perturb_feature)
        return perturb_feature

    def get_perturb(self, 
        src:Tensor, src_mask:Tensor, src_length:Tensor, 
        label:Tensor) -> Tensor:
        """ 
        :param src: batched padded src tokens [batch, len]
        :param src_mask: indicate valid token [batch, 1, len]
        :param src_length: [batch]
        get binary distribution of whether to perturb
        """
        perturb_feature = self.preprocess(src, src_mask, src_length, label)
        out = self.attacker_linear(perturb_feature)
        perturb_dist = F.softmax(out-out.max(), dim=-1)
        return perturb_dist

    def get_critic(self, 
        src:Tensor, src_mask:Tensor, src_length:Tensor, 
        label:Tensor) -> Tensor:
        """
        :param src: batched padded src tokens [batch, len]
        :param src_mask: indicate valid token [batch, 1, len]
        :param src_length: [batch]
        Value function (V value, or state-value function)
        """
        perturb_feature = self.preprocess(src, src_mask, src_length, label)
        value = self.critic_linear(perturb_feature)
        return value

    def forward(self, 
        src:Tensor, src_mask:Tensor, src_length:Tensor, 
        label:Tensor):
        """
        returns the perturbation bianry and value estimation
        """
        perturb_feature = self.preprocess(src, src_mask, src_length, label)
        out = self.attacker_linear(perturb_feature)
        perturb_dist = F.softmax(out-out.max(), dim=-1)
        value = self.critic_linear(perturb_feature)
        return perturb_dist, value

    def _random_UNK(self,
        src:Tensor, src_mask:Tensor, src_length:Tensor,
        src_vocab:Vocabulary,
        rate:float=0.0) -> Tensor:
        """
        randomly choose ids in src and replace by UNK id
        :param src_length: indicates the valid positions for perturbation
        :param src_vocab: don't perturb specific tokens for adversairal data augmentation 
        :return: the perturbed src in the same size with src
        """
        with torch.no_grad():
            perturbed_src = src.detach().clone()
            batch_size, _ = src.shape
            for i in range(batch_size):
                for ii in range(src_length[i].item()-1):
                    if np.random.uniform(0,1)<rate and not re.search(r'\d', src_vocab.itos[src[i][ii]]):
                        # UNK injection does not perturb the numbers
                        perturbed_src[i][ii] = self.UNK_id
        return perturbed_src

    def seq_attack(self, 
        src:Tensor, src_mask:Tensor, src_length:Tensor,
        word2near_vocab:dict, src_vocab:Vocabulary, 
        half_mode:bool, random_inject_ratio=0.0
        ):
        """
        set the agent to validation mode, perturb from left to right,
        provide samples for discriminator training
        :param src: batched padded src tokens [batch, len]
        :param src_mask: indicate valid token [batch, 1, len]
        :param src_length: [batch]
        :param src_vocab: for adversarial data-augmentation, don't perturb specific tokens for positive data
        :param half: if true perturb half of the batch 
        :return perturbed src (ids) and indicators of whether src is perturbed
                by the policy, note that the results always start with BOS
                Tensor, Tensor
        """
        self.eval()
        if src[0][0].item()==self.BOS_id:
            perturbed_src = src.detach().clone()
            extended=False
        else:  # extend with BOS and backup the origin src 
            perturbed_src = torch.cat([self.BOS_id * src.new_ones(src.shape[0], 1), src], dim=-1)
            src_mask = torch.cat([src_mask.new_ones(src.shape[0],1,1), src_mask], dim=-1)
            src_length += 1 
            extended=True
        with torch.no_grad():
            batch_size, max_steps = perturbed_src.shape
            for t in range(1, max_steps-2):  # ignore BOS adn EOS
                inputs = perturbed_src[:, t-1:t+2]  # input (with its surroundings) to be perturbed
                policy_out, _ = self.forward(
                    perturbed_src, src_mask=src_mask, src_length=src_length,
                    label=inputs)
                actions = policy_out.argmax(dim=-1)
                target_of_step = []
                for batch_index in range(batch_size):
                    word_id = inputs[batch_index][1]  # middle is the token
                    # choose a random candidate
                    target_word_id = np.random.choice(word2near_vocab[word_id.item()], 1)[0]
                    target_of_step += [target_word_id]
                # override with random choice from candidates
                perturbed_src[:, t] *=(1-actions)
                adjustification_ = inputs.new_tensor(target_of_step)
                perturbed_src[:, t] += adjustification_ * actions
            
            # by default all src is perturbed
            flags = inputs.new_ones(batch_size).long()
            if half_mode:
                # randomly choose half of the src to be perturbed.
                flags = torch.bernoulli(0.5*flags).long()
                perturbed_src *= flags.unsqueeze(dim=-1)
                random_inject_src = self._random_UNK(
                    src, src_mask, src_length, 
                    src_vocab=src_vocab, rate=random_inject_ratio)
                perturbed_src += torch.cat([self.BOS_id * src.new_ones(src.shape[0], 1), random_inject_src], dim=-1)*(1-flags.unsqueeze(dim=-1))
            # apply mask on results
            perturbed_src *= src_mask.squeeze()
            if extended:  # return the src by original setting without BOS.
                perturbed_src = perturbed_src[:, 1:]
                src_mask = src_mask[:, :, 1:]
                src_length -= 1

        return perturbed_src.detach(), flags.detach()

