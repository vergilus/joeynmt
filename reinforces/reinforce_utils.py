# coding: utf-8
from typing import OrderedDict, Tuple, Callable
from inspect import isfunction

import torch
import os
import torch.nn as nn
import numpy as np
import logging
import torch.nn.functional as F
from torch import Tensor
from joeynmt.constants import PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN
from joeynmt.helpers import load_checkpoint, ConfigurationError
from joeynmt.model import Model
from joeynmt.encoders import Encoder, RecurrentEncoder, TransformerEncoder
from joeynmt.decoders import Decoder, RecurrentDecoder, TransformerDecoder
from joeynmt.embeddings import Embeddings
from joeynmt.vocabulary import Vocabulary
import math
import torch.distributions as pyd
from torch.distributions.transformed_distribution import TransformedDistribution

class TanhTransform(pyd.transforms.Transform):
    """
    by https://github.com/denisyarats/pytorch_sac/blob/master/agent/actor.py
    """
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))

class SquashedNormal(TransformedDistribution):
    # gaussian samples are further squashed by tanh within (-1,1)
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale
        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu

def default(val, d):
    if val is not None:
        return val
    return d() if isfunction(d) else d

def sort_batch(seq_len):
    """Sorts torch tensor of integer indices by decreasing order."""
    with torch.no_grad():
        slens, sidxs = torch.sort(seq_len, descending=True)
    oidxs = torch.sort(sidxs)[1]

    return oidxs, sidxs, slens

def default_init(tensor):
    if tensor.ndimension()==1:
        nn.init.constant_(tensor, val=1.0)
    else:
        nn.init.xavier_normal_(tensor)
    return tensor

def rnn_init(tensor):
    if tensor.ndimension()!=2:
        return default_init(tensor)
    
    r,c = tensor.size()
    if r%c == 0:
        dim = 0
        n = r // c
        sub_size = (c, c)
    elif c%r == 0:
        dim = 1
        n = c // r
        sub_size = (r, r)
    else:
        return default_init(tensor)
    
    sub_tensors = [torch.Tensor(*sub_size).normal_(0,1) for _ in range(n)]
    sub_tensors = [torch.svd(w, some=True)[0] for w in sub_tensors]
    init_tensor = torch.cat(sub_tensors, dim=dim) # [r, c]

    with torch.no_grad():
        tensor.copy_(init_tensor)
    return tensor

class Saver(object):
    """ Saver to save and restore objects.

    Saver only accept objects which contain two method: ```state_dict``` and ```load_state_dict```
    """

    def __init__(self, save_prefix, num_max_keeping=1):
        self.save_prefix = save_prefix.rstrip(".")
        save_dir = os.path.dirname(self.save_prefix)

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        self.save_dir = save_dir

        if os.path.exists(self.save_prefix):
            with open(self.save_prefix) as f:
                save_list = f.readlines()
            save_list = [line.strip() for line in save_list]
        else:
            save_list = []

        self.save_list = save_list
        self.num_max_keeping = num_max_keeping

    @staticmethod
    def savable(obj):
        if hasattr(obj, "state_dict") and hasattr(obj, "load_state_dict"):
            return True
        else:
            return False

    def save(self, global_step, **kwargs):
        state_dict = dict()
        for key, obj in kwargs.items():
            if self.savable(obj):
                state_dict[key] = obj.state_dict()

        saveto_path = '{0}.{1}'.format(self.save_prefix, global_step)
        torch.save(state_dict, saveto_path)

        self.save_list.append(os.path.basename(saveto_path))

        if len(self.save_list) > self.num_max_keeping:
            out_of_date_state_dict = self.save_list.pop(0)
            if os.path.exists(os.path.join(self.save_dir, out_of_date_state_dict)):
                os.remove(os.path.join(self.save_dir, out_of_date_state_dict))

        with open(self.save_prefix, "w") as f:
            f.write("\n".join(self.save_list))

    def raw_save(self, **kwargs):
        """
        simply save the parameter in kwargs
        """
        state_dict = dict()

        for key, obj in kwargs.items():
            if self.savable(obj):
                state_dict[key] = obj.state_dict()
        save_to_path = '{0}.{1}'.format(self.save_prefix, "local")
        torch.save(state_dict, save_to_path)

        with open(self.save_prefix, "w") as f:
            f.write(save_to_path)


    def load_latest(self, **kwargs):
        if len(self.save_list) == 0:
            return
        latest_path = os.path.join(self.save_dir, self.save_list[-1])
        state_dict = torch.load(latest_path)

        for name, obj in kwargs.items():
            if self.savable(obj):
                if name not in state_dict:
                    print("Warning: {0} has no content saved!".format(name))
                else:
                    print("Loading {0}".format(name))
                    obj.load_state_dict(state_dict[name])

class Model_without_emb(nn.Module):
    """ this is a model mainly for RL environment inference, 
    the forward takes the src embedding as inputs, does not include 
    vocabs and src_embedding layers. 
    still takes all the inputs 
    only override the encoding accessories.
    """
    def __init__(
        self, 
        encoder:Encoder, decoder:Decoder,
        src_embed:Embeddings, src_vocab: Vocabulary,
        trg_embed:Embeddings, trg_vocab: Vocabulary) -> None:
        super().__init__()
        self.encoder=encoder
        self.decoder=decoder
        self.trg_embed=trg_embed
        self.trg_vocab=trg_vocab
        self.bos_index = self.trg_vocab.stoi[BOS_TOKEN]
        self.pad_index = self.trg_vocab.stoi[PAD_TOKEN]
        self.eos_index = self.trg_vocab.stoi[EOS_TOKEN]
        self._loss_function = None  # set by loss_function method

    @property
    def loss_function(self):
        return self._x

    @loss_function.setter
    def loss_function(self, loss_function: Callable):
        self._loss_function = loss_function

    def forward(self, return_type: str = None, **kwargs):
        """ Interface for multi-gpu
        the loss is used as a possible reward for the environments
        :param return_type: one of {"loss", "encode", "decode"}
        """
        if return_type is None:
            raise ValueError("Please specify return_type: "
                             "{`loss`, `encode`, `decode`}.")
        return_tuple = (None, None, None, None)
        if return_type == "loss":
            assert self.loss_function is not None
            out, _, _, _ = self._encode_decode(**kwargs)
            # compute log probs
            log_probs = F.log_softmax(out, dim=-1)
            # compute batch loss
            batch_loss = self.loss_function(log_probs, kwargs["trg"])
            # return batch loss
            #     = sum over all elements in batch that are not pad
            return_tuple = (batch_loss, None, None, None)

        elif return_type == "encode":
            encoder_output, encoder_hidden = self._encode(**kwargs)
            # return encoder outputs
            return_tuple = (encoder_output, encoder_hidden, None, None)

        elif return_type == "decode":
            outputs, hidden, att_probs, att_vectors = self._decode(**kwargs)
            # return decoder outputs
            return_tuple = (outputs, hidden, att_probs, att_vectors)
        return return_tuple

    def _encode(self, src_emb:Tensor, src_length:Tensor, src_mask:Tensor, 
                **_kwargs):
        """
        encode the given src embeddings, instead of src token ids
        """
        return self.encoder(src_emb, src_length, src_mask, **_kwargs)
    
    def _encode_decode(
        self, src_emb:Tensor, trg_input: Tensor, src_mask:Tensor,
        src_length: Tensor, trg_mask: Tensor = None, 
        **kwargs):
        encoder_output, encoder_hidden = self._encode(src_emb=src_emb,
                                                      src_length=src_length,
                                                      src_mask=src_mask,
                                                      **kwargs)
        unroll_steps = trg_input.size(1)
        assert "decoder_hidden" not in kwargs
        return self._decode(encoder_output=encoder_output,
                            encoder_hidden=encoder_hidden,
                            src_mask=src_mask, trg_input=trg_input,
                            unroll_steps=unroll_steps,
                            trg_mask=trg_mask, **kwargs)

    def _decode(self, encoder_output: Tensor, encoder_hidden: Tensor,
                src_mask: Tensor, trg_input: Tensor,
                unroll_steps: int, decoder_hidden: Tensor = None,
                att_vector: Tensor = None, trg_mask: Tensor = None, **_kwargs):
        """
        Decode, given an encoded source sentence.

        :param encoder_output: encoder states for attention computation
        :param encoder_hidden: last encoder state for decoder initialization
        :param src_mask: source mask, 1 at valid tokens
        :param trg_input: target inputs
        :param unroll_steps: number of steps to unrol the decoder for
        :param decoder_hidden: decoder hidden state (optional)
        :param att_vector: previous attention vector (optional)
        :param trg_mask: mask for target steps
        :return: decoder outputs (outputs:Tensor, hidden:Tensor, att_probs:Tensor, att_vectors:Tensor)
        """
        return self.decoder(trg_embed=self.trg_embed(trg_input),
                            encoder_output=encoder_output,
                            encoder_hidden=encoder_hidden,
                            src_mask=src_mask,
                            unroll_steps=unroll_steps,
                            hidden=decoder_hidden,
                            prev_att_vector=att_vector,
                            trg_mask=trg_mask,
                            **_kwargs)

    def __repr__(self) -> str:
        """
        String representation: a description of encoder, decoder and embeddings
        :return: string representation
        """
        return f"{self.__class__.__name__}(\n\tencoder={self.encoder}," \
                f"\n\tdecoder={self.decoder},\n\tsrc_embed={self.src_embed}," \
                f"\n\ttrg_embed={self.trg_embed})"

def build_embedding_and_NMT(
    victim_configs:dict,  victim_model_path:str, 
    src_vocab: Vocabulary=None, trg_vocab:Vocabulary=None,
    device:str="cpu"):
    """
    build and reload victim NMT with separate (src) embedding layers.
    """
    logger = logging.getLogger(__name__)
    logger.info("Building embedding...")
    src_padding_idx = src_vocab.stoi[PAD_TOKEN]
    trg_padding_idx = trg_vocab.stoi[PAD_TOKEN]
    src_embed = Embeddings(
        **victim_configs["encoder"]["embeddings"], vocab_size=len(src_vocab),
        padding_idx=src_padding_idx)

    # this ties source and target embeddings
    # for softmax layer tying, see further below
    if victim_configs.get("tied_embeddings", False):
        if src_vocab.itos == trg_vocab.itos:
            # share embeddings for src and trg
            trg_embed = src_embed
        else:
            raise ConfigurationError(
                "Embedding cannot be tied since vocabularies differ.")
    else:
        trg_embed = Embeddings(
            **victim_configs["decoder"]["embeddings"], vocab_size=len(trg_vocab),
            padding_idx=trg_padding_idx)
    
    # build encoder decoder NMT without embedding
    logger.info("Building encoder & decoder for NMT...")
    enc_dropout = victim_configs["encoder"].get("dropout", 0.)
    enc_emb_dropout = victim_configs["encoder"]["embeddings"].get("dropout", enc_dropout)
    if victim_configs["encoder"].get("type", "recurrent") == "transformer":
        assert victim_configs["encoder"]["embeddings"]["embedding_dim"] == \
               victim_configs["encoder"]["hidden_size"], \
               "for transformer, emb_size must be hidden_size"

        encoder = TransformerEncoder(**victim_configs["encoder"],
                                     emb_size=src_embed.embedding_dim,
                                     emb_dropout=enc_emb_dropout)
    else:
        encoder = RecurrentEncoder(**victim_configs["encoder"],
                                   emb_size=src_embed.embedding_dim,
                                   emb_dropout=enc_emb_dropout)
    dec_dropout = victim_configs["decoder"].get("dropout", 0.)
    dec_emb_dropout = victim_configs["decoder"]["embeddings"].get("dropout", dec_dropout)
    if victim_configs["decoder"].get("type", "recurrent") == "transformer":
        decoder = TransformerDecoder(
            **victim_configs["decoder"], encoder=encoder, vocab_size=len(trg_vocab),
            emb_size=trg_embed.embedding_dim, emb_dropout=dec_emb_dropout)
    else:
        decoder = RecurrentDecoder(
            **victim_configs["decoder"], encoder=encoder, vocab_size=len(trg_vocab),
            emb_size=trg_embed.embedding_dim, emb_dropout=dec_emb_dropout)
    
    # a model without src embedding
    model = Model_without_emb(
                encoder=encoder, decoder=decoder,
                src_embed=src_embed, src_vocab=src_vocab,
                trg_embed=trg_embed, trg_vocab=trg_vocab)
    model.to(device)
    src_embed.to(device)
    # tie softmax layer with trg embeddings
    if victim_configs.get("tied_softmax", False):
        if trg_embed.lut.weight.shape == \
                model.decoder.output_layer.weight.shape:
            # (also) share trg embeddings and softmax layer:
            model.decoder.output_layer.weight = trg_embed.lut.weight
        else:
            raise ConfigurationError(
                "For tied_softmax, the decoder embedding_dim and decoder "
                "hidden_size must be the same."
                "The decoder must be a Transformer.")
    logger.info("encoder decoder Transformer built.")
    # load model parameters from checkpoint path
    ckp = torch.load(victim_model_path, map_location=device)
    logger.info("reload model parameters")
    model.load_state_dict(ckp["model_state"], strict=False)
    logger.info("reload embeddings")
    src_embed.load_state_dict(ckp["model_state"], strict=False)
    return src_embed, model

def load_nearest_cand(
        save_to:str, 
        src_vocab:Vocabulary=None, emit_as_id:bool=False
    )->dict:
    """
    """
    if emit_as_id:
        assert src_vocab is not None, "src vocab must be provided for token candidates"
    with open(save_to, "r") as similar_vocab:
        word2near_vocab = OrderedDict()
        for line in similar_vocab:
            line = line.split("\t")
            if emit_as_id:
                word2near_vocab[src_vocab.stoi[line[0]]] = [src_vocab.stoi[tok] for tok in line[1:]]
            else:
                word2near_vocab[line[0]] = line[1:]
    return word2near_vocab

def load_or_extract_near_vocab(
    victim_model:Model, 
    save_to:str, batch_size:int=100, top_reserve:int=12,
    reload:bool=True, emit_as_id:bool=False, 
)->dict:
    """
    extract nearest token candidates for the perturbation. 
    will append UNK as additional candidates.
    saves the near_vocab to the save_to directory
    return the near_vocab dict
    """
    print("extract nearest candidates to %s"%save_to)
    src_vocab = victim_model.src_vocab
    emb = victim_model.src_embed.lut.weight.detach().clone()  # the exact nn.Embeddings
    if victim_model.src_embed.scale:
        emb *= math.sqrt(victim_model.src_embed.embedding_dim)
    len_mat = torch.sum(emb**2, dim=1)**0.5  # length of the embeddings

    if os.path.exists(save_to) and reload:
        print("load from %s:" % save_to)
        return load_nearest_cand(save_to, src_vocab, emit_as_id)
    else:
        print("collect near candidates for vocabulary")
        with open(save_to, "w") as similar_vocab, torch.no_grad():
            # batched collection of topk candidates for average Euclidean
            # distance,  
            avg_dist = 0.
            counter= 0
            word2near_vocab = OrderedDict()
            for i in range(len(src_vocab)//batch_size+1):
                if i*batch_size==len(src_vocab):
                    break
                index = torch.tensor(range(i*batch_size,
                              min(len(src_vocab), (i+1)*batch_size),
                              1))
                # extract embedding data
                slice_emb = emb[index]
                collect_len = torch.mm(len_mat.narrow(0, i * batch_size, min(len(src_vocab), (i+1)*batch_size)-i*batch_size).unsqueeze(1),
                                len_mat.unsqueeze(0))
                # filter top-k cosine-nearest vocab, then calculate avg Euclidean distance 
                similarity = torch.mm(slice_emb,
                                       emb.t()).div(collect_len)
                # get value and index
                topk_index = similarity.topk(top_reserve, dim=1)[1]
                sliceemb = slice_emb.unsqueeze(dim=1).repeat(1, top_reserve, 1)  # [batch_size, 1*k, dim]
                E_dist = ((emb[topk_index]-sliceemb)**2).sum(dim=-1)**0.5
                # print("avg Euclidean distance:", E_dist)
                avg_dist += E_dist.mean()
                counter += 1
            avg_dist = avg_dist.item() / counter
            print("avg_dist",avg_dist)
            
            # traverse the vocabulary and yield the candidates within average E-dist
            for i in range(len(src_vocab)//batch_size +1):
                if i*batch_size==len(src_vocab):
                    break
                index = torch.tensor(range(i*batch_size,
                              min(len(src_vocab), (i+1)*batch_size),
                              1))
                # extract embedding data
                slice_emb = emb[index]
                collect_len = torch.mm(len_mat.narrow(0, i * batch_size, min(len(src_vocab), (i+1)*batch_size)-i*batch_size).unsqueeze(1),
                                len_mat.unsqueeze(0))
                # filter top-k nearest vocab by cosine similarity
                similarity = torch.mm(slice_emb,
                                       emb.t()).div(collect_len)
                _, topk_indices = similarity.topk(top_reserve, dim=1)
                sliceemb = slice_emb.unsqueeze(dim=1).repeat(1, top_reserve, 1)  # [batch_size, 1*k, dim]
                E_dist = ((emb[topk_indices]-sliceemb)**2).sum(dim=-1)**0.5

                topk_val = E_dist.cpu().detach().numpy()
                topk_indices = topk_indices.cpu().detach().numpy()
                for j in range(topk_val.shape[0]):
                    bingo = 0  # count for valid candidate and append UNK if nothing valid
                    src_word_id = j + i * batch_size
                    src_word = src_vocab.itos[src_word_id]
                    near_vocab = []
                    similar_vocab.write(src_word+"\t")
                    # no additional candidates for reserved token
                    if src_word in [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]:
                        near_cand_id = src_word_id
                        near_cand = src_word
                        similar_vocab.write(near_cand+"\t")
                        bingo = 1
                        if emit_as_id:
                            near_vocab+=[near_cand_id]
                        else:
                            near_vocab+=[near_cand]
                    else:
                        for k in range(1, topk_val.shape[1]):
                            near_cand_id = topk_indices[j][k]
                            near_cand = src_vocab.itos[near_cand_id]
                            if (near_cand not in [PAD_TOKEN, EOS_TOKEN, BOS_TOKEN]):
                                # and topk_val[j][k]<avg_dist:
                                bingo += 1
                                similar_vocab.write(near_cand+"\t")
                                if emit_as_id:
                                    near_vocab+=[near_cand_id]
                                else:
                                    near_vocab+=[near_cand]
                        if bingo==0 :
                            last_cand_ids = [src_vocab.stoi[UNK_TOKEN]]
                            for final_reserve_id in last_cand_ids:
                                last_cand = UNK_TOKEN
                                similar_vocab.write(last_cand+"\t")
                                if emit_as_id:
                                    near_vocab+=[final_reserve_id]
                                else:
                                    near_vocab+=[last_cand]
                    similar_vocab.write("\n")
                    if emit_as_id:
                        word2near_vocab[src_word_id] = near_vocab
                    else:
                        word2near_vocab[src_word] = near_vocab

    return  word2near_vocab

"""
Implementation of a mini-batch.
"""
class MyBatch:
    """Object for holding a batch of data with mask during training.
    Input is a batch from a torch text iterator.
    """
    # pylint: disable=too-many-instance-attributes
    def __init__(self, torch_batch, pad_index, device):
        """
        Create a new joey batch from a torch batch.
        This batch extends torch text's batch attributes with src and trg
        length, masks, number of non-padded tokens in trg.
        Furthermore, it can be sorted by src length.

        :param torch_batch:
        :param pad_index:
        :param use_cuda:
        """
        self.src, self.src_length = torch_batch.src
        self.src_mask = (self.src != pad_index).unsqueeze(1)
        self.nseqs = self.src.size(0)
        self.trg_input = None
        self.trg = None
        self.trg_mask = None
        self.trg_length = None
        self.ntokens = None
        self.device=device
        self.use_cuda = True if self.device.startswith("cuda") else False

        if hasattr(torch_batch, "trg"):
            trg, trg_length = torch_batch.trg
            self.trg = trg
            self.trg_mask = (self.trg != pad_index).unsqueeze(1)
            self.trg_length = trg_length
            # we exclude the padded areas from the loss computation
            self.ntokens = (self.trg != pad_index).data.sum().item()

        if self.use_cuda:
            self._make_cuda()

    def _make_cuda(self):
        """
        Move the batch to GPU

        :return:
        """
        self.src = self.src.to(self.device)
        self.src_mask = self.src_mask.to(self.device)
        self.src_length = self.src_length.to(self.device)

        if self.trg is not None:
            self.trg = self.trg.to(self.device)
            self.trg_mask = self.trg_mask.to(self.device)
            self.trg_length = self.trg_length.to(self.device)

    def sort_by_src_length(self):
        """
        Sort by src length (descending) and return index to revert sort

        :return:
        """
        _, perm_index = self.src_length.sort(0, descending=True)
        rev_index = [0]*perm_index.size(0)
        for new_pos, old_pos in enumerate(perm_index.cpu().numpy()):
            rev_index[old_pos] = new_pos

        sorted_src_length = self.src_length[perm_index]
        sorted_src = self.src[perm_index]
        sorted_src_mask = self.src_mask[perm_index]
        if self.trg is not None:
            sorted_trg_length = self.trg_length[perm_index]
            sorted_trg_mask = self.trg_mask[perm_index]
            sorted_trg = self.trg[perm_index]

        self.src = sorted_src
        self.src_length = sorted_src_length
        self.src_mask = sorted_src_mask

        if self.trg is not None:
            self.trg_mask = sorted_trg_mask
            self.trg_length = sorted_trg_length
            self.trg = sorted_trg

        if self.use_cuda:
            self._make_cuda()

        return rev_index

def extract_action_space(src_embed:Embeddings, top_reserve=10, device="cpu")-> float:
    """
    first collect KNN candidates by cosine similarity, where we extract largest L2-radius of a token 
    amongst its candidates. 
    collect the minimum L2-radius for the whole vocabulary.
    return (abs_action_space-mean)/var
    
    :param top_reserve: defines the norm-ball radius by KNN
    :param src_emb: Embedding object 
    return: float tuple [min_act, max_act]
    """
    batch_size = 50
    with torch.no_grad():
        # calculate radius of the norm-ball
        emb = src_embed.lut.weight.detach().clone()
        vocab_size, emb_dim=emb.shape
        if src_embed.scale:
            emb*= math.sqrt(emb_dim)
        len_mat = torch.sum(emb**2, dim=1)**0.5  # length of the embeddings
        min_dist = 99999.
        avg_dist = 0.
        counter= 0
        for i in range(vocab_size//batch_size+1):
            if i*batch_size==vocab_size:
                break
            index = torch.tensor(range(i*batch_size,
                            min(vocab_size, (i+1)*batch_size),
                            1))
            # extract embedding data
            slice_emb = emb[index]
            collect_len = torch.mm(len_mat.narrow(0, i * batch_size, min(vocab_size, (i+1)*batch_size)-i*batch_size).unsqueeze(1),
                            len_mat.unsqueeze(0))
            # filter top-k cosine-nearest vocab, then calculate avg Euclidean distance 
            similarity = torch.mm(slice_emb,
                                    emb.t()).div(collect_len)
            # get value and index and exclude the self-attention
            topk_index = similarity.topk(top_reserve, dim=1)[1]
            
            sliceemb = slice_emb.unsqueeze(dim=1).repeat(1, top_reserve, 1)  # [batch_size, 1*k, dim]
            E_dist = ((emb[topk_index]-sliceemb)**2).sum(dim=-1)**0.5  # Euclidean dist of KNN
            mean_batched_E_radius = E_dist.max(dim=-1)[0].mean().item()  # extract KNN radius for the token, mean over batch
            batched_min = E_dist[E_dist.nonzero(as_tuple=True)].min().item()
            
            if batched_min<min_dist:
                min_dist = batched_min
            # print("avg Euclidean distance:", E_dist)
            avg_dist += mean_batched_E_radius
            counter += 1
        avg_dist = avg_dist / counter
        # distribute on embeddings per dim.
        knn_ball_radius = avg_dist/math.sqrt(emb_dim)
        min_dist = min_dist/math.sqrt(emb_dim)
        abs_action_space = knn_ball_radius
        
        # norm the action space
        total_mean = emb.mean().item()
        total_std = emb.std().item()
        print(min_dist, knn_ball_radius)
        max_act = (abs_action_space-total_mean)/total_std
        print("return normed act space with mean:%f, std:%f"%(total_mean, total_std))

        assert min_dist<max_act, "act range error, check reinforce_utils.py"
        return [min_dist, max_act]


class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]

def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

