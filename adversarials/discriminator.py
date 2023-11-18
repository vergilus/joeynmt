import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from adversarials.adversarial_utils import *
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from joeynmt.embeddings import Embeddings
from joeynmt.encoders import RecurrentEncoder
from joeynmt.vocabulary import Vocabulary
from joeynmt.model import Model
from joeynmt.constants import UNK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN

def sort_batch(seq_len:Tensor):
    """ sort the torch tensor of integer indices by decreasing order for torch RNN
    """
    with torch.no_grad():
        sorted_lens, sorted_idxs = torch.sort(seq_len, descending=True)
    origin_idxs = torch.sort(sorted_idxs)[1]
    return origin_idxs, sorted_idxs, sorted_lens

class TransDiscriminator(nn.Module):
    """
    discriminate whether the trg (y) is a translation of a src (x)
    requires the victim model's src and trg vocab, 
    and their corresponding embedding
    """
    def __init__(
        self, victim_model: Model, d_model:int,
        dropout:float=0.0, **kwargs):

        super(TransDiscriminator, self).__init__()
        d_word_vec = victim_model.src_embed.embedding_dim
        # initiate embedding from victim model 
        self.src_embedding = Embeddings(
            embedding_dim=d_word_vec, scale=victim_model.src_embed.scale, 
            vocab_size=len(victim_model.src_vocab),
            padding_idx=victim_model.src_vocab.stoi[PAD_TOKEN],
            freeze=kwargs["freeze_embedding"])
        self.trg_embedding = Embeddings(
            embedding_dim=d_word_vec, scale=victim_model.trg_embed.scale, 
            vocab_size=len(victim_model.trg_vocab),
            padding_idx=victim_model.trg_vocab.stoi[PAD_TOKEN],
            freeze=kwargs["freeze_embedding"])

        self.src_gru = RecurrentEncoder(
            type="gru", hidden_size=d_model, 
            emb_size=d_word_vec, dropout=dropout, embed_dropout=dropout,
            bidirectional=True)
        self.trg_gru = RecurrentEncoder(
            type="gru", hidden_size=d_model, 
            emb_size=d_word_vec, dropout=dropout, emb_dropout=dropout,
            bidirectional=True)

        # twice of the bi-GRN dimension, concat the src and trg
        self.layer_norm = nn.LayerNorm(d_model*4, elementwise_affine=True)
        # whether the (x,y) is a translation pair
        self.ffn = nn.Linear(in_features=4*d_model, out_features=2)
        self.dropout = nn.Dropout(dropout)

        self.src_embedding.load_state_dict(victim_model.src_embed.state_dict())
        self.trg_embedding.load_state_dict(victim_model.trg_embed.state_dict())
        self._reset_parameters()

    def _reset_parameters(self,):
        # initialize the rest of the model parameters
        for weight in self.src_gru.parameters():
            rnn_init(weight)
        for weight in self.trg_gru.parameters():
            rnn_init(weight)
        default_init(self.ffn.weight)
        return

    def forward(self, 
        src:torch.Tensor, src_mask:torch.Tensor, src_length:torch.Tensor,
        trg:torch.Tensor, trg_mask:torch.Tensor, trg_length:torch.Tensor):
        """
        given src and trg, output classification label
        note that src and trg must be provided with sequences start with BOS, with corresponding len count and mask.
        :param src: batched padded input tokens [batch, len]. 
        :param src_mask: indicate valid token [batch, 1, len].
        :param src_length: [batch]
        :param trg*: batched trg tensor
        :return: labels indicating probability in shape [batch_size, 2]
        """
        x_emb = self.src_embedding(src)
        y_emb = self.trg_embedding(trg)

        # descendent reordering of batch src
        oidx, sidx, len_sorted = sort_batch(src_length)
        x_emb_sorted = torch.index_select(x_emb, index=sidx, dim=0)
        x_mask_sorted = torch.index_select(src_mask, index=sidx, dim=0)
        ctx_x_packed, _ = self.src_gru(x_emb_sorted, len_sorted, x_mask_sorted)
        # restore order
        ctx_x = torch.index_select(ctx_x_packed, dim=0, index=oidx)
        x_ctx_mean = (ctx_x * src_mask.squeeze().unsqueeze(2).float()).sum(1) / src_length.unsqueeze(1)

        # descendent reordering of batch trg
        oidx, sidx, len_sorted = sort_batch(trg_length)
        y_emb_sorted = torch.index_select(y_emb, index=sidx, dim=0)
        trg_mask_sorted = torch.index_select(trg_mask, index=sidx, dim=0)
        ctx_y_packed, _ = self.trg_gru(y_emb_sorted, len_sorted, trg_mask_sorted)
        # restore order
        ctx_y = torch.index_select(ctx_y_packed, dim=0, index=oidx)
        y_ctx_mean = (ctx_y * trg_mask.squeeze().unsqueeze(2).float()).sum(1) / trg_length.unsqueeze(1)
        
        output = self.layer_norm(torch.cat((x_ctx_mean, y_ctx_mean), dim=-1))
        output = F.softmax(self.ffn(self.dropout(output)), dim=-1)
        return output

