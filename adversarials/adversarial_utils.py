# coding: utf-8
from collections import OrderedDict
import torch
import os
import torch.nn as nn
import numpy as np
import math
from joeynmt.constants import PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN
from joeynmt.helpers import load_checkpoint
from joeynmt.model import Model, build_model
from joeynmt.vocabulary import Vocabulary

# keyboard vicinities
char_swap_dict = {
    "a": ["q","s","z","x","w"], 
    "b": ["v","n","g","h",], 
    "c": ["d","x","v","f"],
    "d": ["s", "e", "r", "f", "v", "c", "x"],
    "e": ["w","r","d","f","s"],  
    "f": ["d", "r", "t", "g", "v", "c"],
    "g": ["f", "t", "y", "h", "b", "v","q","9"],
    "h": ["g", "y", "u", "j", "n", "b"],
    "i": ["l","k","j","u","o","8", "9"], 
    "j": ["h", "u", "i", "k", "m", "n"],
    "k": ["j", "i", "o", "l", ".", "," , "m"],
    "l": ["i","I", "k","o","p",";",".",","],
    "m": ["n", "j", "k", ","],
    "n": ["b", "h", "j", "m"],
    "o": ["i", "0", "p", ";", "l", "k"],
    "p": ["o", "0", "-", "[", "'", ";", "l"],
    "q": ["w", "s", "a"],
    "r": ["e", "t", "g", "f", "d"],
    "s": ["a", "w", "e", "d", "c", "x", "z"],
    "t": ["r", "y", "h", "g", "f"],
    "u": ["y", "i", "k", "j", "h"],
    "v": ["c", "f", "g", "b"],
    "w": ["q", "e", "d", "s", "a"],
    "x": ["z", "s", "d", "c"],
    "y": ["g", "t", "u", "j", "h"],
    "z": ["a", "s", "x"], 
}

def sort_batch(seq_len):
    """Sorts torch tensor of integer indices by decreasing order."""
    with torch.no_grad():
        slens, sidxs = torch.sort(seq_len, descending=True)
    oidxs = torch.sort(sidxs)[1]

    return oidxs, sidxs, slens

def check_unk_ratio(src_vocab:Vocabulary, batch_of_ids) -> float:
    UNK_id = src_vocab.stoi[UNK_TOKEN]
    unk_count = batch_of_ids.eq(UNK_id).sum().float()
    total_count = torch.numel(batch_of_ids) 
    ratio = (unk_count/total_count).item()
    # print("unk_rate:",ratio)
    return ratio

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

def build_translate_model(
    victim_configs:dict, victim_model_path:str, 
    src_vocab:Vocabulary, trg_vocab:Vocabulary, 
    device:str) ->Model:
    # build the translator for environments
    translator = build_model(
        victim_configs["model"], 
        src_vocab=src_vocab, trg_vocab=trg_vocab)
    # reload model from model_path
    model_ckpt = torch.load(victim_model_path, map_location=device)
    translator.load_state_dict(model_ckpt["model_state"])

    return translator.to(device)

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
            line = line.strip().split("\t")
            if emit_as_id:
                word2near_vocab[src_vocab.stoi[line[0]]] = [src_vocab.stoi[tok] for tok in line[1:]]
            else:
                word2near_vocab[line[0]] = line[1:]
    return word2near_vocab

def load_or_extract_near_vocab(
    victim_model:Model, 
    save_to:str, batch_size:int=100, top_reserve:int=12,
    reload:bool=True, emit_as_id:bool=False, all_with_unk=False
)->dict:
    """
    extract nearest token candidates for the perturbation. 
    will append UNK as additional candidates.
    saves the near_vocab to the save_to directory
    :param all_with_unk: every candidate contains UNK to trigger char-level perturbation exploration 
    return the near_vocab dict
    """
    print("extract nearest candidates to %s"%save_to)
    with torch.no_grad():
        src_vocab = victim_model.src_vocab
        emb = victim_model.src_embed.lut.weight.detach().clone()  # the exact nn.Embeddings.weight
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
                    # print("avg Euclidean distance:", E_dist.mean())
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
                    topk_indices = similarity.topk(top_reserve, dim=1)[1]
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
                        if src_word in [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]:
                            # no additional candidates for reserved token
                            near_cand_id = src_word_id
                            near_cand = src_word
                            similar_vocab.write(near_cand+"\t")
                            bingo = 1
                            if emit_as_id:
                                near_vocab+=[near_cand_id]
                            else:
                                near_vocab+=[near_cand]
                        else:
                            # k nearest candidates, the nearest might be the <pad> or itself
                            for k in range(1, topk_val.shape[1]):
                                near_cand_id = topk_indices[j][k]
                                near_cand = src_vocab.itos[near_cand_id]
                                if (near_cand not in [PAD_TOKEN, EOS_TOKEN, BOS_TOKEN, src_word])\
                                  and topk_val[j][k]<avg_dist:
                                    bingo += 1
                                    similar_vocab.write(near_cand+"\t")
                                    if emit_as_id:
                                        near_vocab+=[near_cand_id]
                                    else:
                                        near_vocab+=[near_cand]
                            if bingo==0 or all_with_unk:
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
Implementation of a mini-batch with less components.
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

def collect_pinyin(pinyin_path:str, src_path:str):
    # generate pinyin for every Chinese characters in training data
    """
    read from pinyin_path to generate pinyin dictionary
    :param pinyin_path: path to pin data file
    :param src_path: chinese src data path to collect
    :return: two dictionary of pinyin2char:{pinyin: [list of characters]},
             and char2pinyin: {ord(char): [list of pinyin]}
    """
    char2pyDict = {}
    py2charDict = {}
    count_char = {}
    for line in open(pinyin_path):
        k, v = line.strip().split('\t')
        char2pyDict[k] = v.split(" ")  # there can be multiple values(pinyin) for a key

    with open(src_path, "r") as input_src:
        line_counter = 0
        for line in input_src:
            line_counter += 1
            # if line_counter%1000 == 0:
            #     break
            # collect characters and their pinyin
            for char in line.strip():
                key = "%X" % ord(char)
                if char in count_char:
                    count_char[char] += 1
                else:
                    count_char[char] = 1
                try:
                    for pinyin in char2pyDict[key]:
                        pinyin = pinyin.strip()  # .lower()
                        if pinyin in py2charDict:
                            if char not in py2charDict[pinyin]:
                                py2charDict[pinyin].append(char)
                        else:
                            py2charDict[pinyin] = [char]
                except:  # special char without pinyin
                    continue
    return char2pyDict, py2charDict


def gen_UNK(src_token, vocab:Vocabulary,
            char2pyDict:dict, py2charDict:dict, silence=True):
    """
    targeted for BPE sub-tokens, which is segmented by `@@` by default.
    when src_token is to be replaced by UNK, generate a valid UNK
    token (by given src vocab) 
    replace: homophone-based or nearby candidate replacement
    swap: swap places of two adjascent charaters in the token
    insert: traverse to repeat only one character.
    delete: as the last resort

    :param src_token: chinese src_token to be replaced by UNK
    :param vocab: data.vocabulary object to varify if result is UNK
    :param char2pyDict: dictionary {ord(char): pinyin}
    :param py2charDict: dictionary {pinyin}
    :return: a UNK word similar to src_token
    """
    reserved_char = ["▁", "@@", "_@", "%", PAD_TOKEN, EOS_TOKEN, BOS_TOKEN]
    if src_token in reserved_char:  # there is nothing to perturb!
        return src_token   
    
    edit_range = len(src_token)
    if src_token.endswith("@@"):  # don't break the segment token for BPE
        edit_range -= 2
    elif src_token.endswith("%"):  # don't break the segment token for sentencepiece
        edit_range -= 1 

    if (char2pyDict is not None) and (py2charDict is not None):
        index = np.random.randint(edit_range)
        # homophone-based char replace
        for _ in range(edit_range):  # traverse the tokens
            ori_char = src_token[index]
            new_token = src_token
            py_key = "%X" % ord(ori_char)
            if (py_key in char2pyDict):
                # this character is available in gen_UNK
                for pinyin in char2pyDict[py_key]:
                    # check for every possible vocal
                    for candidate in py2charDict[pinyin]:
                        # check for every character share this vocal
                        new_token = list(new_token)
                        new_token[index] = candidate
                        new_token = "".join(new_token)
                        if candidate != ori_char and vocab.is_unk(new_token):
                            return new_token
            index = (index+1) % edit_range
    else:  
        # roman character replacement to generate unk scramble the symble in random place
        if edit_range > 3:
            index = np.random.randint(0, edit_range-2)
            for _ in range(len(src_token)):
                if src_token[index] in reserved_char:  # sample again
                    if not silence:
                        print("swap", src_token)
                    index = np.random.randint(0, edit_range-2)
                else:  # valid for perturb
                    new_token = src_token[:index] + \
                        src_token[index+1]+src_token[index]+\
                        src_token[index+2:]
                    if vocab.is_unk(new_token):
                        return new_token
                    else:
                        index = np.random.randint(0, edit_range-2)
                        continue
        else:  # too short, able to swap by the keyboard vicinity
            index = np.random.randint(edit_range)
            for _ in range(len(src_token)):  # sample at most times 
                if src_token[index] in reserved_char: # sample again
                    if not silence: 
                        print("keyboard", src_token)
                    index = (index+1) % edit_range
                else:   # valid for perturb
                    ori_char = src_token[index]
                    new_token = list(src_token)
                    if ori_char in char_swap_dict:
                        trg_char_cand = char_swap_dict[ori_char]
                        new_token[index] = trg_char_cand[np.random.randint(len(trg_char_cand))]
                        new_token = "".join(new_token)
                        if vocab.is_unk(new_token):
                            return new_token
                        else:
                            index = (index+1) % edit_range
                            continue

    # still nothing returned: insert by random repetition
    index = np.random.randint(edit_range)
    for _ in range(len(src_token)):
        if src_token[index] in reserved_char:  # sample again
            if not silence:
                print("insert replace", src_token)
            index = (index+1) % edit_range
        else:   # valid for perturb
            ori_char = src_token[index]
            token_stem1= src_token[:index]
            token_stem2= src_token[index:edit_range]
            # char = src_token[edit_range - 1]
            # token_stem = src_token[:edit_range]
            main_token = token_stem1 + ori_char + token_stem2
            if src_token.endswith("@@"):
                main_token = main_token+"@@"
            elif src_token.endswith("%"):
                main_token = main_token+"%"

            if vocab.is_unk(main_token):
                return main_token
            else:  # try next insertion position
                index = (index+1) % edit_range
                continue
    
    # repeat last char
    last_char = src_token[edit_range-1]
    main_token = src_token[:edit_range]
    for _ in range(edit_range):
        main_token = main_token + last_char
        if src_token.endswith("@@"):
            main_token += "@@"
        elif src_token.endswith("%"):
            main_token += "%"

        if vocab.is_unk(main_token):
            return main_token
        
    # print(src_token, "-->", new_token)
    return src_token

