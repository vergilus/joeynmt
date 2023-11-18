# coding: utf-8
import numpy as np
import yaml
import matplotlib.pyplot as plt
import torch
import argparse
import os
from torchtext.legacy.data import Dataset, Iterator
from joeynmt.vocabulary import Vocabulary
from joeynmt.data import load_data
from joeynmt.batch import Batch
from joeynmt.search import beam_search
from joeynmt.helpers import bpe_postprocess
from joeynmt.constants import PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN
from adversarials.adversarial_utils import *
from transformers import MBartTokenizer, BartTokenizer, MBartModel
from sacrebleu.metrics import CHRF, TER
import openai
import requests
import time

def APItest():
    # initiate configs files and saver
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    if not os.path.exists(args.save_to):
        os.mkdir(args.save_to)
    
    with open(args.config_path, "r") as f, \
        open(os.path.join(args.save_to, "current_attack_configs.yaml"), "w") as current_configs:
        configs = yaml.safe_load(f)
        yaml.dump(configs, current_configs)
    attack_configs = configs["attack_configs"]
    with open(attack_configs["victim_configs"], "r", encoding="utf-8") as victim_config_f:
        victim_configs = yaml.safe_load(victim_config_f)
    # only load test data and vocabs
    # src_vocab_file=os.path.join(victim_configs["model_dir"], "src_vocab.txt")
    # trg_vocab_file=os.path.join(victim_configs["model_dir"], "trg_vocab.txt")
    _, dev_data, test_data, src_vocab, trg_vocab = load_data(
        data_cfg=victim_configs["data"], datasets=["test"])
    valid_iter = iter(Iterator(
            repeat=True, sort=False, dataset=dev_data,
            batch_size=50, batch_size_fn=None, 
            train=True, sort_within_batch=True,
            sort_key=lambda x: len(x.src), shuffle=True
        )
    )
    # build model
    victim_model_path=attack_configs["victim_model"]
    victim_translator = build_translate_model(
        victim_configs=victim_configs, victim_model_path=victim_model_path,
        src_vocab=src_vocab, trg_vocab=trg_vocab,
        device="cpu"
    )
    victim_translator.eval()

    with torch.no_grad():
        for test_batch in valid_iter:
            batch = Batch(test_batch,pad_index=victim_translator.src_vocab.stoi[PAD_TOKEN])
            if batch.nseqs<1:
                continue
            max_output_length = int(max(batch.src_length.cpu().numpy())*1.5)
            # encode
            encoder_output, encoder_hidden, _, _ = victim_translator(
                return_type="encode", 
                src=batch.src, src_length=batch.src_length, src_mask=batch.src_mask
            )
            stacked_out, _ = beam_search(
                model=victim_translator, size=5, 
                encoder_output=encoder_output, encoder_hidden=encoder_hidden,
                src_mask=batch.src_mask, max_output_length=max_output_length,
                alpha=1.0, n_best=1
            )
            # transit from id to bpe tokens
            decoded_valid = victim_translator.trg_vocab.arrays_to_sentences(stacked_out, cut_at_eos=True)
            reference_valid = victim_translator.trg_vocab.arrays_to_sentences(batch.trg, cut_at_eos=True)
            decoded_valid =[" ".join(t) for t in decoded_valid] 
            reference_valid = [" ".join(t) for t in reference_valid] 
            # transit from bpe tokens to seqs for BLEU
            hyps = [bpe_postprocess(s, "subword-nmt") for s in decoded_valid]
            refs = [bpe_postprocess(s, "subword-nmt") for s in reference_valid]

            # calculate BLEU
            break

def check_unk(src_vocab:Vocabulary):
    atoken = gen_UNK("barbu", src_vocab, None, None)
    print(atoken)
    atoken = gen_UNK("en@@", src_vocab, None, None)
    print(atoken)
    atoken = gen_UNK("homo@@", src_vocab, None, None)
    print(atoken)
    atoken = gen_UNK("substitution", src_vocab, None, None)
    print(atoken)
    atoken = gen_UNK("thing", src_vocab,None, None)
    print(atoken)
    return 

def check_mbart_vocab(plm_path:str, src_vocab):
    plm_tokenizer = BartTokenizer.from_pretrained(plm_path)
    plm_vocab = plm_tokenizer.get_vocab()
    count=0
    for tok in src_vocab.stoi:
        if tok not in plm_vocab:
            print(tok)
            count+=1
    return count

def check_charF1(src_path, perturbed_src_path):
    """
    Gu@@ ta@@ ch : Incre@@ ased safety for pedestri@@ ans
    """
    ter = TER()
    charf = CHRF()
    with open(src_path, "r") as origin_src, open(perturbed_src_path, "r") as perturbed_src:
        processed_src=[]
        processed_perturbed_src=[]
        for src_line, perturbed_src_line in zip(origin_src, perturbed_src):
            processed_src.append([src_line.replace("@@ ", "")])
            processed_perturbed_src.append(perturbed_src_line.replace("@@ ", ""))
            print(src_line.replace("@@ ", ""))
            print(perturbed_src_line.replace("@@ ", ""))

        charf1_score = charf.corpus_score(hypotheses=processed_perturbed_src, references=processed_src).score
        ter_score=ter.corpus_score(hypotheses=processed_perturbed_src, references=processed_src).score
    print("charf=%f ,ter=%f"%(charf1_score, ter_score))
    return charf1_score, ter_score

def check_semantic(src_path, trg_path):
    def ask_gpt(instruction: str):
        messages = [
            {
                'role': 'system',
                'content': 'You are a knowledgable multi-lingual specialist who can tell whether two sentences (might be different languages)  are semantically matched  by only "yes" or "no".'
            },
            {
                'role': 'user',
                'content': instruction
            }
        ]
        respond = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages = messages,
        )
        return respond.choices[0].message.content

    openai.api_base = "https://api.closeai-proxy.xyz/v1"  # "https://api.closeai-proxy.xyz/v1"
    openai.api_key = "sk-ytCw7vhtPgu9iUy9uJEmqKOpBY8ChXjpKlI9k51iQ5M0a4hy" #"sk-hJWcfgy17p0KXkM8j1tGE4IPr9PRlWqEPMAYHhTUD7bp78gr"
    t0=time.time()
    with open(src_path, "r") as src, open(trg_path, "r") as trg:
        neg_count=0.
        total_count=0
        for src_line, trg_line in zip(src, trg):
            test_line = src_line.replace("@@ ", "") + \
                trg_line.replace("@@ ", "") + \
                    "are these two sentences matched?"
            total_count+=1
            try:
                bot_answer = ask_gpt(instruction=test_line)
                print(bot_answer)
                if "No" in bot_answer:
                    print(src_line, trg_line)
                    neg_count+=1
            except Exception as e:
                print(e,"failed:"+test_line)
            # break
        print("fail_rate=", neg_count,total_count, neg_count/total_count)
        t1 = time.time()
        total_adversary_lapse=(t1-t0)
        m, s = divmod(total_adversary_lapse, 60)  # time format with hours, minutes, seconds
        h, m = divmod(m, 60)
        print('adversary time:%d:%02d:%02d...'%(h, m, s))

        return neg_count/len(trg_path)

def check_semantic_baichuan(src_path, trg_path):
    baiochuan_url="http://210.28.134.152:8010/baichuan2/"
    def ask_baichuan(instruction:str):
        messages = [
            {
                'role': 'system',
                'content': 'You are a knowledgable multi-lingual specialist who can tell whether two sentences (might be different languages)  are semantically matched  by only "yes" or "no".'
            },
            {
                'role': 'user',
                'content': instruction
            }
        ]
        return requests.get(baiochuan_url+"{{%s}}"%test_line).json()["message"]
    
    with open(src_path, "r") as src, open(trg_path, "r") as trg:
        neg_count=0.
        total_count=0
        for src_line, trg_line in zip(src, trg):
            test_line = src_line.replace("@@ ", "") + \
                trg_line.replace("@@ ", "") + \
                    "are these two sentences matched?"
            total_count+=1
            try:
                bot_answer = ask_baichuan(trg_line)
                print(bot_answer)
                if "No" in bot_answer:
                    print(src_line, trg_line)
                    neg_count+=1
            except Exception as e:
                print("failed:"+test_line)
        print("fail_rate=", neg_count/len(trg_path))

        return neg_count/len(trg_path)

def count_nearest(nearest_vocab_path:str):
    word2near_vocab = OrderedDict()
    max_index = 500
    with open(nearest_vocab_path, "r") as in_file:
        total_count = 0.
        index = 0
        for line in in_file:
            line = line.strip().split("\t")
            word2near_vocab[line[0]] = line[1:]
            index += 1
            if index > max_index:
                break
            else:
                print(word2near_vocab[line[0]])
                total_count+=len(word2near_vocab[line[0]])
        print(total_count)
    return  total_count/max_index

if __name__=="__main__":
    base_dir="/home/nfs01/zouw/policy/attack_bart_en2de_log/"
    src_path = os.path.join(base_dir, "in_origin.14")
    trg_path = "/home/nfs01/zouw/data/cwmt17_zh-en_processed/subword_processed/newstest2018.sub.de"
    perturbed_src_path = os.path.join(base_dir, "in_perturbed.14")
    # chaf, ter = check_charF1(
    #     src_path=src_path,
    #     perturbed_src_path=perturbed_src_path
    # )
    print(count_nearest("/home/nfs01/zouw/policy/attack_en2zh_split_log/near_vocab"))
    # check_semantic(src_path, trg_path)
    # check_semantic_baichuan(src_path, trg_path)

    # ckpt_model_path = "/home/nfs01/zouw/models/wmt_ende_TF_best/best.ckpt"
    # ckpt_config_path = "/home/nfs01/zouw/models/wmt_ende_TF_best/config.yaml"
    # save_to = "./adversarials/attack_cwmt_log"
    # if not os.path.exists(save_to):
    #     os.mkdir(save_to)
    # # load configs
    # with open(ckpt_config_path, "r", encoding="utf-8") as victim_config_f:
    #     victim_configs = yaml.safe_load(victim_config_f)
    # # load model
    # src_vocab_file=os.path.join(victim_configs["training"]["model_dir"], "src_vocab.txt")
    # trg_vocab_file=os.path.join(victim_configs["training"]["model_dir"], "trg_vocab.txt")
    # src_vocab = Vocabulary(file=src_vocab_file)
    # trg_vocab = Vocabulary(file=trg_vocab_file)
    # print(check_mbart_vocab("/home/nfs01/zouw/models/huggingface/bart-base/", src_vocab))
    # print(len(src_vocab.stoi))
    
    # check_unk(src_vocab=src_vocab)    
    # victim_translator = build_translate_model(
    #     victim_configs=victim_configs,
    #     victim_model_path=ckpt_model_path,
    #     src_vocab=src_vocab, trg_vocab=trg_vocab,
    #     device="cpu"
    # )
    # victim_translator.eval()
    # test KNN candidate
    # word2near_vocab = load_or_extract_near_vocab(
    #     victim_model=victim_translator, 
    #     save_to=os.path.join(save_to, "near_vocab"),
    #     batch_size=50,top_reserve=10,
    #     reload=False)
    # word2near_vocab = load_or_extract_near_vocab_normed(
    #     victim_model=victim_translator, 
    #     save_to=os.path.join(save_to, "near_vocab_norm"),
    #     batch_size=50,top_reserve=10,
    #     reload=False
    # )

    # _, valid_data, _, _, _ = load_data(
    #     data_cfg=victim_configs["data"], datasets=["dev"])
    
    # valid_iter = iter(Iterator(
    #     repeat=True, sort=False, dataset=valid_data,
    #     batch_size=10, batch_size_fn=None, 
    #     train=True, sort_within_batch=True,
    #     sort_key=lambda x: len(x.src), shuffle=True
    # ))

    # batch = next(valid_iter)
    # src, src_length = batch.src
    # print(src_length[0], src[0])
    # trg, trg_length = batch.trg
    # print(trg_length[0], trg[0])
