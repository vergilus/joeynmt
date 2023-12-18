# coding: utf-8
import argparse
import torch
import os
import yaml
import time
import codecs
from torch import Tensor
from torchtext.legacy.data import Iterator, Field
from adversarials.adversarial_utils import *
from adversarials.agent import Agent
from joeynmt.model import Model
from joeynmt.search import beam_search
from joeynmt.helpers import bpe_postprocess, expand_reverse_index
from joeynmt.data import MonoDataset
from joeynmt.helpers import bpe_postprocess
from plm_mt.utils import BartVocab


from transformers import MBart50Tokenizer
from subword_nmt import apply_bpe

from sacrebleu.metrics import BLEU
os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"
os.environ["OMP_NUM_THREADS"]="1"
os.environ["CUDA_LAUNCH_BLOCKING"]="1"

parser = argparse.ArgumentParser()
parser.add_argument("--source_path", type=str, 
    default="/home/data_ti6_d/zouw/data/cwmt17_zh-en_processed/subword_processed/newstest2018.sub.zh",
    # /home/zouw/pycharm_project_NMT_torch/adversarials/search2_attack_transformer_mt02
    help="the path for input files")
parser.add_argument("--batch_size", type=int, default=50,
    help="batch size for test")
parser.add_argument("--ckpt_path", type=str, 
    default="./adversarials/attack_cwmt_log",
    help="path to policy ckpt, contains configs")
parser.add_argument("--output_path", type=str, 
    default="./adversarials/attack_cwmt_log",
    help="output adversarial examples")
parser.add_argument("--unk_ignore", action="store_true", default=False,
    help="Don't replace target words using UNK (default as false)")
parser.add_argument("--use_gpu", action="store_true", default=False,
    help="whether to use GPU (default as False).")

time_format = '%Y-%m-%d %H:%M:%S'

def retokenize(victim_translator, bpe,
               origin_src:Tensor, perturbed_src:Tensor, 
               char2pyDict:dict, py2charDict:dict,
               output_as_token:bool=False):
    """
    retokenize the given src, if none, retokenize env src
    ignores the src BOS-id and valididate UNK tokens during the process
    :param perturbed_src: input ids [batch, len] (always with BOS might be longer then origin_src)
    :param origin_src: determine the UNK source 
    :return: retokenized padded src_ids, src_length, src_mask in 
        the same style of the initial batching process (default without BOS)
    """
    # print(origin_src.shape, perturbed_src.shape)
    if origin_src.shape[1]!=perturbed_src.shape[1]:
        perturbed_src_list = perturbed_src[:, 1:].detach().cpu().numpy().tolist()
    else:
        perturbed_src_list = perturbed_src.detach().cpu().numpy().tolist()
    # valid UNK tokens and detokenized sequence for another tokenization
    origin_src_list = origin_src.detach().cpu().numpy().tolist()
    retokenized_lines = []
    for l_index in range(len(perturbed_src_list)):  # for each line of input token ids  
        raw_line= []
        for id_index in range(len(perturbed_src_list[l_index])):
            id = perturbed_src_list[l_index][id_index]
            origin_id = origin_src_list[l_index][id_index]
            if victim_translator.src_vocab.itos[origin_id] not in [BOS_TOKEN, EOS_TOKEN, PAD_TOKEN]:
                # ignore BOS EOS and PAD, transfer to valid token
                if id ==victim_translator.src_vocab.stoi[UNK_TOKEN] and origin_id!=victim_translator.src_vocab.stoi[UNK_TOKEN]:
                    new_tok = gen_UNK(src_token=victim_translator.src_vocab.itos[origin_id],
                        vocab=victim_translator.src_vocab, 
                        char2pyDict=char2pyDict, py2charDict=py2charDict)
                    raw_line.append(new_tok)
                    if new_tok == UNK_TOKEN:
                        print(victim_translator.src_vocab.itos[origin_id],"generates:",new_tok)
                    # print(origin_id,"->",id, victim_translator.src_vocab.itos[origin_id],"->",new_tok)
                else:  
                    if victim_translator.src_vocab.itos[id] ==  UNK_TOKEN:
                        print("src UNK warning")
                    raw_line.append(victim_translator.src_vocab.itos[id])


        # cleanse and tokenize the new_line
        if "BPE" in bpe.__class__.__name__:
            raw_line = " ".join(raw_line)
            new_line = bpe_postprocess(raw_line, "subword-nmt").split(" ")
            retokenized_line = bpe.segment_tokens(new_line)
        elif "MBart" in bpe.__class__.__name__ :
            new_line = "".join(raw_line).replace("▁", " ").strip()
            seg_ids = bpe.encode(new_line)[1:-1]
            retokenized_line = bpe.convert_ids_to_tokens(seg_ids)

        # transform to ids and append with EOS
        if output_as_token: 
            retokenized_ids = [w for w in retokenized_line]
        else:
            retokenized_ids = [victim_translator.src_vocab.stoi[w] for w in retokenized_line] 
            retokenized_ids.append(victim_translator.src_vocab.stoi[EOS_TOKEN])
        retokenized_lines.append(retokenized_ids)
    if output_as_token:
        return retokenized_lines, None, None
    # batch process: generate src_length
    novel_src_length = [len(line) for line in retokenized_lines]  # with additional EOS
    novel_src_length = perturbed_src.long().new_tensor(novel_src_length)

    # pad the retokenized src
    pad_len = int(max(novel_src_length.cpu().numpy()))
    batch_size = perturbed_src.shape[0]
    PAD_id = victim_translator.src_vocab.stoi[PAD_TOKEN]
    novel_src = np.full((batch_size, pad_len), fill_value=PAD_id, dtype='int64')
    for k in range(batch_size):
        for kk in range(len(retokenized_lines[k])):
            novel_src[k][kk] = retokenized_lines[k][kk]
    novel_src = perturbed_src.new_tensor(novel_src)
    novel_src_mask = (novel_src != PAD_id).unsqueeze(1)
    return novel_src, novel_src_mask, novel_src_length

def translate(translator:Model, src:Tensor, src_mask:Tensor, src_length:Tensor):
    """
    fast translate the given tokenized inputs, if None, translate batch in the env
    returns the padded decoder ids
    """
    with torch.no_grad():
        # print("translate on: ",src.device)
        max_output_length = int(max(src_length.cpu().numpy())*1.5)
        # encode
        encoder_output, encoder_hidden, _, _ = translator(
            return_type="encode", 
            src=src, src_length=src_length,src_mask=src_mask
        )
        # beam-search for the best results (stacked_out as np.array)
        stacked_out, _ = beam_search(
            model=translator, size=5, 
            encoder_output=encoder_output, encoder_hidden=encoder_hidden,
            src_mask=src_mask, max_output_length=max_output_length,
            alpha=1.0, n_best=1
        )
    return stacked_out

def test_attack():
    # initiate configs files and saver
    args = parser.parse_args()
    print("yield to %s"%args.output_path)  
    
    if args.use_gpu:
        print("use gpu 0..")
        device="cuda:0"
    else:
        device="cpu"

    if os.path.exists(args.ckpt_path):
        with open(os.path.join(args.ckpt_path, "current_attack_configs.yaml"), "r") as current_configs_file:
            print("load configs from %s"%args.ckpt_path)
            configs = yaml.safe_load(current_configs_file)
        attack_configs = configs["attack_configs"]
    else:
        print("error, ckpt path not found!")
        return
    if "code_file_path" in attack_configs:
        bpe = apply_bpe.BPE(
            codes=codecs.open(attack_configs["code_file_path"], encoding='utf-8'),
            glossaries=[BOS_TOKEN,EOS_TOKEN,PAD_TOKEN,UNK_TOKEN])
    elif "plm_path" in attack_configs:
        bpe = MBart50Tokenizer.from_pretrained(attack_configs["plm_path"])
    else:
        print("lacks tokenizer models, require code_file_path or plm_path")
        return
    # load victim translator for vocabularies
    with open(attack_configs["victim_configs"], "r", encoding="utf-8") as victim_config_f:
        victim_configs = yaml.safe_load(victim_config_f)
    victim_model_path=attack_configs["victim_model"]
    data_configs = victim_configs["data"]

    # load vocab by the victim trianing config
    # initiate iterator padded with initiated length
    src_vocab_file=os.path.join(victim_configs["training"]["model_dir"], "src_vocab.txt")
    trg_vocab_file=os.path.join(victim_configs["training"]["model_dir"], "trg_vocab.txt")
    if "code_file_path" in attack_configs:
        src_vocab = Vocabulary(file=src_vocab_file)
        trg_vocab = Vocabulary(file=trg_vocab_file)
    elif "plm_path" in attack_configs:
        src_vocab = BartVocab(file=src_vocab_file)
        trg_vocab = BartVocab(file=src_vocab_file)
    tok_fun = lambda s: list(s) if data_configs["level"] == "char" else s.split()
    src_field = Field(init_token=None, eos_token=EOS_TOKEN, pad_token=PAD_TOKEN,
                      tokenize=tok_fun, batch_first=True, lower=data_configs["lowercase"],
                      unk_token=UNK_TOKEN, include_lengths=True)
    src_field.vocab = src_vocab
    
    test_data = MonoDataset(path=args.source_path, ext="", field=src_field)
    print("test src path:", args.source_path)
    test_iter = iter(Iterator(
            repeat=False, dataset=test_data, batch_size=args.batch_size,
            batch_size_fn=None, train=False, sort=False)
    )

    # build and load victim translator and vocab candidates for test
    local_victim_model = build_translate_model(
        victim_configs=victim_configs, victim_model_path=victim_model_path,
        src_vocab=src_vocab, trg_vocab=trg_vocab, device=device
    )
    local_victim_model.eval()
    BOS_id = local_victim_model.src_vocab.stoi[BOS_TOKEN]
    PAD_id = local_victim_model.src_vocab.stoi[PAD_TOKEN]

    word2near_vocab = load_or_extract_near_vocab(
        local_victim_model, 
        save_to=os.path.join(args.ckpt_path, "near_vocab"), 
        batch_size=100, top_reserve=attack_configs["knn"],
        reload=True, emit_as_id=True, 
    )
    # build adversarial agent and reload parameters
    agent_configs = configs["agent_configs"]
    local_agent = Agent(
        local_victim_model,
        **agent_configs["attacker_model_configs"]
    )
    local_agent.to(device)
    local_agent.eval()  # evaluation model pulls from global
    with open(os.path.join(args.ckpt_path,"ACmodel.ckpt")) as ckpt_file:
        ckpt_step = ckpt_file.readlines()[0]
        print("load ckpt from ", ckpt_step)
    agent_ckpt = torch.load(os.path.join(args.ckpt_path,ckpt_step),map_location=device)
    local_agent.load_state_dict(agent_ckpt["model"]) 

    # if attack involves Chinese UNK
    if not args.unk_ignore:
        if "pinyin_data" in attack_configs:
            assert os.path.exists(os.path.join(args.ckpt_path, "char2py.dict")) and \
                os.path.exists(os.path.join(args.ckpt_path, "py2char.dict")), "requires pinyin data for ckpt!"
            print("loading pinyin:")
            char2pyDict = yaml.safe_load(os.path.join(args.ckpt_path, "char2py.dict"))
            py2charDict = yaml.safe_load(os.path.join(args.ckpt_path, "py2char.dict"))
            # char2pyDict, py2charDict = collect_pinyin(
            #     pinyin_path=attack_configs["pinyin_data"],
            #     src_path=victim_configs["data"]["train"] + ".zh" ) 
            # with open(os.path.join(args.ckpt_path, "char2py.dict"), "w", encoding="utf-8") as charfile:
            #     yaml.dump(char2pyDict, charfile)
            # with open(os.path.join(args.ckpt_path, "py2char.dict"), "w", encoding="utf-8") as pyfile:
            #     yaml.dump(py2charDict, pyfile)
        else:
            char2pyDict, py2charDict = None, None
        print("finish UNK preparation.")

    total_adversary_lapse = 0
    # perturbed_inputs = []  # for translate perturbed tokens

    with torch.no_grad():
        batch_counter = 0
        with open(args.output_path,"w", encoding="utf-8") as perturbed_file,\
            open( args.output_path+".origin", "w", encoding='utf-8') as origin_file:
            for batch in test_iter:  # traverse the batch
                print("perturb %d"%(batch_counter*args.batch_size))
                batch_counter += 1
                
                t0 = time.time()
                batch = MyBatch(batch, pad_index=PAD_id, device=device)
                # sort batch src by length and keep track of order
                reverse_index = batch.sort_by_src_length()
                # sort_reverse_index = expand_reverse_index(reverse_index, n_best=1)
                
                # pad the BOS for adversary inputs, and translate without BOS
                BOS_vec = BOS_id * batch.src.new_ones(batch.src.shape[0]).unsqueeze(dim=-1)
                padded_src = torch.cat([BOS_vec, batch.src], dim=-1)
                # traverse the src index for all perturbed padded_src(index for padded_src)
                batch_size, max_steps = padded_src.shape
                for t in range(1, max_steps-1):
                    attack_out, critic_out = local_agent(
                        padded_src[:, 1:], batch.src_mask, batch.src_length,
                        padded_src[:, t-1:t+2]
                    )
                    actions = attack_out.argmax(dim=-1).detach()  # [batch]
                    # critic filters the action
                    action_mask = critic_out.gt(0).long().squeeze()  # [batch]
                    actions *= action_mask
                    # update padded src
                    target_of_step = []
                    for batch_index in range(batch_size):
                        word_id = padded_src[batch_index][t]
                        ## fast select similar candidate based on victim embedding
                        # target_word_id = word2near_vocab[word_id.item()][0]  #[np.random.choice(len(w2vocab[word_id.item()]), 1)[0]]
                
                        # select cosine-nearest candidate based on victim embedding
                        # choose similar candidates
                        origin_emb = local_agent.src_embedding(word_id)
                        candidates_emb = local_agent.src_embedding(padded_src.new_tensor(word2near_vocab[word_id.item()]))
                        nearest = candidates_emb.matmul(origin_emb)\
                            .div((candidates_emb*candidates_emb).sum(dim=-1))\
                            .argmax(dim=-1).item()   # cosine distance
                        target_word_id = word2near_vocab[word_id.item()][nearest]
                        if args.unk_ignore and src_vocab.is_unk(target_word_id.item()):
                            # undo this attack if UNK is set to be ignored
                            target_word_id = word_id.item()
                        target_of_step += [target_word_id]
                    # erase and update
                    padded_src[:, t] *= (1-actions)
                    padded_src[:, t] += padded_src.new_tensor(target_of_step) * actions 
                # valid UNK tokens and retokenize
                novel_src, _, _ = retokenize(
                    victim_translator=local_victim_model, bpe=bpe,
                    origin_src=batch.src, perturbed_src=padded_src,
                    char2pyDict=char2pyDict, py2charDict=py2charDict,
                    output_as_token=True)
                # print(novel_src)
                t1 = time.time()
                total_adversary_lapse+=(t1-t0)
                
                # detokenized sequence and output.
                for line_num in reverse_index:
                    perturbed_file.write(" ".join(novel_src[line_num])+"\n")
                    # this is the padded tensor object:
                    line_tok = []
                    for id in batch.src[line_num]:
                        tok = local_victim_model.src_vocab.itos[id.item()]
                        if tok not in [PAD_TOKEN, EOS_TOKEN, BOS_TOKEN]:
                            line_tok.append(tok)
                    origin_file.write(" ".join(line_tok)+"\n")
                # break  # for batch test
                # break

            
        m, s = divmod(total_adversary_lapse, 60)  # time format with hours, minutes, seconds
        h, m = divmod(m, 60)
        print('adversary time:%d:%02d:%02d...'%(h, m, s))


if __name__ == "__main__":
    test_attack()  # generates perturbed inputs