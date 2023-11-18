# coding: utf-8
import argparse
import torch
import os
import yaml
import time
from torch import Tensor
from torchtext.legacy.data import Dataset, Iterator, Field

from reinforces.reinforce_utils import *
from reinforces.agent import Agent

from joeynmt.loss import XentLoss
from joeynmt.data import MonoDataset, load_data, token_batch_size_fn
from joeynmt.search import beam_search
from joeynmt.batch import Batch
from joeynmt.helpers import bpe_postprocess, expand_reverse_index
from sacrebleu.metrics import BLEU

os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"
os.environ["OMP_NUM_THREADS"]="1"
os.environ["CUDA_LAUNCH_BLOCKING"]="1"

parser = argparse.ArgumentParser()
parser.add_argument("--source_path", type=str, 
    default="/data0/zouw/data/cwmt17_zh-en_processed/subword_processed/newstest2018.sub.zh",
    help="the path for input files")
parser.add_argument("--batch_size", type=int, default=40,
    help="batch size for test")
parser.add_argument("--ckpt_path", type=str, 
    default="./reinforces/reinforce_cwmt_log",
    help="reinforce ckpt (model) and configs path")
parser.add_argument("--output_path", type=str,
    default="./reinforces/reinforce_cwmt_log",
    help="output path")
parser.add_argument("--use_gpu", action="store_true", default=False,
    help="whether to use GPU (default as False).")

time_format = '%Y-%m-%d %H:%M:%S'

def translate(translator: nn.Module, src_emb:Tensor, src_mask:Tensor, src_length:Tensor):
    """
    translate the given perturbed representation (beam-search)
    translator is the model without embedding layer.

    return: the best final result from beam-search 
    """
    with torch.no_grad():
        # print("translate on: ",src.device)
        max_output_length = int(max(src_length.cpu().numpy())*1.5)
        # encode
        encoder_output, encoder_hidden, _, _ = translator(
            return_type="encode", 
            src_emb=src_emb, src_length=src_length,src_mask=src_mask
        )
        # beam-search for the best results (stacked_out as np.array)
        stacked_out, _ = beam_search(
            model=translator, size=5, 
            encoder_output=encoder_output, encoder_hidden=encoder_hidden,
            src_mask=src_mask, max_output_length=max_output_length,
            alpha=1.0, n_best=1
        )
    return stacked_out

def test_reinforce(max_reinforce_rounds:int=100):
    """
    reinforce the inputs embeddings and yield the outputs to file
    :param max_reinforce_rounds: maximum steps for denoising
    :param noising_ratio: the ratio of noising steps by the given denoising steps. 
    
    """
    # load configs
    args = parser.parse_args()
    if not os.path.exists(args.output_path):
        print("making output path...")
        os.mkdir(args.output_path)
    if args.use_gpu:
        print("using gpu 0...")
        device = "cuda:0"
    else:
        device = "cpu"

    with open(os.path.join(args.ckpt_path, "current_reinforce_configs.yaml"), "r") as f:
        configs = yaml.safe_load(f)

    reinforce_configs = configs["reinforce_configs"]
    with open(reinforce_configs["victim_configs"], "r", encoding="utf-8") as victim_config_f:
        victim_configs = yaml.safe_load(victim_config_f)
    data_configs = victim_configs["data"]
    # load data and dictionary from victim configs
    src_vocab_file=os.path.join(victim_configs["training"]["model_dir"], "src_vocab.txt")
    trg_vocab_file=os.path.join(victim_configs["training"]["model_dir"], "trg_vocab.txt")
    src_vocab = Vocabulary(file=src_vocab_file)
    trg_vocab = Vocabulary(file=trg_vocab_file)
    BOS_id = src_vocab.stoi[BOS_TOKEN]
    PAD_id = src_vocab.stoi[PAD_TOKEN]

    tok_fun = lambda s: list(s) if data_configs["level"] == "char" else s.split()
    src_field = Field(init_token=None, eos_token=EOS_TOKEN, pad_token=PAD_TOKEN,
                      tokenize=tok_fun, batch_first=True, lower=data_configs["lowercase"],
                      unk_token=UNK_TOKEN, include_lengths=True)
    src_field.vocab = src_vocab

    print("load data %s..."%args.source_path, end=" ")
    test_data = MonoDataset(path=args.source_path, ext="", field=src_field)
    test_iter = iter(Iterator(
            repeat=False, dataset=test_data, batch_size=args.batch_size,
            batch_size_fn=None, train=False, sort=False)
    )  # iterator does not sort the batch by sequence length
    print(" finished")

    # build and reload victim model and reinforce policy model
    victim_model_path = reinforce_configs["victim_model"]
    local_src_embed, local_victim_translator = build_embedding_and_NMT(
        victim_configs=victim_configs["model"], victim_model_path=victim_model_path,
        src_vocab=src_vocab, trg_vocab=trg_vocab,
        device=device
    )
    local_src_embed.eval()
    local_victim_translator.eval()

    with torch.no_grad():
        weight = local_src_embed.lut.weight.detach().clone()
        if local_src_embed.scale:
            weight*=math.sqrt(weight.shape[1])
    emb_mean = weight.mean()
    emb_std = weight.std()

    # load the trained policy
    with open(os.path.join(args.ckpt_path,"ACmodel.ckpt")) as ckpt_file:
        ckpt_step = ckpt_file.readlines()[0]
        print("load ckpt from ", ckpt_step)
    agent_ckpt = torch.load(os.path.join(args.ckpt_path,ckpt_step),map_location=device)
    agent_configs = configs["agent_configs"]
    local_agent = Agent(
            embedding_dim=victim_configs["model"]["encoder"]["embeddings"]["embedding_dim"],
            **agent_configs["agent_model_configs"])
    local_agent.load_state_dict(agent_ckpt["agent"])
    local_agent.to(device)
    local_agent.eval()
    print("load policy... finished")
    total_reinforce_lapse = 0

    total_origin_outputs = []
    total_reinforce_outputs = []
    batch_counter = 0
    for batch in test_iter:
        batch_counter += 1
        # preprocess the batch src inputs 
        batch = Batch(batch, pad_index=PAD_id, use_cuda=True)
        batch.to(device)
        reverse_index = batch.sort_by_src_length()
        sort_reverse_index = expand_reverse_index(reverse_index, n_best=1)

        origin_src_emb = local_src_embed(batch.src)
        reinforced_src_emb = origin_src_emb.detach().clone() 
        
        src_mask = batch.src_mask  # [batch, 1, len] 
        src_length = batch.src_length

        t0=time.time()
        # batched states for diffusion policy
        normed_input = (reinforced_src_emb - emb_mean)/emb_std

        # embedding_V = local_agent.get_V(
        #         normed_input, src_mask, src_length,
        #         i*torch.ones_like(src_length)
        #     )
        # print(i, "estimate degradation:", embedding_V.mean())

        # sample initial data points f_\theta(\hat{x})
        # mu, std = local_agent.forward(
        #     normed_input, src_mask, src_length,
        #     noising_round*torch.ones_like(src_length)
        # )
        with torch.no_grad():
            # initial guesss on the flow
            normed_input = local_agent.denoise(
                normed_input, src_mask, src_length
            )
            for i in range(max_reinforce_rounds):
                embedding_V=local_agent.get_V(
                    normed_input, src_mask, src_length
                )
                print("%d value:"%i, embedding_V.mean())
                noised_input, _, _ = local_agent.sample_noise(
                    normed_input, src_mask, src_length
                )
                ###############
                L2_dist = ((normed_input-noised_input)**2).sum(dim=-1).sum(dim=-1).mean()
                print(L2_dist)
                ###############
                embedding_V=local_agent.get_V(
                    noised_input, src_mask, src_length
                )
                print("%d noised value:"%i, embedding_V.mean())
                """ a larger V indicates possible degradations
                the denoise will mask the perturbation that increase the V
                """
                temp_normed_input = local_agent.denoise(
                    noised_input, src_mask, src_length,
                )
                temp_embedding_V=local_agent.get_V(
                    normed_input, src_mask, src_length
                )
                rectify_mask = (temp_embedding_V-embedding_V).le(0).float()
                normed_input = (1-rectify_mask) * normed_input+(rectify_mask)*temp_normed_input
                
                # normed_input=temp_normed_input
                # print("%d rectified value:"%i, embedding_V.mean())
            # restore the embeddings
            reinforced_src_emb = normed_input*emb_std+emb_mean
        t1=time.time()
        total_reinforce_lapse += (t1-t0)
        with torch.no_grad(): 
            # translate the origin batch
            origin_out = translate(local_victim_translator, origin_src_emb, src_mask, src_length)
            total_origin_outputs.extend(origin_out[sort_reverse_index])
            # translate the reinforced batch
            reinforced_out = translate(local_victim_translator, reinforced_src_emb, src_mask, src_length)
            total_reinforce_outputs.extend(reinforced_out[sort_reverse_index]) 
            print("perturb %d sents:"%(batch_counter * args.batch_size), (origin_src_emb-reinforced_src_emb).mean().item())
        break

    # yield the outputs to file 
    with open(os.path.join(args.output_path, "out_origin"), "w") as out_file0, \
        open(os.path.join(args.output_path, "out_reinforced%d"%max_reinforce_rounds), "w") as out_file1:
        # post process
        for (sent, reinforced_sent) in zip(total_origin_outputs, total_reinforce_outputs):
            origin_decode = trg_vocab.array_to_sentence(sent, cut_at_eos=True)
            origin_decode = " ".join(origin_decode) 
            reinforced_decode = trg_vocab.array_to_sentence(reinforced_sent, cut_at_eos=True)
            reinforced_decode = " ".join(reinforced_decode)

            origin_hyp = bpe_postprocess(origin_decode, "subword-nmt")
            reinforced_hyp = bpe_postprocess(reinforced_decode, "subword-nmt")
            out_file0.write(origin_hyp+"\n")
            out_file1.write(reinforced_hyp+"\n")

    print("total reinforce lapse:", total_reinforce_lapse)
    return


def bleu_variant(max_perturbation_rounds:int=100):
    """
    collect bleu variants within each linear-scheduled diffusion transition
    """

    # load configs
    args = parser.parse_args()
    if not os.path.exists(args.output_path):
        print("making output path...")
        os.mkdir(args.output_path)
    if args.use_gpu:
        print("using gpu 0...")
        device = "cuda:0"
    else:
        device = "cpu"
    
    interval = 0.01
    print("step interval:",interval)


    with open(os.path.join(args.ckpt_path, "current_reinforce_configs.yaml"), "r") as f:
        configs = yaml.safe_load(f)

    reinforce_configs = configs["reinforce_configs"]
    with open(reinforce_configs["victim_configs"], "r", encoding="utf-8") as victim_config_f:
        victim_configs = yaml.safe_load(victim_config_f)
    
    data_configs = victim_configs["data"]
    # load data from victim configs
    _, dataset, _, _, _ = load_data(
        data_cfg=victim_configs["data"], datasets=["dev"])
    data_iter = iter(Iterator(
            repeat=False, sort=False, dataset=dataset,
            batch_size=50, batch_size_fn=None, 
            train=True, sort_within_batch=True,
            sort_key=lambda x: len(x.src), shuffle=False
        ))
    print("load data... finished")

    # build and reload victim model and reinforce policy model
    victim_model_path = reinforce_configs["victim_model"]
    src_vocab_file=os.path.join(victim_configs["training"]["model_dir"], "src_vocab.txt")
    trg_vocab_file=os.path.join(victim_configs["training"]["model_dir"], "trg_vocab.txt")
    src_vocab = Vocabulary(file=src_vocab_file)
    trg_vocab = Vocabulary(file=trg_vocab_file)
    BOS_id = src_vocab.stoi[BOS_TOKEN]
    PAD_id = src_vocab.stoi[PAD_TOKEN]
    local_src_embed, local_victim_translator = build_embedding_and_NMT(
        victim_configs=victim_configs["model"], victim_model_path=victim_model_path,
        src_vocab=src_vocab, trg_vocab=trg_vocab,
        device=device
    )
    local_src_embed.eval()
    local_victim_translator.eval()

    with torch.no_grad():
        weight = local_src_embed.lut.weight.detach().clone()
        if local_src_embed.scale:
            weight*=math.sqrt(weight.shape[1])
    emb_mean = weight.mean()
    emb_std = weight.std()

    total_origin_outputs = []
    total_reinforce_outputs = []

    with open("bleu_variant_log", "w") as bleu_log:
        # initiation
        batch  = next(data_iter)
        batch = Batch(batch, pad_index=PAD_id, use_cuda=True)
        batch.to(device)

        reverse_index = batch.sort_by_src_length()
        sort_reverse_index = expand_reverse_index(reverse_index, n_best=1)
        
        initial_src_emb = local_src_embed(batch.src)
        src_mask = batch.src_mask  # [batch, 1, len] 
        src_length = batch.src_length
        perturbed_src_emb = initial_src_emb.detach().clone()
        refs = []
        for _, ref in enumerate(batch.trg):
            ref = " ".join( [trg_vocab.itos[id] for id in ref.tolist() if id not in [1,2,3]])
            refs.append([ref])

        metric = BLEU()
        stds = torch.linspace(0.000, 0.3, max_perturbation_rounds, dtype=torch.float64).cpu().numpy().tolist()
        for i in range(max_perturbation_rounds):
            origin_out = translate(local_victim_translator, perturbed_src_emb, src_mask, src_length)
            
            normed_emb = (perturbed_src_emb-emb_mean) /emb_std
            perturbed_emb = normed_emb*math.sqrt(1-stds[i]**2) + stds[i]*torch.randn_like(normed_emb)
            perturbed_src_emb = perturbed_emb*emb_std+emb_mean

            perturbed_out = translate(local_victim_translator,  perturbed_src_emb, src_mask, src_length)

            # calculate BLEU
            process_origin_hyps = []
            process_perturbed_hyps = []
            for j, hyp in enumerate(origin_out):
                process_origin_hyps.append(" ".join([trg_vocab.itos[id] for id in hyp if id not in [1,2,3]])) 
                process_perturbed_hyps.append(" ".join([trg_vocab.itos[id] for id in perturbed_out[j] if id not in [1,2,3]]))
            # print(process_origin_hyps, refs)
            bleu_value=metric.corpus_score(process_perturbed_hyps, refs).score
            print(i, ":",bleu_value)
            bleu_log.write(str(bleu_value)+"\n")
            # calculate BLEU
            # perturb_bleu_value=metric.corpus_score([], [[]])

            # print(origin_out[0])
            # print(batch.trg[0].tolist())

    return

def loss_variant(max_perturbation_rounds:int=100):
    """
    collect loss variants within each linear-scheduled diffusion transition
    """
    # load configs
    args = parser.parse_args()
    if not os.path.exists(args.output_path):
        print("making output path...")
        os.mkdir(args.output_path)
    if args.use_gpu:
        print("using gpu 0...")
        device = "cuda:0"
    else:
        device = "cpu"
    
    interval = 0.01
    print("step interval:",interval)


    with open(os.path.join(args.ckpt_path, "current_reinforce_configs.yaml"), "r") as f:
        configs = yaml.safe_load(f)

    reinforce_configs = configs["reinforce_configs"]
    with open(reinforce_configs["victim_configs"], "r", encoding="utf-8") as victim_config_f:
        victim_configs = yaml.safe_load(victim_config_f)
    
    data_configs = victim_configs["data"]
    # load data from victim configs
    _, dataset, _, _, _ = load_data(
        data_cfg=victim_configs["data"], datasets=["dev"])
    data_iter = iter(Iterator(
            repeat=False, sort=False, dataset=dataset,
            batch_size=30, batch_size_fn=None, 
            train=True, sort_within_batch=True,
            sort_key=lambda x: len(x.src), shuffle=False
        ))
    print("load data... finished")

    # build and reload victim model and reinforce policy model
    victim_model_path = reinforce_configs["victim_model"]
    src_vocab_file=os.path.join(victim_configs["training"]["model_dir"], "src_vocab.txt")
    trg_vocab_file=os.path.join(victim_configs["training"]["model_dir"], "trg_vocab.txt")
    src_vocab = Vocabulary(file=src_vocab_file)
    trg_vocab = Vocabulary(file=trg_vocab_file)
    BOS_id = src_vocab.stoi[BOS_TOKEN]
    PAD_id = src_vocab.stoi[PAD_TOKEN]
    local_src_embed, local_victim_translator = build_embedding_and_NMT(
        victim_configs=victim_configs["model"], victim_model_path=victim_model_path,
        src_vocab=src_vocab, trg_vocab=trg_vocab,
        device=device
    )
    local_victim_translator.loss_function = XentLoss(
            pad_index=local_victim_translator.pad_index, smoothing=0.1,
            reduction="none")  # smooth whatever u like
    local_src_embed.train()
    local_victim_translator.train()

    with torch.no_grad():
        weight = local_src_embed.lut.weight.detach().clone()
        if local_src_embed.scale:
            weight*=math.sqrt(weight.shape[1])
    emb_mean = weight.mean()
    emb_std = weight.std()

    with open("loss_variant_log", "w") as bleu_log:
        # initiation
        batch  = next(data_iter)
        batch = Batch(batch, pad_index=PAD_id, use_cuda=True)
        batch.to(device)

        reverse_index = batch.sort_by_src_length()
        # sort_reverse_index = expand_reverse_index(reverse_index, n_best=1)
        
        initial_src_emb = local_src_embed(batch.src)
        src_mask = batch.src_mask  # [batch, 1, len] 
        src_length = batch.src_length
        perturbed_src_emb = initial_src_emb.detach().clone()
        refs = []
        for _, ref in enumerate(batch.trg):
            ref = " ".join( [trg_vocab.itos[id] for id in ref.tolist() if id not in [1,2,3]])
            refs.append([ref])
        
        origin_loss, _, _, _ = local_victim_translator(
            return_type="loss",
            src_emb=perturbed_src_emb, 
            trg=batch.trg, trg_input=batch.trg_input, 
            src_mask=src_mask, src_length=src_length,
            trg_mask=batch.trg_mask)
        l_0 = origin_loss.view(batch.trg.shape[0], -1, origin_loss.shape[-1]).sum(dim=-1).detach()
            

        stds = torch.linspace(0.000, 0.3, max_perturbation_rounds, dtype=torch.float64).cpu().numpy().tolist()
        for i in range(max_perturbation_rounds):
            # calculate loss 

            normed_emb = (perturbed_src_emb-emb_mean) /emb_std
            perturbed_emb = normed_emb*math.sqrt(1-stds[i]**2) + stds[i]*torch.randn_like(normed_emb)
            perturbed_src_emb = perturbed_emb*emb_std+emb_mean

            # calculate novel loss
            perturbed_loss, _, _, _ = local_victim_translator(
                return_type="loss",
                src_emb=perturbed_src_emb, 
                trg=batch.trg, trg_input=batch.trg_input, 
                src_mask=src_mask, src_length=src_length,
                trg_mask=batch.trg_mask)
            l_1 = perturbed_loss.view(batch.trg.shape[0], -1, perturbed_loss.shape[-1]).sum(dim=-1).detach()
            loss_variant = abs(l_1-l_0).mean()

            print(i, ":",loss_variant.item())
            bleu_log.write(str(loss_variant.item())+"\n")
            l_0=l_1


    return


if __name__ == "__main__":
    # bleu_variant(300)
    # loss_variant(300)
    # for i in range(1,2):
        test_reinforce(20)