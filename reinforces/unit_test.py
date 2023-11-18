# coding: utf-8
from typing import OrderedDict
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import torch
from joeynmt.vocabulary import Vocabulary
import os
from joeynmt.constants import PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN

def plot_emb(save_dir:str):
    """
    :param model_dir: the dir for model parameters 
    """
    # load model parameters for embeddings
    model_dir = os.path.join(save_dir, "best.ckpt")
    state_dict = torch.load(model_dir, map_location="cpu")
    # load vocabulary to show tokens 
    vocab_dir = os.path.join(save_dir, "src_vocab.txt")
    src_vocab = Vocabulary(file=vocab_dir)

    # extract embedding matrix
    token_emb = state_dict["model_state"]["src_embed.lut.weight"]
    print("<unk>", token_emb[0].mean())
    print("pad", token_emb[1].mean())  # the pad
    print("<s>", token_emb[2].mean())
    print("</s>", token_emb[3].mean())
    token_emb = np.array(token_emb)
    tsne = TSNE(n_components=2, verbose=1)
    tsne_proj = tsne.fit_transform(token_emb)
    tsne_proj = torch.tensor(tsne_proj)
    # tsne_proj = torch.randn(token_emb.shape[0],2)  # dummy points to show  

    # output to 2d-pixels
    norm = plt.Normalize(1,4)
    fig, ax = plt.subplots()
    sc = plt.scatter(tsne_proj[:, 0],tsne_proj[:, 1], c="m", alpha=0.1)
    plt.rcParams['font.sans-serif']='SimHei'

    annot = ax.annotate(
        "", xy=(0,0), xytext=(20,20),textcoords="offset points",
        bbox=dict(boxstyle="round", fc="w"),
        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)
    
    def update_annot(ind:dict):
        """
        ind: point in path collection, dict:["ind", array]
        """
        pos = sc.get_offsets()[ind["ind"][0]]  # get point in path collections
        print("pos", pos)
        annot.xy = pos  # set annotation position
        # make annotation context [id, token] 
        text = "{}".format(" ".join([src_vocab.itos[n] for n in ind["ind"]]))
        print(text)
        annot.set_text(text)
        annot.set_fontfamily('sans-serif')
        annot.get_bbox_patch().set_facecolor("b")
        annot.get_bbox_patch().set_alpha(0.4)
        
    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:  # listen to the events of ax
            cont, ind = sc.contains(event)  # ind contains the dots in trajectory
            # scatter plot contains target mouse event
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)  # listen to hover function
    plt.title('嵌入', color='blue')
    plt.show()
    return

def nearest_emb(save_dir: str, batch_size:int=50):
    """
    check the nearest embedding
    """
    # load model parameters for embeddings
    model_dir = os.path.join(save_dir, "best.ckpt")
    state_dict = torch.load(model_dir, map_location="cpu")
    # load vocabulary to show tokens 
    vocab_dir = os.path.join(save_dir, "src_vocab.txt")
    src_vocab = Vocabulary(file=vocab_dir)

    # extract embedding matrix
    emb = state_dict["model_state"]["src_embed.lut.weight"]
    len_mat = torch.sum(emb**2, dim=1)**0.5  # length of the embeddings

    with open("near_cand.txt", "w") as similar_vocab:
        top_reserve=5
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
            # get topk indices by cosine similarity
            topk_index = similarity.topk(top_reserve, dim=1)[1]
            sliceemb = slice_emb.unsqueeze(dim=1).repeat(1, top_reserve, 1)  # [batch_size, 1*k, dim]
            E_dist = ((emb[topk_index]-sliceemb)**2).sum(dim=-1)**0.5
            # print(i,"euc_dis:", E_dist.mean(), "cos sim:", similarity.topk(top_reserve, dim=1)[0].mean())
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
            # print("batch std:", E_dist.std())
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
                    near_vocab+=[near_cand]
                else:
                    for k in range(1, topk_val.shape[1]):
                        near_cand_id = topk_indices[j][k]
                        near_cand = src_vocab.itos[near_cand_id]
                        if (near_cand not in [PAD_TOKEN, EOS_TOKEN, BOS_TOKEN]):
                            # and topk_val[j][k]<avg_dist:
                            bingo += 1
                            similar_vocab.write(near_cand+"\t")
                            near_vocab+=[near_cand]
                    if bingo==0 :
                        last_cand_ids = [src_vocab.stoi[UNK_TOKEN]]
                        for final_reserve_id in last_cand_ids:
                            last_cand = UNK_TOKEN
                            similar_vocab.write(last_cand+"\t")
                            near_vocab+=[last_cand]
                similar_vocab.write("\n")
                word2near_vocab[src_word] = near_vocab

    return

def plot_perturbation(save_dir:str):
    """
    plot perturbation of embeddings 
    """
    # load model parameters for embeddings
    model_dir = os.path.join(save_dir, "best.ckpt")
    state_dict = torch.load(model_dir, map_location="cpu")
    # load vocabulary to show tokens 
    vocab_dir = os.path.join(save_dir, "src_vocab.txt")
    src_vocab = Vocabulary(file=vocab_dir)

    # extract embedding matrix
    emb = state_dict["model_state"]["src_embed.lut.weight"]

    token_emb = emb[torch.tensor([188, 8019])]
    print(token_emb.shape)
    # add a noise and check for nearest token sets
    token_emb += 0.001* torch.randn_like(token_emb)
    E_dist = ((token_emb.unsqueeze(dim=1)-emb.unsqueeze(dim=0))**2).sum(dim=-1)
    v, id = E_dist.topk(dim=-1, k=3,  largest=False)
    print(id)
    return

if __name__=="__main__":
    save_dir = "/data0/zouw/models/cwmt_zhen_TF_best/"
    # plot_emb(save_dir)
    # nearest_emb(save_dir)
    plot_perturbation(save_dir)