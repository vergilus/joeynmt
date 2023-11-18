# coding: utf-8

from collections import defaultdict, Counter
from typing import List
import os
import sys
import torch

from torch.nn.parameter import Parameter
from transformers import BartTokenizer, MBart50Tokenizer, BartModel, MBartModel
from torchtext.legacy.datasets import TranslationDataset
from torchtext.legacy.data import Field
from joeynmt.vocabulary import Vocabulary
from joeynmt.data import MonoDataset,build_vocab
from joeynmt.model import Model
from joeynmt.initialization import initialize_model
from joeynmt.helpers import ConfigurationError
from joeynmt.embeddings import Embeddings
from joeynmt.encoders import RecurrentEncoder, TransformerEncoder
from joeynmt.decoders import RecurrentDecoder, TransformerDecoder

BOS_TOKEN='<s>'
EOS_TOKEN='</s>'
SEP_TOKEN='</s>'
CLS_TOKEN='<s>'
UNK_TOKEN='<unk>'
PAD_TOKEN='<pad>'
MASK_TOKEN='<mask>'

class BartVocab(Vocabulary):
    """ convert a bart vocabulary item for joeynmt"""
    def __init__(self, tokens: List[str] = None, file: str = None) -> None:
        super().__init__(tokens, file)
        # self.specials = [UNK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]  # joeynmt settting
        self.specials = [BOS_TOKEN,PAD_TOKEN,EOS_TOKEN,UNK_TOKEN]  # default bart setting
        self.stoi = defaultdict(self.default_unk_id)  # for any unregistered key yield UNK_id of 4
        self.itos = []
        if tokens is not None:
            # tokens does not include special tokens
            self._from_list(tokens)
        elif file is not None:
            self._from_file(file)
    
    def default_unk_id(self):
        return 4

    def is_unk(self, token: str) -> bool:
        return self.stoi[token] == self.default_unk_id()

# the translation must be implemented by 
def load_vocab_tokenizer_from_plm(temp_dir, plm_path):
    """
    load tokenizers from plm, extract vocabulary for joeynmt
    """
    if "mbart" in plm_path:
        print("build mbart plm")
        tokenizer = MBart50Tokenizer.from_pretrained(plm_path)
    elif "bart" in plm_path:
        print("build bart as plm")
        tokenizer = BartTokenizer.from_pretrained(plm_path)

    # if not os.path.exists(os.path.join(temp_dir, "plm_vocab.txt")):
    #     print("recommend python scripts/build_vocab.py train.src_lang train.trg_lang instead")
    #     # generate vocab in temp if needed
    #     vocab_stoi = tokenizer.get_vocab()
    #     print("plm vocab size:", len(vocab_stoi))
    #     list_of_vocab = list(vocab_stoi.keys())[4:]
    #     # output to a vocab file in temp dir
    #     with open(os.path.join(temp_dir, "plm_vocab.txt"), "w") as vocab_file:
    #         for w in list_of_vocab:
    #             vocab_file.write(w+"\n")
    # build vocab from that temp vocab file
    # plm_vocab = BartVocab(file=os.path.join(temp_dir, "plm_vocab.txt"))
    # print("built vocab size:", len(plm_vocab.stoi))
    return tokenizer


def build_or_load_vocab_by_dataset(max_size:int, min_freq: int,vocab_file:str, dataset=None) -> Vocabulary:
    """
    filter smaller vocabulary by a given max size
    """

    if os.path.exists(vocab_file):  # buffered vocabulary
        vocab = BartVocab(file=vocab_file)
    else:
        assert dataset is not None, "filtering must be given a dataset object"
        def filter_min(counter: Counter, min_freq: int):
            """ Filter counter by min frequency """
            filtered_counter = Counter({t: c for t, c in counter.items()
                                        if c >= min_freq})
            return filtered_counter
        def sort_and_cut(counter: Counter, limit: int):
            """ Cut counter to most frequent,
            sorted numerically and alphabetically"""
            # sort by frequency, then alphabetically
            tokens_and_frequencies = sorted(counter.items(),
                                            key=lambda tup: tup[0])
            tokens_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)
            vocab_tokens = [i[0] for i in tokens_and_frequencies[:limit]]
            return vocab_tokens
        # create and save to vocab_file
        tokens = []
        for i in dataset.examples:
            tokens.extend(i.src)
            tokens.extend(i.trg)
        counter=Counter(tokens)
        if min_freq>-1:
            counter=filter_min(counter, min_freq)
        vocab_tokens = sort_and_cut(counter, max_size)
        assert len(vocab_tokens) <= max_size
        
        vocab = BartVocab(tokens=vocab_tokens)
        with open(vocab_file, "w", encoding='utf-8') as f:
            for w in vocab_tokens:
                f.write(w + "\n")
        assert len(vocab) <= max_size + len(vocab.specials)
    return vocab

def load_and_process_data(data_cfg: dict, datasets:list=None):
    """
    will process the data by a plm tokenizer that cater to joeynmt and store in temp_dir
    the raw data is saved and tokenized with space. 

    seg_ids = toker.encode(line)
    seg_tokens = toker.convert_ids_to_tokens(seg_ids)
    line = toker.convert_tokens_to_string(seg_tokens) ## with BOS and EOS appended.
    """
    if datasets is None:
        datasets=["train", "dev", "test"]

    if not os.path.exists(data_cfg["temp_path"]):
        os.mkdir(data_cfg["temp_path"])
    # plm vocab is used to process the data
    print(data_cfg["plm_path"])
    tokenizer = load_vocab_tokenizer_from_plm(data_cfg["temp_path"], data_cfg["plm_path"])
    # process the tokenized raw data
    train_path = data_cfg.get("train", None)
    dev_path = data_cfg.get("dev", None)
    test_path = data_cfg.get("test", None)

    src_lang = data_cfg["src"]
    trg_lang = data_cfg["trg"]
    max_sent_length = data_cfg["max_sent_length"]
    level = data_cfg["level"]
    lowercase = data_cfg["lowercase"]
    tok_fun = lambda s: list(s) if level == "char" else s.split()
    # the src and trg processed by plm tokenizer.
    src_field = Field(init_token=None, eos_token=EOS_TOKEN, pad_token=PAD_TOKEN,
                      tokenize=tok_fun, batch_first=True, lower=lowercase,
                      unk_token=UNK_TOKEN, include_lengths=True)

    trg_field = Field(init_token=BOS_TOKEN, eos_token=EOS_TOKEN,
                      pad_token=PAD_TOKEN, unk_token=UNK_TOKEN,
                      tokenize=tok_fun, batch_first=True, lower=lowercase,
                      include_lengths=True)        
    # build dataset with bilingual field
    train_data = None
    dev_data = None
    test_data = None
    if "train" in datasets and train_path is not None:
        print("loading training data",end="...")
        if not os.path.exists(os.path.join(data_cfg["temp_path"], "train.temp."+src_lang)):
            with open(train_path+"."+src_lang, "r") as raw_in, \
                open(os.path.join(data_cfg["temp_path"], "train.temp."+src_lang), "w") as processed_in:
                print("processing src", end="...")
                for line in raw_in:
                    seg_ids = tokenizer.encode(line)
                    seg_tokens = tokenizer.convert_ids_to_tokens(seg_ids)
                    processed_line = " ".join(seg_tokens[1:-1])
                    processed_in.write(processed_line+'\n')
        if not os.path.exists(os.path.join(data_cfg["temp_path"], "train.temp."+trg_lang)):
            with open(train_path+"."+trg_lang, "r") as raw_in, \
                open(os.path.join(data_cfg["temp_path"], "train.temp."+trg_lang), "w") as processed_in:
                print("processing trg", end="..." )
                for line in raw_in:
                    seg_ids = tokenizer.encode(line)
                    seg_tokens = tokenizer.convert_ids_to_tokens(seg_ids)
                    processed_line = " ".join(seg_tokens[1:-1])
                    processed_in.write(processed_line+'\n')
        train_data = TranslationDataset(
            path=data_cfg["temp_path"]+"train.temp",
            exts=("."+src_lang, "."+trg_lang),
            fields=(src_field, trg_field),
            filter_pred=lambda x: len(vars(x)['src'])
                        <= max_sent_length
                        and len(vars(x)['trg'])
                        <= max_sent_length
        )
        print("finished")

    # filter the vocabulary by given training data, export vocabulary for MT to temp dir
    # if not os.path.exists(os.path.join(data_cfg["temp_path"], "vocab.txt")):
    print("filter vocabulary", end="...")
    src_vocab = build_or_load_vocab_by_dataset(
        max_size=data_cfg["src_voc_limit"], min_freq=1, dataset=train_data,
        vocab_file=os.path.join(data_cfg["temp_path"], "vocab.txt")
    )
    # src_vocab = BartVocab(file=os.path.join(data_cfg["temp_path"], "vocab.txt"))
    trg_vocab=src_vocab
    print("finish loading vocabulary")

    if "dev" in datasets and dev_path is not None:
        print("loading dev data...")
        if not os.path.exists(os.path.join(data_cfg["temp_path"], "dev.temp."+src_lang)):
            with open(dev_path+"."+src_lang, "r") as raw_in, \
                open(os.path.join(data_cfg["temp_path"], "dev.temp."+src_lang), "w") as processed_in:
                for line in raw_in:
                    seg_ids = tokenizer.encode(line)
                    seg_tokens = tokenizer.convert_ids_to_tokens(seg_ids)
                    processed_line = " ".join(seg_tokens[1:-1])
                    processed_in.write(processed_line+'\n')
        if not os.path.exists(os.path.join(data_cfg["temp_path"], "dev.temp."+trg_lang)):
            with open(dev_path+"."+trg_lang, "r") as raw_in, \
                open(os.path.join(data_cfg["temp_path"], "dev.temp."+trg_lang), "w") as processed_in:
                for line in raw_in:
                    seg_ids = tokenizer.encode(line)
                    seg_tokens = tokenizer.convert_ids_to_tokens(seg_ids)
                    processed_line = " ".join(seg_tokens[1:-1])
                    processed_in.write(processed_line+'\n')
        dev_data = TranslationDataset(
            path=data_cfg["temp_path"]+"dev.temp",
            exts=("." + src_lang, "." + trg_lang),
            fields=(src_field, trg_field)
        )
    if "test" in datasets and test_path is not None:
        print("loading test data...")
        if not os.path.exists(os.path.join(data_cfg["temp_path"], "test.temp."+src_lang)):
            with open(test_path+"."+src_lang, "r") as raw_in, \
                open(os.path.join(data_cfg["temp_path"], "test.temp."+src_lang), "w") as processed_in:
                print("processing test src"+test_path+"."+src_lang)
                for line in raw_in:
                    seg_ids = tokenizer.encode(line)
                    seg_tokens = tokenizer.convert_ids_to_tokens(seg_ids)
                    processed_line = " ".join(seg_tokens[1:-1])
                    processed_in.write(processed_line+'\n')
        
        if os.path.isfile(test_path+"."+trg_lang):
            if not os.path.exists(os.path.join(data_cfg["temp_path"], "test.temp."+trg_lang)):
                with open(test_path+"."+trg_lang, "r") as raw_in, \
                    open(os.path.join(data_cfg["temp_path"], "test.temp."+trg_lang), "w") as processed_in:
                    print("processing test trg"+test_path+"."+trg_lang)
                    for line in raw_in:
                        seg_ids = tokenizer.encode(line)
                        seg_tokens = tokenizer.convert_ids_to_tokens(seg_ids)
                        processed_line = " ".join(seg_tokens[1:-1])
                        processed_in.write(processed_line+'\n')
            test_data = TranslationDataset(
                path=data_cfg["temp_path"]+"test.temp",
                exts=("." + src_lang, "." + trg_lang),
                fields=(src_field, trg_field)
            )
        else:
            test_data = MonoDataset(
                path=data_cfg["temp_path"]+"test.temp",
                ext="."+src_lang,
                field=src_field)

    src_field.vocab = src_vocab
    trg_field.vocab = trg_vocab
    print("Data loaded.")
    return train_data, dev_data, test_data, src_vocab, trg_vocab

def build_model_w_Pemb(cfg: dict = None,
                src_vocab: Vocabulary = None,
                trg_vocab: Vocabulary = None) ->Model:
    """
    Build and initialize the model according to the configuration.

    :param cfg: model configuration
    :param src_vocab: source vocabulary
    :param trg_vocab: target vocabulary
    :return: built and initialized model
    """
    print("Building an encoder-decoder model with PLM embedding", end="...")
    src_padding_idx = src_vocab.stoi[PAD_TOKEN]
    trg_padding_idx = trg_vocab.stoi[PAD_TOKEN]

    # load PLM and extract embedding
    if "mbart" in cfg["plm_path"]:
        plm = MBartModel.from_pretrained(cfg["plm_path"])
        toker = MBart50Tokenizer.from_pretrained(cfg["plm_path"])
    elif "bart" in cfg["plm_path"]:
        plm = BartModel.from_pretrained(cfg["plm_path"])
        toker = BartTokenizer.from_pretrained(cfg["plm_path"])
    
    plm_embedding_dim = plm.shared.embedding_dim  # will override the dim setting in transformer

    cfg["encoder"]["embeddings"]["embedding_dim"]= plm_embedding_dim
    cfg["decoder"]["embeddings"]["embedding_dim"]= plm_embedding_dim

    src_embed = Embeddings(
        **cfg["encoder"]["embeddings"], vocab_size=len(src_vocab),
        padding_idx=src_padding_idx,
        )

    # this ties source and target embeddings
    # for softmax layer tying, see further below
    if cfg.get("tied_embeddings", False):
        if src_vocab.itos == trg_vocab.itos:
            # share embeddings for src and trg
            trg_embed = src_embed
        else:
            raise ConfigurationError(
                "Embedding cannot be tied since vocabularies differ.")
    else: # additional target embedding
        trg_embed = Embeddings(
            **cfg["decoder"]["embeddings"], vocab_size=len(trg_vocab),
            padding_idx=trg_padding_idx,
            )

    # build encoder
    enc_dropout = cfg["encoder"].get("dropout", 0.)
    enc_emb_dropout = cfg["encoder"]["embeddings"].get("dropout", enc_dropout)
    if cfg["encoder"].get("type", "recurrent") == "transformer":
        cfg["encoder"]["hidden_size"] = plm_embedding_dim
        cfg["decoder"]["hidden_size"] = plm_embedding_dim
        assert cfg["encoder"]["embeddings"]["embedding_dim"] == \
               cfg["encoder"]["hidden_size"], \
               "for transformer, emb_size must be hidden_size"

        encoder = TransformerEncoder(**cfg["encoder"],
                                     emb_size=src_embed.embedding_dim,
                                     emb_dropout=enc_emb_dropout)
    else:
        encoder = RecurrentEncoder(**cfg["encoder"],
                                   emb_size=src_embed.embedding_dim,
                                   emb_dropout=enc_emb_dropout)

    # build decoder
    dec_dropout = cfg["decoder"].get("dropout", 0.)
    dec_emb_dropout = cfg["decoder"]["embeddings"].get("dropout", dec_dropout)
    if cfg["decoder"].get("type", "recurrent") == "transformer":
        decoder = TransformerDecoder(
            **cfg["decoder"], encoder=encoder, vocab_size=len(trg_vocab),
            emb_size=trg_embed.embedding_dim, emb_dropout=dec_emb_dropout)
    else:
        decoder = RecurrentDecoder(
            **cfg["decoder"], encoder=encoder, vocab_size=len(trg_vocab),
            emb_size=trg_embed.embedding_dim, emb_dropout=dec_emb_dropout)

    model = Model(encoder=encoder, decoder=decoder,
                  src_embed=src_embed, trg_embed=trg_embed,
                  src_vocab=src_vocab, trg_vocab=trg_vocab)
    # custom initialization of model parameters, freeze the plm embedding
    initialize_model(model, cfg, src_padding_idx, trg_padding_idx)
    # convert corresponding ids (list object)
    ids = toker.convert_tokens_to_ids(list(src_vocab.stoi.keys()))
    
    src_embed.lut.weight.data = plm.shared.weight[torch.tensor(ids)].data
    for _, p in src_embed.named_parameters():
        p.requires_grad = False
    print("PLM embedding loaded")

    # tie softmax layer with trg embeddings
    if cfg.get("tied_softmax", False):
        if trg_embed.lut.weight.shape == \
                model.decoder.output_layer.weight.shape:
            # (also) share trg embeddings and softmax layer:
            model.decoder.output_layer.weight = trg_embed.lut.weight
        else:
            raise ConfigurationError(
                "For tied_softmax, the decoder embedding_dim and decoder "
                "hidden_size must be the same."
                "The decoder must be a Transformer.")
    
    return model
