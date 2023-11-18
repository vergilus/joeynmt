import argparse
import shutil
import logging
import os

from joeynmt.helpers import set_seed, make_logger, make_model_dir, load_config, log_cfg, log_data_info, load_checkpoint
from joeynmt.training import TrainManager
from joeynmt.batch import Batch
from joeynmt.model import build_model
from joeynmt.prediction import validate_on_data, parse_test_args
from plm_mt.utils import BartVocab
from plm_mt.utils import *
# from plm_mt.utils import load_and_process_data,build_or_load_vocab_by_dataset, build_model_w_Pemb

logger = logging.getLogger(__name__)

def train(cfg_file: str):
    """
    :param cfg_file: path to configs
    """
    cfg = load_config(cfg_file)
    # make logger
    model_dir = make_model_dir(
        cfg["training"]["model_dir"],
        overwrite=cfg["training"].get("overwrite", False)
    )
    _ = make_logger(model_dir, mode="train")  # version string returned

    # set the random seed
    set_seed(seed=cfg["training"].get("random_seed", 42))

    # load from plm and process the raw data to temp_dir (cater to joetnmt)
    train_data, dev_data, test_data, src_vocab, trg_vocab = load_and_process_data(
        data_cfg=cfg["data"])
    # build an encoder-decoder model with pretrained embedding.
    model = build_model_w_Pemb(cfg["model"], src_vocab=src_vocab, trg_vocab=trg_vocab)
    # for training management, e.g. early stopping and model selection
    trainer = TrainManager(model=model, config=cfg)

    # store copy of original training config in model dir
    shutil.copy2(cfg_file, model_dir + "/config.yaml")

    # log all entries of config
    log_cfg(cfg)

    log_data_info(train_data=train_data,
                  valid_data=dev_data,
                  test_data=test_data,
                  src_vocab=src_vocab,
                  trg_vocab=trg_vocab)

    logger.info(str(model))

    # store the vocabs
    src_vocab_file = f"{model_dir}/src_vocab.txt"
    src_vocab.to_file(src_vocab_file)
    trg_vocab_file = f"{model_dir}/trg_vocab.txt"
    trg_vocab.to_file(trg_vocab_file)

    # train the model
    trainer.train_and_validate(train_data=train_data, valid_data=dev_data)
    
    return

def translate(cfg_file: str, ckpt_path,
              output_path, n_best):
    """
    :param cfg_file: path to configs
    """
    cfg = load_config(cfg_file)
    toker = MBart50Tokenizer.from_pretrained(cfg["data"]["plm_path"])

    # load from plm and process the raw data to temp_dir (cater to joetnmt)
    data_cfg = cfg["data"]
    src_vocab_file = data_cfg.get("src_vocab", "./src_vocab.txt")
    trg_vocab_file = data_cfg.get("trg_vocab", "./trg_vocab.txt")
    src_vocab = BartVocab(file=src_vocab_file)
    trg_vocab = BartVocab(file=trg_vocab_file)

    level = data_cfg["level"]
    tok_fun = lambda s: list(s) if level == "char" else s.split()
    lowercase = data_cfg["lowercase"]
    src_field = Field(init_token=None, eos_token=EOS_TOKEN, pad_token=PAD_TOKEN,
                      tokenize=tok_fun, batch_first=True, lower=lowercase,
                      unk_token=UNK_TOKEN, include_lengths=True)
    src_field.vocab = src_vocab    
    
    # build an encoder-decoder model with pretrained embedding.
    batch_size, batch_type, use_cuda, device, n_gpu, level, eval_metric, \
        max_output_length, beam_size, beam_alpha, postprocess, \
        bpe_type, sacrebleu, decoding_description, tokenizer_info \
        = parse_test_args(cfg, mode="test")

    # reload embedding test
    # src_emb = Embeddings(
    #     **cfg["encoder"]["embeddings"], vocab_size=len(src_vocab),
    #     padding_idx=src_vocab.stoi[PAD_TOKEN])
    # src_emb.load_state_dict(model_checkpoint["model_state"])

    model_checkpoint = load_checkpoint(ckpt_path, use_cuda=use_cuda)
    model = build_model(cfg["model"], src_vocab=src_vocab, trg_vocab=trg_vocab)
    model.load_state_dict(model_checkpoint["model_state"], strict=True)
    # for training management reload models and validate models
    if use_cuda:
        model.to(device)

    test_data = MonoDataset(path=sys.stdin, ext="", field=src_field)

    score, loss, ppl, sources, sources_raw, references, hypotheses, \
        hypotheses_raw, attention_scores = validate_on_data(
            model=model, data=test_data, batch_size=1000,
            batch_class=Batch, batch_type=batch_type, level=level,
            max_output_length=max_output_length, eval_metric="",
            use_cuda=use_cuda, compute_loss=False, beam_size=beam_size,
            beam_alpha=beam_alpha, postprocess=True,
            bpe_type=bpe_type, sacrebleu=sacrebleu, n_gpu=n_gpu, n_best=n_best)
    # all_hypotheses = hypotheses
    all_hypotheses = []
    for hyp in hypotheses:
        hyp=hyp.split()
        all_hypotheses.append(toker.convert_tokens_to_string(hyp))
    
    if output_path is not None:
        # write to outputfile if given
        def write_to_file(output_path_set, hypotheses):
            with open(output_path_set, mode="w", encoding="utf-8") as out_file:
                for hyp in hypotheses:
                    out_file.write(hyp + "\n")
            logger.info("Translations saved to: %s.", output_path_set)

        if n_best > 1:
            for n in range(n_best):
                file_name, file_extension = os.path.splitext(output_path)
                write_to_file(
                    f"{file_name}-{n}" \
                    f"{file_extension if file_extension else ''}",
                    [all_hypotheses[i]
                        for i in range(n, len(all_hypotheses), n_best)]
                )
        else:
            write_to_file(f"{output_path}", all_hypotheses)
    else:
        # print to stdout
        for hyp in all_hypotheses:
            print(hyp)


def main():
    ap = argparse.ArgumentParser("Joey NMT")
    ap.add_argument("mode", choices=["train", "translate"], 
                    help="mode: train or translate")
    ap.add_argument("config_path", type=str,
                    help="path to YAML config file")
    ap.add_argument("--ckpt", type=str,
                    help="checkpoint for prediction")
    ap.add_argument("--output_path", type=str,
                    help="path for saving translation output")

    ap.add_argument("--save_attention", action="store_true",
                    help="save attention visualizations")

    ap.add_argument("-n", "--nbest", type=int, default=1,
                    help="Display n-best candidates when translating")
    args = ap.parse_args()

    if args.mode=="train":
        print("train:")
        train(cfg_file=args.config_path)
    elif args.mode=="translate":
        translate(cfg_file=args.config_path, ckpt_path=args.ckpt, 
                  output_path=args.output_path, n_best=args.nbest)


if __name__ == "__main__":
    main()
