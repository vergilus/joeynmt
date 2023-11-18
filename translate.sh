# ordinary translation: test file is coded in the configs, and it requires pre-process
# python3 -m joeynmt test configs/transformer_cwmt17_zhen.yaml --ckpt models/cwmt_zhen_TF_best/best.ckpt
N=$1
# interactive translation:
python3 -m joeynmt.main translate configs/transformer_cwmt17_zhen.yaml --ckpt models/cwmt_zhen_TF_best/best.ckpt </home/data_ti6_d/username/data/cwmt17_zh-en_processed/subword_processed/newstest20$1.sub.zh > newstest20$1.trans.en

perl /home/data_ti6_d/username/tokenizer/detokenizer.perl -l en -thread 5 <newstest20$1.trans.en > newstest20$1.trans.dek.en
mv newstest20$1.trans.dek.en newstest20$1.trans.en
cat newstest20$1.trans.en | sacrebleu --tok intl /home/data_ti6_d/username/data/cwmt17_zh-en_processed/newstest20$1.en