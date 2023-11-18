N=$1

python -m plm_mt.main translate  \
    ../models/cwmt_zhen_TF_bart/config.yaml  \
    --ckpt ../models/cwmt_zhen_TF_bart/best.ckpt  \
    --output_path "trans$1.en" \
    < ../temp/zh-en_data/newstest20$1.sub.zh

perl /home/nfs01/zouw/tokenizer/detokenizer.perl  -l en -thread 5 < trans$1.en > trans$1.en.out
# mv trans$1.en.out trans$1.en
cat trans$1.en.out | sacrebleu --tok intl /home/nfs01/zouw/data/cwmt17_zh-en_processed/newstest20$1.en
