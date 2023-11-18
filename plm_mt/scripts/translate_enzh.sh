N=$1

save_path='/home/nfs01/zouw/models/cwmt_enzh_TF_bart'
python -m plm_mt.main translate  \
    ${save_path}/config.yaml  \
    --ckpt ${save_path}/best.ckpt  \
    --output_path ${save_path}/trans$1.zh \
    < ../temp/zh-en_data/newstest20$1.sub.en

python char.py ${save_path}/trans$1.zh
cat ${save_path}/trans$1.zh.char | sacrebleu --tok intl /home/nfs01/zouw/data/cwmt17_zh-en_processed/newstest20$1.tok.zh.char
