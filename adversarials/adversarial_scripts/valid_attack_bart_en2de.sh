N=$1

save_path='./adversarials/attack_bart_en2de_log'

python3 -m adversarials.test_attack \
	--ckpt_path "${save_path}"\
	--output_path "${save_path}/in_perturbed.$1" \
	--use_gpu \
	--source_path "/home/nfs01/zouw/temp/en-de_data/newstest20$1.sub.en"

# subword-nmt apply-bpe -c \
# 	/home/data_ti6_d/zouw/data/wmt14_en-de_processed/code.zh-en.txt \
# 	--vocabulary /home/data_ti6_d/zouw/data/wmt14_en-de_processed/vocab.bpe32k.zh \
# 	--vocabulary-threshold 50 < ${save_path}/perturbed_input > ${save_path}/perturbed.sub
mv ${save_path}/in_perturbed.$1.origin  ${save_path}/in_origin.$1

victim_path='/'
python3 -m plm_mt.main translate \
	/home/nfs01/zouw/models/wmt_ende_TF_bart_tied/config.yaml \
	--ckpt /home/nfs01/zouw/models/wmt_ende_TF_bart_tied/best.ckpt \
	--output_path "${save_path}/perturbed_trans.$1" \
	< ${save_path}/in_perturbed.$1

perl  /home/nfs01/zouw/tokenizer/detokenizer.perl -l en -thread 5 < ${save_path}/perturbed_trans.$1 \
	> ${save_path}/perturbed_trans.out
cat ${save_path}/perturbed_trans.out | sacrebleu --tok intl /home/nfs01/zouw/data/wmt14_en-de_processed/newstest20$1.tok.de
