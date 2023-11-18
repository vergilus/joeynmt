N=$1

save_path='/home/nfs01/zouw/policy/attack_bart_zh2en_log'

python3 -m adversarials.test_attack \
	--ckpt_path "${save_path}"\
	--output_path "${save_path}/in_perturbed.$1" \
	--use_gpu \
	--source_path "/home/nfs01/zouw/temp/zh-en_data/newstest20$1.sub.zh"

mv ${save_path}/in_perturbed.$1.origin  ${save_path}/in_origin.$1

victim_path='/'
python3 -m plm_mt.main translate \
	/home/nfs01/zouw/models/cwmt_zhen_TF_bart/config.yaml \
	--ckpt /home/nfs01/zouw/models/cwmt_zhen_TF_bart/best.ckpt \
	--output_path "${save_path}/perturbed_trans.$1" \
	< ${save_path}/in_perturbed.$1

perl  /home/nfs01/zouw/tokenizer/detokenizer.perl -l en -thread 5 < ${save_path}/perturbed_trans.$1 \
	> ${save_path}/perturbed_trans.out
cat ${save_path}/perturbed_trans.out | sacrebleu --tok intl /home/nfs01/zouw/data/cwmt17_zh-en_processed/newstest20$1.en
