N=$1

save_path='/home/nfs01/zouw/policy/attack_en2zh_log'

python3 -m adversarials.test_attack \
	--ckpt_path "${save_path}"\
	--output_path "${save_path}/in_perturbed.$1" \
	--use_gpu \
	--source_path "/home/nfs01/zouw/data/cwmt17_zh-en_processed/subword_processed/newstest20$1.sub.en"
	
mv ${save_path}/in_perturbed.$1.origin  ${save_path}/in_origin.$1

victim_path='/'
python3 -m joeynmt.main translate \
	/home/nfs01/zouw/models/cwmt_enzh_TF_best/config.yaml \
	--ckpt /home/nfs01/zouw/models/cwmt_enzh_TF_best/best.ckpt \
	  < ${save_path}/in_perturbed.$1 > ${save_path}/perturbed_trans.$1

python char.py ${save_path}/perturbed_trans.$1
cat ${save_path}/perturbed_trans.$1.char | sacrebleu --tok intl /home/nfs01/zouw/data/cwmt17_zh-en_processed/newstest20$1.tok.zh.char
