# CUDA_VISIBLE_DEVICES="0" bash reinforces/reinforce_scripts/test.sh 17
N=$1

save_path='./reinforces/reinforce_cwmt_log'

python3 -m reinforces.test_reinforce \
    --ckpt_path "${save_path}" \
	--output_path "${save_path}" \
	--source_path "/home/nfs01/zouw/data/cwmt17_zh-en_processed/subword_processed/newstest20$1.sub.zh"\
	--use_gpu \
	# --source_path "reinforces/in_perturbed$1"
	

perl  /home/nfs01/zouw/tokenizer/detokenizer.perl -l en -thread 5 < ${save_path}/out_origin \
			 > ${save_path}/out_origin.detok
perl  /home/nfs01/zouw/tokenizer/detokenizer.perl -l en -thread 5 < ${save_path}/out_reinforced5 \
			 > ${save_path}/out_reinforced5.detok

cat ${save_path}/out_origin.detok|sacrebleu --tok intl  /home/nfs01/zouw/data/cwmt17_zh-en_processed/newstest20$1.en
cat ${save_path}/out_reinforced5.detok|sacrebleu --tok intl  /home/nfs01/zouw/data/cwmt17_zh-en_processed/newstest20$1.en