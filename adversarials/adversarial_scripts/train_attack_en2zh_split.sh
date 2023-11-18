export CUDA_VISIBLE_DEVICES="0,1,2"
echo "using gpu $CUDA_VISIBLE_DEVICES"

python3 -m adversarials.main_attack \
		--config_path "./adversarials/adversarial_configs/attack_cwmt_enzh_split.yaml" \
		--save_to "/home/nfs01/zouw/policy/attack_en2zh_split_log" \
	    --n 2 \
		--reload
