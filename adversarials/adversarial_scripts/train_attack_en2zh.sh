export CUDA_VISIBLE_DEVICES="0,2,5"
echo "using gpu $CUDA_VISIBLE_DEVICES"

python3 -m adversarials.main_attack \
		--config_path "./adversarials/adversarial_configs/attack_cwmt_enzh.yaml" \
		--save_to "/home/nfs01/zouw/policy/attack_en2zh_log" \
	    --n 2 \
		--reload
