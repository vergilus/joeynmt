export CUDA_VISIBLE_DEVICES="0,3,4"
echo "using gpu $CUDA_VISIBLE_DEVICES"

python3 -m adversarials.main_attack \
		--config_path "./adversarials/adversarial_configs/attack_wmt_ende_split.yaml" \
		--save_to "/home/nfs01/zouw/policy/attack_en2de_split_log" \
	    --n 2 \
		--reload
