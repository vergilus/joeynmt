export CUDA_VISIBLE_DEVICES="0,1,2,3"
echo "using gpu $CUDA_VISIBLE_DEVICES"

python3 -m adversarials.main_attack \
		--config_path "./adversarials/adversarial_configs/attack_wmt_bart_deen.yaml" \
		--save_to "/home/nfs01/zouw/policy/attack_bart_de2en_log" \
	    --n 2 \
		--reload
