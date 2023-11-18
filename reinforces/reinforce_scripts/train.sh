CUDA_VISIBLE_DEVICES="0,1,2,3,4,5"  python3 -m reinforces.main_reinforce \
    --save_to "./reinforces/reinforce_cwmt_log" \
    --n=3 --use_gpu --reload
