# python3 -m joeynmt train configs/wmt_ende_best.yaml
CUDA_VISIBLE_DEVICES="7" nohup python3 -m joeynmt train configs/transformer_cwmt17_zhen.yaml >tf_zhen.log
