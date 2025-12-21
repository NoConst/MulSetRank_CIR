# fashioniq
CUDA_VISIBLE_DEVICES=6 nohup python src/blip_fine_tune_2.py \
   --dataset fashioniq \
   --blip-model-name 'blip2_cir_multivector' \
   --num-epochs 30 \
   --learning-rate 1e-5 \
   --batch-size 256 \
   --transform targetpad \
   --target-ratio 1.25  \
   --validation-frequency 1 \
   --experiment-name blip2-colbert-fiq-camv-cluster10-val-exp5 > blip2_colbert_fiq_camv_cluster10_val_exp5.log &