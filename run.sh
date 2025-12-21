# fashioniq
CUDA_VISIBLE_DEVICES=0 nohup python src/blip_fine_tune_2.py \
   --dataset fashioniq \
   --blip-model-name 'blip2_cir_align_prompt' \
   --num-epochs 30 \
   --learning-rate 1e-5 \
   --batch-size 256 \
   --transform targetpad \
   --target-ratio 1.25  \
   --validation-frequency 1 \
   --experiment-name blip2-test > blip2_test.log &