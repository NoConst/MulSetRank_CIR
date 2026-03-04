CUDA_VISIBLE_DEVICES=0 nohup python src/clip_fine_tune_fusion.py \
    --dataset fashionIQ \
    --clip-model-name ViT-B/32 \
    --pretrained-clip-path /root/siton-data-92a7d2fc7b594215b07e48fd8818598b/MulSetRank_CIR/models/clip_ance_fiq_ViT-B-32_clip_ance_fiq_2025-12-31_12:59:14/best_model \
    --fusion-type cross_attention \
    --num-cross-attn-layers 4 \
    --num-heads 8 \
    --learning-rate 1e-4 \
    --batch-size 32 \
    --num-epochs 50 \
    --ance-num-negatives 16 \
    --ance-topk-candidates 100 \
    --save-training \
    --save-best > clip_fusion_fiq_ViT-B-32_hard_16.log &

CUDA_VISIBLE_DEVICES=1 nohup python src/clip_fine_tune_fusion.py \
    --dataset fashionIQ \
    --clip-model-name ViT-B/32 \
    --fusion-type cross_attention \
    --num-cross-attn-layers 4 \
    --num-heads 8 \
    --learning-rate 1e-4 \
    --batch-size 128 \
    --num-epochs 50 \
    --ance-num-negatives 4 \
    --ance-topk-candidates 100 \
    --save-training \
    --save-best > clip_fusion_fiq_ViT-B-32_original.log &



CUDA_VISIBLE_DEVICES=3 nohup python src/clip_fine_tune_ance.py \
    --dataset cirr \
    --clip-model-name ViT-H/14 \
    --batch-size 4 \
    --num-epochs 10 \
    --learning-rate 1e-4 \
    --ance-num-negatives 4 \
    --use-lora \
    --lora-r 16 \
    --lora-alpha 32 \
    --ance-warmup-epochs 0 \
    --save-training --save-best > clip_ance_cirr_H_14_neg_8_with_hard_in_batch_1e-4_lora_16_32.log &

nohup ./run_clip_ance_deepspeed.sh 4 > train_deepspeed.log 2>&1 &


CUDA_VISIBLE_DEVICES=0,1,2,3 nohup torchrun --nproc_per_node=4 src/fsdp_clip_ance_train.py \
  --dataset fashionIQ \
  --clip-model-name ViT-H/14 \
  --batch-size 8 \
  --num-epochs 20 \
  --learning-rate 5e-5 \
  --ance-num-negatives 4 \
  --use-lora \
  --lora-r 32 \
  --lora-alpha 64 \
  --ance-warmup-epochs 0 \
  --use-fsdp > fsdp_clip_ance_fiq.log &


1、去掉loss hard negative
2、去掉loss hard in batch
3、



# CUDA_VISIBLE_DEVICES=0,1,2,3 nohup deepspeed --num_gpus=4 src/deepspeed_clip_ance_train.py \
#     --dataset cirr \
#     --clip-model-name "ViT-H/14" \
#     --batch-size 8 \
#     --num-epochs 8 \
#     --learning-rate 5e-5 \
#     --ance-num-negatives 4 \
#     --use-lora \
#     --lora-r 16 \
#     --lora-alpha 32 \
#     --save-training --save-best \
#     --deepspeed-config ds_config_zero2.json > deepspeed_clip_ance_cirr_H_14_neg_4_with_lora_16_32.log &


# deepspeed --num_gpus 4 src/deepspeed_blip2_ance_train.py \
#   --dataset fashioniq \
#   --blip-model-name blip2_cir_align_prompt --backbone pretrain \
#   --batch-size 32 --grad-accum-steps 4 \
#   --learning-rate 2e-6 \
#   --ance-num-negatives 10 --ance-topk-candidates 100 --ance-weight 1.0 --ance-warmup-epochs 0 \
#   --deepspeed-config ds_config_zero2.json