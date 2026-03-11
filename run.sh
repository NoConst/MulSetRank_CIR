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


CUDA_VISIBLE_DEVICES=0,1,2,3 nohup deepspeed --num_gpus=4 src/deepspeed_clip_ance_train.py \
    --dataset fashionIQ \
    --clip-model-name "ViT-H/14" \
    --batch-size 8 \
    --num-epochs 20 \
    --learning-rate 5e-5 \
    --lora-learning-rate 5e-5 \
    --fusion-learning-rate 3e-5 \
    --ance-num-negatives 3 \
    --use-lora \
    --ance-warmup-epochs 0 \
    --lora-r 16 \
    --lora-alpha 32 \
    --init-temperature 0.03 \
    --save-training \
    --save-best \
    --partial-intent-queries-path outputs/fiq_partial_intent_queries/partial_intent_queries.json \
    --deepspeed-config ds_config_zero2.json > deepspeed_clip_ance_fiq_H_14_partial_intent_lora5e-5_fusion1e-4.log &



python src/generate_partial_intent_queries.py \
    --dress_types dress shirt toptee \
    --max_workers 8 \
    --output_dir outputs/fiq_partial_intent_queries

nohup python src/generate_single_intent_queries.py \
    --dataset fashioniq \
    --max_workers 8 \
    --output_dir outputs/fiq_single_intent_queries > fiq_single_intent_queries.log 2>&1 &

nohup python src/generate_single_intent_queries.py \
    --dataset cirr \
    --max_workers 8 \
    --output_dir outputs/cirr_single_intent_queries > cirr_single_intent_queries.log 2>&1 &
