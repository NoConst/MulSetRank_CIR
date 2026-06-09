#!/bin/bash

nohup deepspeed --include localhost:0,1,2,3,4 src/deepspeed_clip_ance_train.py \
    --dataset fashionIQ \
    --clip-model-name "ViT-H/14" \
    --batch-size 8 \
    --num-epochs 8 \
    --learning-rate 5e-5 \
    --lora-learning-rate 5e-5 \
    --fusion-learning-rate 5e-5 \
    --ance-num-negatives 3 \
    --listwise-weight 0.5 \
    --use-lora \
    --lora-r 16 \
    --lora-alpha 32 \
    --init-temperature 0.05 \
    --fusion-type sum \
    --fashioniq-val-split-mode original-split \
    --save-training \
    --save-best \
    --partial-intent-queries-path outputs/fiq_partial_intent_queries/partial_intent_queries.json \
    --deepspeed-config ds_config_zero2.json > deepspeed_clip_ance_fiq_H_14_partial_intent_all_listwise_loss_sum.log &

nohup deepspeed --include localhost:1,2,3,4 --master_port 29641 src/deepspeed_clip_ance_train.py \
    --dataset fashionIQ \
    --clip-model-name "ViT-H/14" \
    --batch-size 8 \
    --num-epochs 8 \
    --learning-rate 5e-5 \
    --lora-learning-rate 5e-5 \
    --fusion-learning-rate 1e-5 \
    --ance-num-negatives 3 \
    --ance-weight 1.0 \
    --ref-ance-weight 1.0 \
    --partial-intent-num-negatives 3 \
    --partial-intent-weight 0.75 \
    --listwise-weight 0.5 \
    --use-lora \
    --lora-r 16 \
    --lora-alpha 32 \
    --init-temperature 0.05 \
    --fusion-type cross_attention \
    --fusion-num-heads 8 \
    --fashioniq-val-split-mode original-split \
    --save-training \
    --save-best \
    --partial-intent-queries-path outputs/fiq_partial_intent_queries/partial_intent_queries.json \
    --single-intent-queries-path outputs/fiq_single_intent_queries/single_intent_queries.json \
    --intent-consistency-weight 0.5 \
    --intent-orthogonality-weight 0.3 \
    --intent-global-consistency-weight 0.5 \
    --intent-global-consistency-temperature 0.2 \
    --intent-consistency-epsilon 0.05 \
    --experiment-name contrastive_listwise_crossattn_ic_orth_all_loss_gpu1234 \
    --deepspeed-config ds_config_zero2.json \
    > logs/contrastive_listwise_crossattn_ic_orth_all_loss_gpu1234.log 2>&1 &

python src/cirr_test_submission.py \
  --model-path models/clip_ance_cirr_ViT-H-14_lora_2026-03-21_22:10:55/best_model \
  --submission-name clip_ance_cirr_ViT-H-14_lora_best


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
