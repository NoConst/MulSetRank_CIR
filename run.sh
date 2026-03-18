CUDA_VISIBLE_DEVICES=0,1,2,3 nohup deepspeed --num_gpus=4 src/deepspeed_clip_ance_train.py \
    --dataset fashionIQ \
    --clip-model-name "ViT-H/14" \
    --batch-size 8 \
    --num-epochs 8 \
    --learning-rate 5e-5 \
    --lora-learning-rate 5e-5 \
    --fusion-learning-rate 2e-5 \
    --ance-num-negatives 3 \
    --listwise-weight 0.2 \
    --use-lora \
    --lora-r 16 \
    --lora-alpha 32 \
    --init-temperature 0.05 \
    --fusion-type adaptive_residual \
    --save-training \
    --save-best \
    --partial-intent-queries-path outputs/fiq_partial_intent_queries/partial_intent_queries.json \
    --deepspeed-config ds_config_zero2.json > deepspeed_clip_ance_fiq_H_14_partial_intent_all_listwise_loss.log &

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup deepspeed --num_gpus=4 src/deepspeed_clip_ance_train.py \
    --dataset cirr \
    --clip-model-name "ViT-H/14" \
    --batch-size 8 \
    --num-epochs 7 \
    --learning-rate 5e-5 \
    --lora-learning-rate 5e-5 \
    --fusion-learning-rate 1e-5 \
    --ance-num-negatives 3 \
    --listwise-weight 0.2 \
    --use-lora \
    --lora-r 16 \
    --lora-alpha 32 \
    --init-temperature 0.07 \
    --save-training \
    --save-best \
    --partial-intent-queries-path outputs/cirr_partial_intent_queries/partial_intent_queries.json \
    --deepspeed-config ds_config_zero2.json > deepspeed_clip_ance_cirr_H_14_partial_intent_batch4_neg5.log &

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
