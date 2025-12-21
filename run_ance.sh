#!/bin/bash

# BLIP2 Fine-tuning with ANCE (Hard Negative Mining) Training Script
# Inspired by: Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval

# Set CUDA device
export CUDA_VISIBLE_DEVICES=1

# Common parameters
NUM_WORKERS=4
BATCH_SIZE=128  # Smaller batch size due to additional memory for hard negatives
LEARNING_RATE=1e-5
NUM_EPOCHS=30

# ANCE specific parameters
ANCE_NUM_NEGATIVES=16       # Number of hard negatives per query
ANCE_TOPK_CANDIDATES=100    # Top-k candidates to sample hard negatives from
ANCE_REFRESH_INTERVAL=1     # Refresh ANN index every N epochs
ANCE_WEIGHT=1.0             # Weight for hard negative samples
ANCE_WARMUP_EPOCHS=0        # Warmup epochs before enabling ANCE (0 = start immediately)

# Choose dataset: CIRR or FashionIQ
# DATASET="CIRR"
DATASET="FashionIQ"

# Model configuration
BLIP_MODEL_NAME="blip2_cir_align_prompt"
BACKBONE="pretrain"  # pretrain for vit-g, pretrain_vitL for vit-l

# Experiment name (for organizing output directories)
EXPERIMENT_NAME="ance_exp_v1"

cd /root/siton-data-92a7d2fc7b594215b07e48fd8818598b/MulSetRank_CIR/src

echo "=========================================="
echo "Starting BLIP2 + ANCE Training"
echo "Dataset: ${DATASET}"
echo "ANCE Negatives: ${ANCE_NUM_NEGATIVES}"
echo "ANCE Top-K Candidates: ${ANCE_TOPK_CANDIDATES}"
echo "ANCE Refresh Interval: ${ANCE_REFRESH_INTERVAL} epochs"
echo "ANCE Warmup Epochs: ${ANCE_WARMUP_EPOCHS}"
echo "=========================================="

python blip_fine_tune_ance.py \
    --dataset ${DATASET} \
    --num-workers ${NUM_WORKERS} \
    --num-epochs ${NUM_EPOCHS} \
    --blip-model-name ${BLIP_MODEL_NAME} \
    --backbone ${BACKBONE} \
    --learning-rate ${LEARNING_RATE} \
    --batch-size ${BATCH_SIZE} \
    --validation-frequency 1 \
    --target-ratio 1.25 \
    --transform targetpad \
    --save-memory \
    --experiment-name ${EXPERIMENT_NAME} \
    --ance-num-negatives ${ANCE_NUM_NEGATIVES} \
    --ance-topk-candidates ${ANCE_TOPK_CANDIDATES} \
    --ance-refresh-interval ${ANCE_REFRESH_INTERVAL} \
    --ance-weight ${ANCE_WEIGHT} \
    --ance-warmup-epochs ${ANCE_WARMUP_EPOCHS}

echo "=========================================="
echo "Training completed!"
echo "=========================================="

