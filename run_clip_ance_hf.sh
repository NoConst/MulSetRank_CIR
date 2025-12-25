#!/bin/bash

# CLIP ANCE Training Script using Hugging Face Transformers
# This script trains CLIP models with ANCE (hard negative mining) for CIR tasks

# Set CUDA device
export CUDA_VISIBLE_DEVICES=3

# ===== Configuration =====
DATASET="fashionIQ"  # or "CIRR"
# DATASET="CIRR"

# Hugging Face CLIP Model Options:
# - "ViT-B/32" (maps to openai/clip-vit-base-patch32)
# - "ViT-B/16" (maps to openai/clip-vit-base-patch16)  
# - "ViT-L/14" (maps to openai/clip-vit-large-patch14)
# - Or use direct HF model path: "openai/clip-vit-base-patch32"

CLIP_MODEL="ViT-B/32"  # Default: ViT-B/32
EXPERIMENT_NAME="clip_ance_hf_v1_high_low_binary_repeat"

# Training hyperparameters
NUM_EPOCHS=30
BATCH_SIZE=32
LEARNING_RATE=2e-6
NUM_WORKERS=4
VALIDATION_FREQ=1

# ANCE specific parameters
ANCE_NUM_NEGATIVES=16        # Number of hard negatives per query
ANCE_TOPK_CANDIDATES=100    # Top-k candidates from ANN search
ANCE_REFRESH_INTERVAL=1      # Refresh FAISS index every N epochs
ANCE_WEIGHT=1.0              # Weight for hard negative loss
ANCE_WARMUP_EPOCHS=0         # Train N epochs without ANCE first

# Transform settings
TRANSFORM="targetpad"
TARGET_RATIO=1.25

# ===== Run Training =====
cd src

echo "=========================================="
echo "CLIP ANCE Training (Hugging Face) for $DATASET"
echo "Model: $CLIP_MODEL"
echo "Experiment: $EXPERIMENT_NAME"
echo "=========================================="
echo ""
echo "Training Configuration:"
echo "  - Epochs: $NUM_EPOCHS"
echo "  - Batch Size: $BATCH_SIZE"
echo "  - Learning Rate: $LEARNING_RATE"
echo ""
echo "ANCE Configuration:"
echo "  - Hard Negatives: $ANCE_NUM_NEGATIVES"
echo "  - Top-K Candidates: $ANCE_TOPK_CANDIDATES"
echo "  - Refresh Interval: $ANCE_REFRESH_INTERVAL epochs"
echo "  - ANCE Weight: $ANCE_WEIGHT"
echo "  - Warmup Epochs: $ANCE_WARMUP_EPOCHS"
echo "=========================================="
echo ""

python clip_fine_tune_ance_hf.py \
    --dataset $DATASET \
    --clip-model-name $CLIP_MODEL \
    --num-epochs $NUM_EPOCHS \
    --batch-size $BATCH_SIZE \
    --learning-rate $LEARNING_RATE \
    --num-workers $NUM_WORKERS \
    --validation-frequency $VALIDATION_FREQ \
    --transform $TRANSFORM \
    --target-ratio $TARGET_RATIO \
    --ance-num-negatives $ANCE_NUM_NEGATIVES \
    --ance-topk-candidates $ANCE_TOPK_CANDIDATES \
    --ance-refresh-interval $ANCE_REFRESH_INTERVAL \
    --ance-weight $ANCE_WEIGHT \
    --ance-warmup-epochs $ANCE_WARMUP_EPOCHS \
    --experiment-name $EXPERIMENT_NAME \
    --save-training \
    --save-best

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="

