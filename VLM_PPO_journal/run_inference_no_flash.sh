#!/bin/bash

# Safe Inference without Flash Attention
# Flash Attention 없이 안전하게 실행

echo "========================================="
echo "Safe Inference (No Flash Attention)"
echo "========================================="
echo ""

# Environment variables
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_LAUNCH_BLOCKING=1

# Disable Flash Attention
export DISABLE_FLASH_ATTN=1

# Remove PYTHONNOUSERSITE
unset PYTHONNOUSERSITE

# Paths
FUTR_CHECKPOINT="/home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/save_dir/utkinects/long/model/transformer/1/i3d_transcript/runs0/_20_30_50_erank_40p_64_latent_20251226/futr_joint_epoch_66.ckpt"
UTKINECT_ROOT="/home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/datasets/utkinect"
MODEL_PATH="liuhaotian/llava-v1.5-7b"
OUTPUT_DIR="./inference_results_no_flash"
NUM_STEPS=100

echo "Configuration:"
echo "  FUTR: $FUTR_CHECKPOINT"
echo "  Dataset: $UTKINECT_ROOT"
echo "  Output: $OUTPUT_DIR"
echo "  Steps: $NUM_STEPS"
echo "  Flash Attention: DISABLED"
echo "========================================="
echo ""

python3 inference_robust.py \
    --futr-checkpoint "$FUTR_CHECKPOINT" \
    --utkinect-root "$UTKINECT_ROOT" \
    --model-path "$MODEL_PATH" \
    --utkinect-split test \
    --utkinect-history 6 \
    --utkinect-frame-skip 1 \
    --num-inference-steps $NUM_STEPS \
    --output-dir "$OUTPUT_DIR" \
    --seed 1

echo ""
echo "========================================="
echo "Inference completed!"
echo "Results: $OUTPUT_DIR"
echo "========================================="
