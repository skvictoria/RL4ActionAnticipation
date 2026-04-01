#!/bin/bash

# Robust Inference Script for Action Anticipation
# Segmentation fault 방지를 위한 강력한 inference 실행 스크립트

# Environment variables to prevent segmentation faults
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_LAUNCH_BLOCKING=1

# FUTR checkpoint path
FUTR_CHECKPOINT="/home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/save_dir/utkinects/long/model/transformer/1/i3d_transcript/runs0/_20_30_50_erank_40p_64_latent_20251226/futr_joint_epoch_66.ckpt"

# Dataset path
UTKINECT_ROOT="/home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/datasets/utkinect"

# Model path (for tokenizer only)
MODEL_PATH="liuhaotian/llava-v1.5-7b"

# Output directory
OUTPUT_DIR="./inference_results_robust"

# Number of inference steps
NUM_STEPS=100

echo "========================================="
echo "Robust Action Anticipation Inference"
echo "========================================="
echo "FUTR Checkpoint: $FUTR_CHECKPOINT"
echo "Dataset: $UTKINECT_ROOT"
echo "Output: $OUTPUT_DIR"
echo "Steps: $NUM_STEPS"
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
echo "Results saved to: $OUTPUT_DIR"
echo "========================================="
