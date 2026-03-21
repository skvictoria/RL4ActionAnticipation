#!/bin/bash

# Minimal inference script - uses training code's model loading

VLM_CHECKPOINT="vlm_checkpoints/epoch_4"
FUTR_CHECKPOINT="/home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/save_dir/utkinects/utkinect_futr_joint_epoch_4.ckpt"
UTKINECT_ROOT="/home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/datasets/utkinect"
OUTPUT_DIR="./inference_results_minimal"

echo "Running minimal inference (without VLM generation)..."
echo "This uses coarse labels directly instead of generating fine-grained text."
echo ""

CUDA_VISIBLE_DEVICES=0 python inference_minimal.py \
    --model-path liuhaotian/llava-v1.5-7b \
    --vlm-checkpoint "$VLM_CHECKPOINT" \
    --futr-checkpoint "$FUTR_CHECKPOINT" \
    --utkinect-root "$UTKINECT_ROOT" \
    --utkinect-split test \
    --utkinect-history 6 \
    --utkinect-frame-skip 1 \
    --num-steps 10 \
    --output-dir "$OUTPUT_DIR" \
    --seed 42

echo ""
echo "Inference completed! Results saved to $OUTPUT_DIR"
