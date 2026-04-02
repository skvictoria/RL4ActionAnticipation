#!/bin/bash

# FUTR-only Inference Script
# diagnose_segfault.py를 베이스로 만든 안전한 inference
# VLM 없이 FUTR만 사용하여 segmentation fault 회피

echo "========================================="
echo "FUTR-only Inference"
echo "========================================="
echo "Based on diagnose_segfault.py (which works)"
echo "No VLM, No Tokenizer, No HuggingFace Hub"
echo "========================================="

# Paths
FUTR_CHECKPOINT="/home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/save_dir/utkinects/long/model/transformer/1/i3d_transcript/runs0/_20_30_50_erank_40p_64_latent_20251226/futr_joint_epoch_99.ckpt"
DATASET_ROOT="/home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/datasets/utkinect"
OUTPUT_DIR="./inference_results_futr_only"

# Inference settings
NUM_STEPS=100

echo ""
echo "Configuration:"
echo "FUTR: $FUTR_CHECKPOINT"
echo "Dataset: $DATASET_ROOT"
echo "Output: $OUTPUT_DIR"
echo "Steps: $NUM_STEPS"
echo "PYTHONNOUSERSITE: ${PYTHONNOUSERSITE:-Not set}"
echo "========================================="
echo ""

# Run inference
python3 inference_futr_only.py \
    --futr-checkpoint "$FUTR_CHECKPOINT" \
    --dataset-root "$DATASET_ROOT" \
    --env-name utkinect/test \
    --split test \
    --history-window 6 \
    --frame-skip 1 \
    --num-steps $NUM_STEPS \
    --output-dir "$OUTPUT_DIR" \
    --seed 1

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "✓ Inference completed successfully!"
    echo "========================================="
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "View results:"
    echo "  cat $OUTPUT_DIR/inference_summary.txt"
    echo "========================================="
else
    echo ""
    echo "========================================="
    echo "✗ Inference failed!"
    echo "========================================="
    echo "Check the error messages above"
    echo "========================================="
    exit 1
fi
