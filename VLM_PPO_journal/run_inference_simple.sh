#!/bin/bash
export OMP_NUM_THREADS=1
export PYTHONNOUSERSITE=1
# Simple inference script (segfault-safe version)

# Checkpoint paths
# VLM_CHECKPOINT="vlm_checkpoints/epoch_4"  # or vlm_checkpoints/vlm_epoch_4.pt
# FUTR_CHECKPOINT="/home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/save_dir/utkinects/utkinect_futr_joint_epoch_4.ckpt"

VLM_CHECKPOINT="/home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/save_dir/utkinects/long/model/transformer/1/i3d_transcript/runs0/_20_30_50_erank_40p_64_latent_20251226/vlm_checkpoints/vlm_epoch_99.pt"  # 학습된 VLM 체크포인트
FUTR_CHECKPOINT="/home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/save_dir/utkinects/long/model/transformer/1/i3d_transcript/runs0/_20_30_50_erank_40p_64_latent_20251226/futr_joint_epoch_99.ckpt"  # 학습된 FUTR 체크포인트

# Dataset path
UTKINECT_ROOT="/home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/datasets/utkinect"

# Output directory
OUTPUT_DIR="./inference_results_simple"

# Run inference with fewer steps for testing
CUDA_VISIBLE_DEVICES=0 python inference_simple.py \
    --model-path liuhaotian/llava-v1.5-7b \
    --vlm-checkpoint "$VLM_CHECKPOINT" \
    --futr-checkpoint "$FUTR_CHECKPOINT" \
    --env-name utkinect/test \
    --utkinect-root "$UTKINECT_ROOT" \
    --utkinect-split test \
    --utkinect-history 6 \
    --utkinect-frame-skip 1 \
    --num-inference-steps 10 \
    --temperature 0.2 \
    --max-new-tokens 256 \
    --output-dir "$OUTPUT_DIR" \
    --seed 42

echo "Inference completed! Results saved to $OUTPUT_DIR"
