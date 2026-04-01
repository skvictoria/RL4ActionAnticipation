#!/bin/bash

# Action Anticipation Inference Script

export OMP_NUM_THREADS=1
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false

# 학습된 체크포인트 경로 설정
VLM_CHECKPOINT="vlm_checkpoints/vlm_epoch_89.pt"  # 학습된 VLM 체크포인트
FUTR_CHECKPOINT="/home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/save_dir/utkinects/long/model/transformer/1/i3d_transcript/runs0/_20_30_50_erank_40p_64_latent_20251226/futr_joint_epoch_89.ckpt"  # 학습된 FUTR 체크포인트

# 데이터셋 경로
UTKINECT_ROOT="/home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/datasets/utkinect"

# 출력 디렉토리
OUTPUT_DIR="./inference_results_$(date +%Y%m%d_%H%M%S)"

echo "========================================="
echo "Action Anticipation Inference"
echo "========================================="
echo "VLM Checkpoint: $VLM_CHECKPOINT"
echo "FUTR Checkpoint: $FUTR_CHECKPOINT"
echo "Dataset: $UTKINECT_ROOT"
echo "Output: $OUTPUT_DIR"
echo "========================================="

CUDA_VISIBLE_DEVICES=0 python inference_anticipation.py \
    --model-path liuhaotian/llava-v1.5-7b \
    --vlm-checkpoint "$VLM_CHECKPOINT" \
    --futr-checkpoint "$FUTR_CHECKPOINT" \
    --env-name utkinect/test \
    --utkinect-root "$UTKINECT_ROOT" \
    --utkinect-split test \
    --utkinect-history 6 \
    --utkinect-frame-skip 1 \
    --num-inference-steps 100 \
    --temperature 0.2 \
    --max-new-tokens 256 \
    --output-dir "$OUTPUT_DIR" \
    --seed 42

echo ""
echo "========================================="
echo "Inference completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "========================================="
