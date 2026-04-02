#!/bin/bash

# Complete Inference with VLM + FUTR
# diagnose_segfault.py 환경 + VLM 로딩

echo "========================================="
echo "Complete Inference (VLM + FUTR)"
echo "========================================="
echo "Using diagnose_segfault.py environment"
echo "+ VLM model loading"
echo "========================================="
echo ""

# diagnose_segfault.py와 동일한 환경 변수
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONNOUSERSITE=1

# Paths
VLM_CHECKPOINT="/home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/save_dir/utkinects/long/model/transformer/1/i3d_transcript/runs0/_20_30_50_erank_40p_64_latent_20251226/vlm_checkpoints/epoch_99"
FUTR_CHECKPOINT="/home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/save_dir/utkinects/long/model/transformer/1/i3d_transcript/runs0/_20_30_50_erank_40p_64_latent_20251226/futr_joint_epoch_99.ckpt"
DATASET_ROOT="/home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/datasets/utkinect"
OUTPUT_DIR="./inference_results_complete"
NUM_STEPS=100

echo "Configuration:"
echo "  VLM: $VLM_CHECKPOINT"
echo "  FUTR: $FUTR_CHECKPOINT"
echo "  Dataset: $DATASET_ROOT"
echo "  Output: $OUTPUT_DIR"
echo "  Steps: $NUM_STEPS"
echo "  PYTHONNOUSERSITE: $PYTHONNOUSERSITE"
echo "========================================="
echo ""

python3 inference_complete.py \
    --vlm-checkpoint "$VLM_CHECKPOINT" \
    --futr-checkpoint "$FUTR_CHECKPOINT" \
    --dataset-root "$DATASET_ROOT" \
    --env-name utkinect/test \
    --split test \
    --history-window 6 \
    --frame-skip 1 \
    --temperature 0.2 \
    --max-new-tokens 256 \
    --conv-mode vicuna_v1 \
    --num-steps $NUM_STEPS \
    --output-dir "$OUTPUT_DIR" \
    --seed 1

echo ""
echo "========================================="
echo "Inference completed!"
echo "Results: $OUTPUT_DIR"
echo "========================================="
