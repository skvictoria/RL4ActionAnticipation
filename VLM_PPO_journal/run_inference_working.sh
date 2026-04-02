#!/bin/bash

# Working Inference Script
# diagnose_segfault.py와 동일한 환경 설정 사용

echo "========================================="
echo "Working Inference (From diagnose_segfault.py)"
echo "========================================="
echo "Using the exact same configuration that works"
echo "in diagnose_segfault.py"
echo "========================================="
echo ""

# diagnose_segfault.py와 동일한 환경 변수
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONNOUSERSITE=1

# Paths (diagnose_segfault.py 기본값과 동일)
FUTR_CHECKPOINT="/home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/save_dir/utkinects/long/model/transformer/1/i3d_transcript/runs0/_20_30_50_erank_40p_64_latent_20251226/futr_joint_epoch_99.ckpt"
DATASET_ROOT="/home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/datasets/utkinect"
OUTPUT_DIR="./inference_results_working"
NUM_STEPS=100

echo "Configuration:"
echo "  FUTR: $FUTR_CHECKPOINT"
echo "  Dataset: $DATASET_ROOT"
echo "  Output: $OUTPUT_DIR"
echo "  Steps: $NUM_STEPS"
echo "  PYTHONNOUSERSITE: $PYTHONNOUSERSITE"
echo "========================================="
echo ""

python3 inference_from_working.py \
    --futr-checkpoint "$FUTR_CHECKPOINT" \
    --dataset-root "$DATASET_ROOT" \
    --env-name utkinect/test \
    --split test \
    --history-window 6 \
    --frame-skip 1 \
    --num-steps $NUM_STEPS \
    --output-dir "$OUTPUT_DIR" \
    --seed 1

echo ""
echo "========================================="
echo "Inference completed!"
echo "Results: $OUTPUT_DIR"
echo "========================================="
