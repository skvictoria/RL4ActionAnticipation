#!/bin/bash

# Full VLM + FUTR Inference Script
# 학습된 VLM checkpoint를 사용한 전체 파이프라인

# Environment variables to prevent segmentation faults
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_LAUNCH_BLOCKING=1
export PYTHONNOUSERSITE=1

# CRITICAL: Unset PYTHONNOUSERSITE (known to cause segfaults)
unset PYTHONNOUSERSITE

# CRITICAL: Disable Flash Attention (known to cause segfaults)
export DISABLE_FLASH_ATTN=1

echo "========================================="
echo "Full VLM + FUTR Inference"
echo "========================================="
echo "This script uses your trained VLM checkpoint"
echo "for fine-grained descriptions + FUTR for"
echo "action anticipation."
echo "========================================="
echo ""

# ============================================================================
# 경로 설정 (실제 경로로 수정하세요)
# ============================================================================

# 학습된 VLM checkpoint (LoRA weights)
VLM_CHECKPOINT="/home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/save_dir/utkinects/long/model/transformer/1/i3d_transcript/runs0/_20_30_50_erank_40p_64_latent_20251226/vlm_checkpoints/epoch_99"

# 학습된 FUTR checkpoint
FUTR_CHECKPOINT="/home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/save_dir/utkinects/long/model/transformer/1/i3d_transcript/runs0/_20_30_50_erank_40p_64_latent_20251226/futr_joint_epoch_99.ckpt"

# Dataset path
UTKINECT_ROOT="/home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/datasets/utkinect"

# Base model path (HuggingFace Hub 또는 로컬 경로)
MODEL_PATH="liuhaotian/llava-v1.5-7b"

# HuggingFace cache directory (선택사항)
# CACHE_DIR="~/.cache/huggingface"

# Output directory
OUTPUT_DIR="./inference_results_with_vlm"

# Number of inference steps
NUM_STEPS=100

# ============================================================================
# 체크포인트 존재 확인
# ============================================================================

echo "========================================="
echo "Full VLM + FUTR Inference"
echo "========================================="
echo ""

echo "Checking paths..."
echo ""

# VLM checkpoint 확인
if [ -d "$VLM_CHECKPOINT" ]; then
    echo "✓ VLM checkpoint found: $VLM_CHECKPOINT"
    
    # 필수 파일 확인
    if [ -f "$VLM_CHECKPOINT/config.json" ]; then
        echo "  ✓ config.json found"
    else
        echo "  ✗ config.json NOT found"
        echo "  ⚠ VLM checkpoint may be incomplete"
    fi
    
    if [ -f "$VLM_CHECKPOINT/adapter_config.json" ]; then
        echo "  ✓ adapter_config.json found"
    else
        echo "  ✗ adapter_config.json NOT found"
    fi
    
    if [ -f "$VLM_CHECKPOINT/tokenizer_config.json" ]; then
        echo "  ✓ tokenizer_config.json found"
    else
        echo "  ✗ tokenizer_config.json NOT found"
        echo "  ⚠ Will use base model tokenizer"
    fi
else
    echo "✗ VLM checkpoint NOT found: $VLM_CHECKPOINT"
    echo ""
    echo "Please update VLM_CHECKPOINT path in this script"
    exit 1
fi

echo ""

# FUTR checkpoint 확인
if [ -f "$FUTR_CHECKPOINT" ]; then
    echo "✓ FUTR checkpoint found: $FUTR_CHECKPOINT"
else
    echo "✗ FUTR checkpoint NOT found: $FUTR_CHECKPOINT"
    echo ""
    echo "Please update FUTR_CHECKPOINT path in this script"
    exit 1
fi

echo ""

# Dataset 확인
if [ -d "$UTKINECT_ROOT" ]; then
    echo "✓ Dataset found: $UTKINECT_ROOT"
else
    echo "✗ Dataset NOT found: $UTKINECT_ROOT"
    echo ""
    echo "Please update UTKINECT_ROOT path in this script"
    exit 1
fi

echo ""
echo "========================================="
echo "Starting inference..."
echo "========================================="
echo "VLM Checkpoint: $VLM_CHECKPOINT"
echo "FUTR Checkpoint: $FUTR_CHECKPOINT"
echo "Dataset: $UTKINECT_ROOT"
echo "Output: $OUTPUT_DIR"
echo "Steps: $NUM_STEPS"
echo "========================================="
echo ""

# ============================================================================
# Inference 실행
# ============================================================================

python3 inference_anticipation.py \
    --model-path "$MODEL_PATH" \
    --vlm-checkpoint "$VLM_CHECKPOINT" \
    --futr-checkpoint "$FUTR_CHECKPOINT" \
    --utkinect-root "$UTKINECT_ROOT" \
    --utkinect-split test \
    --utkinect-history 6 \
    --utkinect-frame-skip 1 \
    --num-inference-steps $NUM_STEPS \
    --temperature 0.2 \
    --max-new-tokens 256 \
    --conv-mode vicuna_v1 \
    --output-dir "$OUTPUT_DIR" \
    --seed 1

# ============================================================================
# 결과 확인
# ============================================================================

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "✓ Inference completed successfully!"
    echo "========================================="
    echo ""
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    
    if [ -f "$OUTPUT_DIR/inference_summary.txt" ]; then
        echo "Summary:"
        echo "========================================="
        cat "$OUTPUT_DIR/inference_summary.txt"
        echo "========================================="
    fi
else
    echo ""
    echo "========================================="
    echo "✗ Inference failed!"
    echo "========================================="
    echo ""
    echo "Troubleshooting:"
    echo "1. Check if segmentation fault occurred"
    echo "2. Try FUTR-only inference: sh run_inference_robust.sh"
    echo "3. Run diagnostics: python3 diagnose_segfault.py"
    echo "4. Check guides: cat SEGFAULT_FIX_GUIDE.md"
    exit 1
fi
