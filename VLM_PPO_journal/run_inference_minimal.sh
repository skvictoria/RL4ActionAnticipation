#!/bin/bash
export OMP_NUM_THREADS=1
export PYTHONNOUSERSITE=1
# Minimal inference script - uses training code's model loading

# VLM_CHECKPOINT="vlm_checkpoints/epoch_4"
# FUTR_CHECKPOINT="/home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/save_dir/utkinects/utkinect_futr_joint_epoch_4.ckpt"
VLM_CHECKPOINT="/home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/save_dir/utkinects/long/model/transformer/1/i3d_transcript/runs0/_20_30_50_erank_40p_64_latent_20251226/vlm_checkpoints/epoch_99"  # 학습된 VLM 체크포인트
FUTR_CHECKPOINT="/home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/save_dir/utkinects/long/model/transformer/1/i3d_transcript/runs0/_20_30_50_erank_40p_64_latent_20251226/futr_joint_epoch_99.ckpt"  # 학습된 FUTR 체크포인트

UTKINECT_ROOT="/home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/datasets/utkinect"
OUTPUT_DIR="./inference_results_minimal"

echo "Running minimal inference (without VLM generation)..."
echo "This uses coarse labels directly instead of generating fine-grained text."
echo ""

if [ -z "$VLM_CHECKPOINT" ]; then
    echo "⚠ No VLM checkpoint specified, using base model only"
    CUDA_VISIBLE_DEVICES=0 python inference_minimal.py \
        --model-path liuhaotian/llava-v1.5-7b \
        --futr-checkpoint "$FUTR_CHECKPOINT" \
        --utkinect-root "$UTKINECT_ROOT" \
        --utkinect-split test \
        --utkinect-history 6 \
        --utkinect-frame-skip 1 \
        --num-steps 10 \
        --output-dir "$OUTPUT_DIR" \
        --seed 42
else
    echo "Using VLM checkpoint: $VLM_CHECKPOINT"
    CUDA_VISIBLE_DEVICES=0 python inference_minimal.py \
        --model-path /home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/save_dir/utkinects/long/model/transformer/1/i3d_transcript/runs0/_20_30_50_erank_40p_64_latent_20251226/vlm_checkpoints/epoch_99 \
        --vlm-checkpoint "$VLM_CHECKPOINT" \
        --futr-checkpoint "$FUTR_CHECKPOINT" \
        --utkinect-root "$UTKINECT_ROOT" \
        --utkinect-split test \
        --utkinect-history 6 \
        --utkinect-frame-skip 1 \
        --num-steps 10 \
        --output-dir "$OUTPUT_DIR" \
        --seed 42
        #--model-path liuhaotian/llava-v1.5-7b \
fi

echo ""
echo "Inference completed! Results saved to $OUTPUT_DIR"
