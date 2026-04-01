#!/bin/bash
export OMP_NUM_THREADS=1
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false

# 메모리 최적화 설정
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

echo "=========================================="
echo "Memory Optimized Training Configuration"
echo "=========================================="
echo "num-steps: 128 (reduced from 256)"
echo "mini-batch-size: 2 (reduced from 4)"
echo "grad-accum-steps: 32 (increased from 16)"
echo "CLIP: CPU-based (moved to GPU only when needed)"
echo "=========================================="

CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --num_processes 1 \
    --config_file scripts/config_no_deepspeed.yaml \
    --main_process_port 29501 \
    main.py \
    --env-name utkinect/train \
    --model-path liuhaotian/llava-v1.5-7b \
    --utkinect-root /home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/datasets/utkinect \
    --utkinect-split train \
    --utkinect-history 6 \
    --utkinect-frame-skip 1 \
    --num-env-steps 51200 \
    --num-steps 128 \
    --grad-accum-steps 32 \
    --max-new-tokens 256 \
    --thought-prob-coef 0.1 \
    --use-gae \
    --seed 1 \
    --temperature 0.2 \
    --ppo-epoch 4 \
    --mini-batch-size 2 \
    --use-lora \
    --train-vision none \
    --use-wandb \
    --wandb-project "ActionAnticipation_VLM" \
    --wandb-run "memory_optimized" \
    --save_interval 5
