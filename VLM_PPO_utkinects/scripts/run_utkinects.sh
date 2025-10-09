#!/bin/bash
export OMP_NUM_THREADS=1
export PYTHONNOUSERSITE=1

python main.py \
    --env-name "Utkinects-v0" \
    --dataset-path "/home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/datasets/utkinect/RGB" \
    --num-processes 1 \
    --num-steps 128 \
    --ppo-epoch 4 \
    --mini-batch-size 1 \
    --entropy-coef 0.01 \
    --value-loss-coef 0.5 \
    --gamma 0.99 \
    --gae-lambda 0.95 \
    --num-env-steps 10000000 \
    --log-interval 1 \
    --save-interval 100 \
    --save-dir "../logs/utkinects_logs" \
    --init-lr 2.5e-4 \
    --llava-model-path "liuhaotian/llava-v1.5-7b" \
    --llava-model-base "lmsys/vicuna-7b-v1.5" \