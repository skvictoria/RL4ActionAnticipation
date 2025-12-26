# inference.sh
export OMP_NUM_THREADS=1
export PYTHONNOUSERSITE=1
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --num_processes 1 \
    --config_file scripts/config_zero2.yaml \
    --main_process_port 29502 \
    inference.py \
    --env-name utkinect/eval \
    --model-path liuhaotian/llava-v1.5-7b \
    --cache_dir /home/hice1/skim3513/RL4ActionAnticipation/hf_cache \
    --utkinect-root /home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/datasets/utkinect \
    --utkinect-split val \
    --utkinect-history 6 \
    --utkinect-frame-skip 1 \
    --max-new-tokens 256 \
    --use-lora \
    --train-vision all \
    --conv-mode vicuna_v1 # main.py에 설정된 conv_mode 확인 필요