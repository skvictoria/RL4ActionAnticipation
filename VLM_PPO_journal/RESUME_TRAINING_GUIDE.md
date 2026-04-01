# 학습 재개 가이드 (Resume Training Guide)

## 문제 상황

Epoch 96부터 학습을 재개하려고 했는데 바로 종료되는 문제

## 원인

학습 iteration 수(`num_updates`)가 현재 epoch보다 작거나 같아서 학습 루프가 실행되지 않음

### num_updates 계산 방식

```python
num_updates = num_env_steps // num_steps // num_processes
```

**예시**:
- `num_env_steps = 25000`
- `num_steps = 256`
- `num_processes = 1`

```python
num_updates = 25000 // 256 // 1 = 97
```

따라서 학습 루프는 `range(0, 97)` = 0~96 epoch까지만 실행됩니다.

만약 epoch 96부터 재개하면:
```python
for j in range(96, 97):  # 96 epoch만 1번 실행
    train(...)
```

## 해결 방법

### 방법 1: num-env-steps 증가 (권장)

원하는 총 epoch 수를 계산하여 `num-env-steps`를 설정:

```bash
# 공식: num_env_steps = 원하는_epoch * num_steps * num_processes

# 200 epoch까지 학습하려면:
num_env_steps = 200 * 256 * 1 = 51200

# 300 epoch까지 학습하려면:
num_env_steps = 300 * 256 * 1 = 76800

# 500 epoch까지 학습하려면:
num_env_steps = 500 * 256 * 1 = 128000
```

**run_no_deepspeed.sh 수정**:
```bash
--num-env-steps 51200  # 200 epoch까지
```

### 방법 2: num-steps 증가

각 iteration의 step 수를 늘림 (비권장 - 학습 동작이 변경됨):

```bash
--num-steps 512  # 기존 256에서 증가
```

이 경우:
```python
num_updates = 25000 // 512 // 1 = 48
```

⚠️ 주의: `num-steps`를 변경하면 학습 동작이 달라질 수 있습니다.

## 수정된 스크립트

### run_no_deepspeed.sh (200 epoch까지)

```bash
#!/bin/bash
export OMP_NUM_THREADS=1
export PYTHONNOUSERSITE=1

TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --num_processes 1 \
    --config_file scripts/config_no_deepspeed.yaml \
    --main_process_port 29501 \
    main.py \
    --env-name utkinect/train \
    --model-path liuhaotian/llava-v1.5-7b \
    --utkinect-root /path/to/utkinect \
    --utkinect-split train \
    --utkinect-history 6 \
    --utkinect-frame-skip 1 \
    --num-env-steps 51200 \
    --num-steps 256 \
    --grad-accum-steps 16 \
    --max-new-tokens 256 \
    --thought-prob-coef 0.1 \
    --use-gae \
    --seed 1 \
    --temperature 0.2 \
    --ppo-epoch 4 \
    --mini-batch-size 4 \
    --use-lora \
    --train-vision none \
    --use-wandb \
    --wandb-project "ActionAnticipation_VLM" \
    --wandb-run "experiment_1" \
    --save_interval 5
```

## 학습 재개 시 확인 사항

### 1. 체크포인트 확인

```bash
# FUTR 체크포인트
ls -lh /path/to/save_dir/utkinects/*epoch_96.ckpt

# VLM 체크포인트
ls -lh /path/to/save_dir/utkinects/vlm_checkpoints/epoch_96/
```

### 2. num_updates 계산

현재 설정으로 몇 epoch까지 학습할 수 있는지 확인:

```python
num_updates = num_env_steps // num_steps // num_processes
print(f"Total epochs: {num_updates}")
```

### 3. 학습 시작 시 로그 확인

수정된 `main.py`는 다음과 같은 경고를 출력합니다:

```
================================================================================
⚠ WARNING: start_epoch (96) >= num_updates (97)
Training would end immediately!

Current settings:
  num_env_steps: 25000
  num_steps: 256
  num_processes: 1
  Calculated num_updates: 97

To continue training, you need to:
  1. Increase --num-env-steps (currently 25000)
  2. Or increase --num-steps (currently 256)

Example: --num-env-steps 50000
================================================================================

Continue anyway? (y/n):
```

## 권장 설정

### 짧은 학습 (100 epoch)
```bash
--num-env-steps 25600  # 100 * 256 * 1
```

### 중간 학습 (200 epoch)
```bash
--num-env-steps 51200  # 200 * 256 * 1
```

### 긴 학습 (500 epoch)
```bash
--num-env-steps 128000  # 500 * 256 * 1
```

### 매우 긴 학습 (1000 epoch)
```bash
--num-env-steps 256000  # 1000 * 256 * 1
```

## 체크포인트 저장 간격

`--save_interval` 옵션으로 저장 빈도 조절:

```bash
--save_interval 5   # 5 epoch마다 저장
--save_interval 10  # 10 epoch마다 저장
--save_interval 20  # 20 epoch마다 저장
```

## 예시: Epoch 96부터 200까지 학습

```bash
# 1. num-env-steps 계산
num_env_steps = 200 * 256 * 1 = 51200

# 2. 스크립트 실행
sh run_no_deepspeed.sh

# 3. 로그 확인
# [Training] Will train from epoch 96 to 200
# [Training] Total iterations: 104
```

## 요약

- ✅ `num-env-steps`를 증가시켜 총 epoch 수 늘리기
- ✅ 공식: `num_env_steps = 원하는_epoch * num_steps * num_processes`
- ✅ 200 epoch까지: `--num-env-steps 51200`
- ✅ 수정된 `main.py`가 자동으로 경고 출력
- ⚠️ `num-steps` 변경은 학습 동작을 바꾸므로 비권장

이제 epoch 96부터 200까지 정상적으로 학습할 수 있습니다!
