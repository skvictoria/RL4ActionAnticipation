# CUDA Out of Memory 해결 가이드

## 문제 상황

```
CUDA out of memory. Tried to allocate 230.00 MiB. 
GPU 0 has a total capacity of 79.18 GiB of which 136.25 MiB is free.
Of the allocated memory 74.63 GiB is allocated by PyTorch
```

H200 (79GB)에서 메모리 부족 발생.

---

## 🔍 원인 분석

1. **Gradient Accumulation 미작동**: 74.63GB 사용 중
2. **메모리 누수**: 반복 학습 중 메모리 증가
3. **Large Batch**: rollout storage가 너무 큼
4. **CLIP 모델**: 추가 메모리 사용

---

## ✅ 해결 방법

### 1. Gradient Accumulation 수정 (가장 중요)

현재 `--grad-accum-steps 16`이지만 제대로 작동하지 않는 것 같습니다.

**run.sh 수정**:
```bash
--grad-accum-steps 32  # 16 → 32로 증가
--mini-batch-size 2    # 4 → 2로 감소
```

### 2. 메모리 정리 추가

`train_rl.py`에 메모리 정리 코드 추가:

```python
# train_rl.py의 train 함수 끝에 추가
def train(...):
    # ... 기존 코드 ...
    
    rollouts.after_update()
    
    # 메모리 정리
    if (j + 1) % 10 == 0:
        torch.cuda.empty_cache()
        import gc
        gc.collect()
```

### 3. Rollout Steps 감소

```bash
--num-steps 128  # 256 → 128로 감소
```

### 4. CLIP 모델 최적화

`main.py`에서 CLIP을 필요할 때만 GPU로 이동:

```python
# main.py
clip_model, _ = clip.load("ViT-B/32", device='cpu')  # CPU에 로드
clip_model = clip_model.float()
for param in clip_model.parameters():
    param.requires_grad = False
```

### 5. Mixed Precision 확인

`config_no_deepspeed.yaml`:
```yaml
mixed_precision: bf16  # 이미 설정되어 있는지 확인
```

---

## 🚀 권장 설정

### run_memory_optimized.sh

```bash
#!/bin/bash
export OMP_NUM_THREADS=1
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false

# 메모리 최적화 설정
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

CUDA_VISIBLE_DEVICES=0 accelerate launch \
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
    --num-env-steps 25000 \
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
    --save_interval 10
```

---

## 📊 메모리 사용량 예상

| 컴포넌트 | 메모리 | 최적화 후 |
|---------|--------|----------|
| LLaVA-7B | ~15GB | ~15GB |
| Value Model | ~15GB | ~15GB |
| FUTR | ~2GB | ~2GB |
| CLIP | ~2GB | ~0.5GB (CPU) |
| Rollout (256 steps) | ~20GB | ~10GB (128 steps) |
| Gradients | ~15GB | ~7GB (grad accum) |
| Activations | ~10GB | ~5GB (bf16) |
| **Total** | **~79GB** | **~55GB** |

---

## 🔧 코드 수정

### 1. train_rl.py - 메모리 정리 추가

```python
def train(args, actor_critic, prompt, tokenizer, rollouts, infos, envs, episode_rewards, 
          running_episode_rewards, running_episode_steps, episode_success_rate, 
          episode_action_tokens_log_prob, agent, lr_scheduler, start, j, num_updates, 
          clip_model, joint_model=None):
    
    # ... 기존 코드 ...
    
    rollouts.after_update()
    
    # [추가] 주기적 메모리 정리
    if (j + 1) % 10 == 0:
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        
        # 메모리 사용량 로깅
        if args.use_wandb:
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            wandb.log({
                "memory/allocated_gb": allocated,
                "memory/reserved_gb": reserved,
            }, step=(j + 1) * args.num_steps * args.num_processes)
```

### 2. main.py - CLIP CPU 로드

```python
import clip
clip_model, _ = clip.load("ViT-B/32", device='cpu')  # CPU에 로드
clip_model = clip_model.float()
for param in clip_model.parameters():
    param.requires_grad = False
```

### 3. train_rl.py - CLIP 사용 시 GPU 이동

```python
# Step 3에서
if joint_model is not None:
    clip_model = clip_model.to(reward.device).float().eval()
    
    # ... CLIP 사용 ...
    
    # 사용 후 CPU로 이동
    clip_model = clip_model.to('cpu')
```

---

## 🐛 디버깅

### 메모리 사용량 모니터링

```python
# train_rl.py 시작 부분에 추가
def print_memory_usage(step):
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"[Step {step}] Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

# train 함수 내에서
for step in range(args.num_steps):
    if step % 10 == 0:
        print_memory_usage(step)
    # ...
```

### nvidia-smi 모니터링

```bash
# 다른 터미널에서
watch -n 1 nvidia-smi
```

---

## ⚡ 긴급 해결 (메모리 부족 시)

### Option 1: 더 작은 배치

```bash
--num-steps 64
--mini-batch-size 1
--grad-accum-steps 64
```

### Option 2: Gradient Checkpointing 강화

```python
# main.py
use_grad_ckpt = True
if use_grad_ckpt:
    base.gradient_checkpointing_enable()  # 추가
    base.enable_input_require_grads()
```

### Option 3: 8-bit Quantization

```bash
# run.sh에 추가
--q8  # 8-bit quantization 활성화
```

---

## 📈 성능 vs 메모리 트레이드오프

| 설정 | 메모리 | 속도 | 성능 |
|------|--------|------|------|
| 기본 (256 steps, bs=4) | 79GB | 빠름 | 최고 |
| 최적화 (128 steps, bs=2) | 55GB | 중간 | 좋음 |
| 긴급 (64 steps, bs=1) | 35GB | 느림 | 괜찮음 |
| 8-bit (256 steps, bs=4) | 45GB | 느림 | 약간 낮음 |

---

## ✅ 체크리스트

- [ ] `run_memory_optimized.sh` 생성
- [ ] `--num-steps 128` 설정
- [ ] `--mini-batch-size 2` 설정
- [ ] `--grad-accum-steps 32` 설정
- [ ] CLIP을 CPU에 로드
- [ ] 메모리 정리 코드 추가
- [ ] `nvidia-smi`로 모니터링
- [ ] WandB에서 메모리 로그 확인

---

## 🎯 최종 권장 사항

1. **run_memory_optimized.sh 사용**
2. **CLIP을 CPU에 로드**
3. **주기적 메모리 정리**
4. **메모리 사용량 모니터링**

이렇게 하면 H200 (79GB)에서 안정적으로 학습 가능합니다!
