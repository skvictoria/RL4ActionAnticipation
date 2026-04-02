# 체크포인트 저장 형식 비교

## 두 가지 저장 형식

학습 시 **동일한 epoch**에 대해 **두 가지 형식**으로 저장됩니다:

### 1. LoRA 디렉토리 형식 (작음, ~256MB)
```
vlm_checkpoints/epoch_66/
├── adapter_config.json      # LoRA 설정
├── adapter_model.bin        # LoRA 가중치만 (~256MB)
├── config.json              # 모델 설정
├── tokenizer_config.json    # Tokenizer 설정
└── tokenizer.model          # Tokenizer
```

**용도:** ✅ **Inference (추론)**
- LoRA weights만 포함
- 작고 가벼움 (~256MB)
- Base model + LoRA weights 조합으로 사용

### 2. .pt 파일 형식 (큼, ~30GB)
```
vlm_checkpoints/vlm_epoch_66.pt
```

**용도:** ✅ **Training Resume (학습 재개)**
- 전체 모델 state_dict 포함 (~30GB)
- Optimizer state 포함
- Learning rate scheduler state 포함
- Iteration 정보 포함

---

## 상세 비교

| 항목 | LoRA 디렉토리 | .pt 파일 |
|------|--------------|----------|
| **크기** | ~256MB | ~30GB |
| **포함 내용** | LoRA weights만 | 전체 모델 + optimizer + scheduler |
| **용도** | Inference | Training resume |
| **로딩 방법** | `PeftModel.from_pretrained()` | `torch.load()` |
| **Base model 필요?** | ✅ 필요 | ❌ 불필요 (전체 포함) |

---

## .pt 파일 내용 (30GB)

```python
{
    'iteration': 66,                          # 학습 iteration
    'model_state_dict': {...},                # 전체 모델 가중치 (~30GB)
    'optimizer_state_dict': {...},            # Optimizer 상태
    'lr_scheduler_state_dict': {...},         # LR scheduler 상태
}
```

**왜 30GB나 될까?**
- `model_state_dict`에 **전체 LLaVA 모델**이 포함됨
- Base model (7B parameters) + LoRA weights
- Optimizer state (Adam의 경우 momentum, variance 저장)

---

## 언제 어떤 형식을 사용하나?

### Inference (추론) → LoRA 디렉토리 사용

```python
# inference_anticipation.py에서
from peft import PeftModel

# Base model 로드
base = LlavaLlamaForCausalLM.from_pretrained("liuhaotian/llava-v1.5-7b")

# LoRA weights 로드 (256MB만 로드)
model = PeftModel.from_pretrained(base, "./vlm_checkpoints/epoch_66")
```

**장점:**
- 빠른 로딩 (256MB만)
- 메모리 효율적
- 여러 LoRA를 쉽게 교체 가능

---

### Training Resume (학습 재개) → .pt 파일 사용

```python
# main.py에서 학습 재개 시
checkpoint = torch.load("vlm_checkpoints/vlm_epoch_66.pt")

# 모델 복원
model.load_state_dict(checkpoint['model_state_dict'])

# Optimizer 복원 (중요!)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# LR scheduler 복원
lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

# Iteration 복원
start_iteration = checkpoint['iteration'] + 1
```

**장점:**
- 정확히 같은 상태에서 학습 재개
- Optimizer momentum 보존
- Learning rate schedule 보존

---

## 실제 사용 예시

### 시나리오 1: Inference만 하고 싶음

```bash
# LoRA 디렉토리만 사용 (256MB)
python3 inference_anticipation.py \
    --vlm-checkpoint ./vlm_checkpoints/epoch_66 \
    --futr-checkpoint ./futr_joint_epoch_66.ckpt \
    ...
```

**필요한 파일:**
- `vlm_checkpoints/epoch_66/` (256MB)
- `futr_joint_epoch_66.ckpt` (1.2GB)

**불필요한 파일:**
- `vlm_checkpoints/vlm_epoch_66.pt` (30GB) ← 삭제 가능!

---

### 시나리오 2: 학습을 더 하고 싶음 (Resume)

```bash
# .pt 파일 사용 (30GB)
python3 main.py \
    --resume-from ./vlm_checkpoints/vlm_epoch_66.pt \
    ...
```

**필요한 파일:**
- `vlm_checkpoints/vlm_epoch_66.pt` (30GB)
- `futr_joint_epoch_66.ckpt` (1.2GB)

**불필요한 파일:**
- `vlm_checkpoints/epoch_66/` (256MB) ← 삭제 가능 (inference용)

---

## 디스크 공간 절약 팁

### 옵션 1: Inference만 할 경우

```bash
# .pt 파일 삭제 (30GB 절약)
rm vlm_checkpoints/vlm_epoch_*.pt

# LoRA 디렉토리만 유지
ls vlm_checkpoints/
# epoch_0/  epoch_10/  epoch_20/  ...  epoch_66/
```

---

### 옵션 2: 학습 재개가 필요한 경우

```bash
# 최신 .pt 파일만 유지
rm vlm_checkpoints/vlm_epoch_{0..60}.pt

# 최신 것만 남김
ls vlm_checkpoints/vlm_epoch_*.pt
# vlm_epoch_66.pt
```

---

### 옵션 3: 중간 체크포인트 정리

```bash
# 10 epoch마다만 유지
cd vlm_checkpoints
rm -rf epoch_{1..9}
rm -rf epoch_{11..19}
rm -rf epoch_{21..29}
# ...

# 결과: epoch_0, epoch_10, epoch_20, ..., epoch_66만 유지
```

---

## 저장 코드 분석

### main.py에서 저장 시

```python
# 1. LoRA 디렉토리 저장 (Inference용)
vlm_save_dir = os.path.join(vlm_checkpoint_dir, f"epoch_{j}")
vlm_model.save_pretrained(vlm_save_dir)  # LoRA weights만
tokenizer.save_pretrained(vlm_save_dir)

# 2. .pt 파일 저장 (Resume용)
vlm_checkpoint_file = os.path.join(vlm_checkpoint_dir, f"vlm_epoch_{j}.pt")
torch.save({
    'iteration': j,
    'model_state_dict': vlm_model.state_dict(),      # 전체 모델
    'optimizer_state_dict': optimizer.state_dict(),  # Optimizer
    'lr_scheduler_state_dict': lr_scheduler.state_dict(),
}, vlm_checkpoint_file)
```

---

## 로딩 코드 비교

### LoRA 디렉토리 로딩 (Inference)

```python
# inference_anticipation.py
from peft import PeftModel

# Base model
base = LlavaLlamaForCausalLM.from_pretrained("liuhaotian/llava-v1.5-7b")

# LoRA weights 로드
if os.path.isdir(args.vlm_checkpoint):
    model = PeftModel.from_pretrained(base, args.vlm_checkpoint)
```

---

### .pt 파일 로딩 (Training Resume)

```python
# main.py (resume 시)
checkpoint = torch.load(args.vlm_checkpoint, map_location=device)

# 모델 로드
base = get_peft_model(base, lora_config)
base.load_state_dict(checkpoint['model_state_dict'])

# Optimizer 로드
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# LR scheduler 로드
lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

# Iteration 복원
start_epoch = checkpoint['iteration'] + 1
```

---

## 자주 묻는 질문

### Q1: 30GB .pt 파일을 inference에 사용할 수 있나요?

**A:** 가능하지만 비효율적입니다.

```python
# 가능하지만 권장하지 않음
checkpoint = torch.load("vlm_epoch_66.pt")
model.load_state_dict(checkpoint['model_state_dict'])
```

**문제점:**
- 30GB 전체를 로드해야 함 (느림)
- 메모리 낭비
- LoRA 디렉토리 (256MB)를 사용하는 것이 훨씬 효율적

---

### Q2: LoRA 디렉토리로 학습을 재개할 수 있나요?

**A:** 불가능합니다.

**이유:**
- Optimizer state가 없음
- LR scheduler state가 없음
- 학습이 불안정해짐 (momentum 손실)

**해결:**
- .pt 파일 사용
- 또는 처음부터 다시 학습

---

### Q3: 어떤 파일을 백업해야 하나요?

**A:** 용도에 따라 다릅니다.

**Inference만 할 경우:**
```bash
# 백업 필요
vlm_checkpoints/epoch_66/
futr_joint_epoch_66.ckpt

# 백업 불필요
vlm_checkpoints/vlm_epoch_66.pt  # 30GB 절약!
```

**학습 재개가 필요한 경우:**
```bash
# 백업 필요
vlm_checkpoints/vlm_epoch_66.pt
futr_joint_epoch_66.ckpt

# 백업 불필요 (재생성 가능)
vlm_checkpoints/epoch_66/
```

---

### Q4: 두 형식의 성능 차이가 있나요?

**A:** 없습니다!

- 동일한 epoch의 동일한 weights
- Inference 성능은 완전히 동일
- 차이는 로딩 속도와 메모리 사용량뿐

---

## 권장 사항

### Inference 사용자

```bash
# LoRA 디렉토리만 유지
vlm_checkpoints/
├── epoch_66/          # ← 이것만 사용
│   ├── adapter_model.bin
│   └── ...
└── vlm_epoch_66.pt    # ← 삭제 가능 (30GB 절약)
```

---

### 연구자 (학습 재개 필요)

```bash
# 최신 .pt 파일 유지
vlm_checkpoints/
├── epoch_66/          # ← Inference 테스트용
│   └── ...
└── vlm_epoch_66.pt    # ← 학습 재개용 (중요!)
```

---

## 요약

| 파일 | 크기 | 용도 | 삭제 가능? |
|------|------|------|-----------|
| `epoch_66/` | 256MB | Inference | 학습 재개 시 ✅ |
| `vlm_epoch_66.pt` | 30GB | Training resume | Inference만 할 경우 ✅ |

**핵심:**
- **Inference만 한다면** → LoRA 디렉토리만 사용, .pt 삭제 (30GB 절약)
- **학습을 더 한다면** → .pt 파일 유지, LoRA 디렉토리는 선택사항
- **두 파일의 성능은 동일**, 차이는 용도와 크기뿐!

---

## 실전 예시

### 당신의 상황

```bash
# 현재 가지고 있는 파일
vlm_checkpoints/
├── epoch_66/              # 256MB (Inference용)
└── vlm_epoch_66.pt        # 30GB (Resume용)

# Inference만 할 경우
rm vlm_checkpoints/vlm_epoch_66.pt  # 30GB 절약!

# 학습을 더 할 경우
# 두 파일 모두 유지
```

**추천:**
- 지금은 inference 테스트 중이므로 LoRA 디렉토리만 사용
- .pt 파일은 나중에 학습을 더 할 때까지 백업 또는 삭제
