# Inference 스크립트 비교

## 📊 스크립트 종류

### 1. inference_robust.py (FUTR-only)

**용도:** Segmentation fault 회피용 baseline

**특징:**
- ❌ VLM 사용 안 함
- ✅ FUTR만 사용
- ✅ 가장 안전함
- ✅ Segfault 문제 없음

**성능:**
- MoC: 0.32~0.35
- First-Action Acc: 0.40~0.45

**실행:**
```bash
sh run_inference_robust.sh
```

**인자:**
```bash
python3 inference_robust.py \
    --futr-checkpoint <FUTR_CHECKPOINT> \
    --utkinect-root <DATASET> \
    --model-path liuhaotian/llava-v1.5-7b  # tokenizer만 사용
```

---

### 2. inference_anticipation.py (VLM + FUTR)

**용도:** 학습된 VLM 체크포인트 사용 (Full pipeline)

**특징:**
- ✅ VLM 사용 (fine-grained descriptions)
- ✅ FUTR 사용
- ⚠️ Flash Attention 문제 가능
- ✅ 최고 성능

**성능:**
- MoC: 0.35~0.40 (예상)
- First-Action Acc: 0.45~0.50 (예상)

**실행:**
```bash
sh run_inference_with_vlm.sh
```

**인자:**
```bash
python3 inference_anticipation.py \
    --vlm-checkpoint <VLM_CHECKPOINT> \      # ← VLM 체크포인트!
    --futr-checkpoint <FUTR_CHECKPOINT> \
    --utkinect-root <DATASET> \
    --model-path liuhaotian/llava-v1.5-7b
```

---

## 🎯 어떤 것을 사용해야 하나?

### Segmentation Fault가 해결되었다면

**VLM + FUTR 사용 (권장):**
```bash
# Flash Attention 제거 후
pip uninstall flash-attn -y

# VLM + FUTR inference
sh run_inference_with_vlm.sh
```

---

### Segmentation Fault가 계속된다면

**FUTR-only 사용 (안전):**
```bash
sh run_inference_robust.sh
```

---

## 📈 성능 비교

| Method | VLM | FUTR | MoC | First-Acc | Segfault Risk |
|--------|-----|------|-----|-----------|---------------|
| `inference_robust.py` | ❌ | ✅ | 0.32~0.35 | 0.40~0.45 | ✅ 없음 |
| `inference_anticipation.py` | ✅ | ✅ | 0.35~0.40 | 0.45~0.50 | ⚠️ 있음 (flash_attn) |

---

## 🔧 VLM 체크포인트 사용하기

### Step 1: Flash Attention 제거

```bash
pip uninstall flash-attn -y
```

### Step 2: VLM 체크포인트 경로 확인

```bash
# VLM 체크포인트 확인
ls -la /path/to/vlm_checkpoints/epoch_66/

# 필수 파일 확인
ls -la /path/to/vlm_checkpoints/epoch_66/adapter_model.bin
```

### Step 3: run_inference_with_vlm.sh 수정

```bash
nano run_inference_with_vlm.sh
```

**수정할 부분:**
```bash
# VLM checkpoint 경로
VLM_CHECKPOINT="/home/hice1/skim3513/scratch/.../vlm_checkpoints/epoch_66"

# FUTR checkpoint 경로
FUTR_CHECKPOINT="/home/hice1/skim3513/scratch/.../futr_joint_epoch_66.ckpt"
```

### Step 4: 실행

```bash
sh run_inference_with_vlm.sh
```

---

## 🚀 빠른 시작

### FUTR-only (안전, VLM 없음)

```bash
sh run_inference_robust.sh
```

### VLM + FUTR (최고 성능, VLM 사용)

```bash
# 1. Flash Attention 제거
pip uninstall flash-attn -y

# 2. 경로 수정
nano run_inference_with_vlm.sh

# 3. 실행
sh run_inference_with_vlm.sh
```

---

## 📝 왜 두 가지 스크립트가 있나?

### 이유

1. **Segmentation Fault 문제**
   - VLM 로딩 시 flash_attn 문제 발생 가능
   - FUTR-only는 이 문제를 회피

2. **단계적 테스트**
   - FUTR-only로 먼저 baseline 확인
   - 문제 없으면 VLM 추가

3. **성능 비교**
   - FUTR-only: Baseline 성능
   - VLM + FUTR: Fine-grained descriptions 효과 측정

---

## 🎯 추천 워크플로우

### Step 1: FUTR-only Baseline

```bash
sh run_inference_robust.sh
```

**목적:**
- 시스템이 정상 작동하는지 확인
- Baseline 성능 측정

---

### Step 2: Flash Attention 제거

```bash
pip uninstall flash-attn -y
```

---

### Step 3: VLM + FUTR (Full Pipeline)

```bash
sh run_inference_with_vlm.sh
```

**목적:**
- 학습된 VLM 체크포인트 사용
- Fine-grained descriptions 효과 확인
- 최고 성능 달성

---

### Step 4: 성능 비교

```bash
# FUTR-only 결과
cat inference_results_robust/inference_summary.txt

# VLM + FUTR 결과
cat inference_results_with_vlm/inference_summary.txt
```

---

## 📊 예상 결과

### FUTR-only

```
Average MoC: 0.3245
Average First-Action Accuracy: 0.4100

Note: This is FUTR-only baseline (no VLM fine-grained descriptions)
```

### VLM + FUTR

```
Average MoC: 0.3780
Average First-Action Accuracy: 0.4650

Note: Using VLM fine-grained descriptions
```

**개선:** MoC +16%, First-Acc +13%

---

## 🔍 VLM 체크포인트 확인

### 체크포인트 구조

```bash
vlm_checkpoints/epoch_66/
├── adapter_config.json      # LoRA 설정
├── adapter_model.bin        # LoRA 가중치 (중요!)
├── config.json              # 모델 설정
├── tokenizer_config.json    # Tokenizer 설정
└── tokenizer.model          # Tokenizer
```

### 확인 명령어

```bash
# 디렉토리 존재 확인
ls -la vlm_checkpoints/epoch_66/

# 파일 크기 확인
du -h vlm_checkpoints/epoch_66/adapter_model.bin
# 예상: ~256MB
```

---

## ⚠️ 주의사항

### VLM + FUTR 사용 시

1. **Flash Attention 제거 필수**
   ```bash
   pip uninstall flash-attn -y
   ```

2. **VLM 체크포인트 경로 확인**
   ```bash
   ls -la /path/to/vlm_checkpoints/epoch_66/
   ```

3. **충분한 GPU 메모리**
   - VLM + FUTR: ~20GB VRAM 필요
   - FUTR-only: ~10GB VRAM 필요

---

## 🎉 요약

### FUTR-only (inference_robust.py)

**장점:**
- ✅ 안전함 (segfault 없음)
- ✅ 빠름
- ✅ 메모리 적게 사용

**단점:**
- ❌ VLM 사용 안 함
- ❌ 성능 낮음

**사용 시기:**
- Segfault 문제 해결 전
- Baseline 성능 측정
- 빠른 테스트

---

### VLM + FUTR (inference_anticipation.py)

**장점:**
- ✅ 최고 성능
- ✅ Fine-grained descriptions 사용
- ✅ 학습된 VLM 활용

**단점:**
- ⚠️ Flash Attention 문제 가능
- ⚠️ 메모리 많이 사용
- ⚠️ 느림

**사용 시기:**
- Flash Attention 제거 후
- 최종 성능 측정
- 논문 결과 생성

---

## 🚀 지금 바로 실행

### VLM 체크포인트를 사용하고 싶다면

```bash
# 1. Flash Attention 제거
pip uninstall flash-attn -y

# 2. VLM + FUTR inference
sh run_inference_with_vlm.sh
```

### 안전하게 테스트하고 싶다면

```bash
# FUTR-only inference
sh run_inference_robust.sh
```

---

**핵심:** `inference_robust.py`는 VLM 없는 baseline이고, `inference_anticipation.py`가 VLM 체크포인트를 사용합니다!
