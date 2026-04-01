# Inference Segmentation Fault 해결 방안

## 문제 요약

Inference 실행 시 segmentation fault 발생:
- `--model-path liuhaotian/llava-v1.5-7b`에서 tokenizer 로드 실패
- `--model-path vlm_checkpoints/epoch_4`에서도 동일한 문제 발생

## 제공된 해결책

### 🎯 Solution 1: Robust Inference (권장)

**가장 안전하고 강력한 방법**

```bash
cd VLM_PPO_journal
sh run_inference_robust.sh
```

**특징:**
- ✅ 5가지 tokenizer 로딩 전략 (자동 fallback)
- ✅ FUTR-only inference (VLM 없이 실행)
- ✅ 환경 변수 자동 설정
- ✅ 상세한 에러 로깅 및 진단
- ✅ Per-class 성능 분석

**Tokenizer 로딩 전략 (순차적 시도):**
1. Local checkpoint (config.json 있는 경우)
2. LoRA checkpoint의 base model
3. HuggingFace Hub (liuhaotian/llava-v1.5-7b)
4. Llama tokenizer (huggyllama/llama-7b)
5. Llama-2 tokenizer (meta-llama/Llama-2-7b-hf)

**파일:**
- `inference_robust.py`: 메인 스크립트
- `run_inference_robust.sh`: 실행 스크립트

---

### 🔍 Solution 2: Diagnostic Tool

**문제의 정확한 원인 파악**

```bash
cd VLM_PPO_journal
python3 diagnose_segfault.py
```

**테스트 항목:**
1. ✓ 기본 라이브러리 import (PyTorch, Transformers, CLIP)
2. ✓ HuggingFace Hub tokenizer 로드
3. ✓ Local checkpoint tokenizer 로드
4. ✓ CLIP 모델 로드 및 인코딩
5. ✓ FUTR 모델 로드
6. ✓ 환경 변수 확인

**출력 예시:**
```
================================================================================
1. Testing Basic Imports
================================================================================
✓ PyTorch: 2.0.1
  CUDA available: True
  CUDA version: 12.1
✓ Transformers: 4.36.0
✓ Tokenizers: 0.15.0
✓ CLIP imported successfully

================================================================================
2. Testing Tokenizer from HuggingFace Hub
================================================================================
✓ Tokenizer loaded from HuggingFace Hub
  Vocab size: 32000
✓ Tokenization test passed

...
```

**파일:**
- `diagnose_segfault.py`: 진단 스크립트

---

### 📚 Solution 3: Comprehensive Guide

**상세한 문제 해결 가이드**

```bash
cat VLM_PPO_journal/SEGFAULT_FIX_GUIDE.md
```

**포함 내용:**
- 원인 분석 (tokenizer, 메모리, 라이브러리 충돌)
- 4가지 해결 방법 (단계별 설명)
- 체크포인트 구조 확인 방법
- 디버깅 방법 (코드 예제 포함)
- 일반적인 에러와 해결책
- 추천 워크플로우

**파일:**
- `SEGFAULT_FIX_GUIDE.md`: 종합 가이드

---

## 빠른 시작 가이드

### Step 1: 진단 실행

```bash
cd VLM_PPO_journal
python3 diagnose_segfault.py \
    --vlm-checkpoint ./vlm_checkpoints/epoch_4 \
    --futr-checkpoint /path/to/futr_checkpoint.ckpt \
    --dataset-root /path/to/utkinect
```

어떤 컴포넌트에서 문제가 발생하는지 확인합니다.

### Step 2: Robust Inference 실행

```bash
sh run_inference_robust.sh
```

또는 커스텀 설정:

```bash
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

python3 inference_robust.py \
    --futr-checkpoint /path/to/futr_checkpoint.ckpt \
    --utkinect-root /path/to/utkinect \
    --model-path liuhaotian/llava-v1.5-7b \
    --num-inference-steps 100 \
    --output-dir ./results
```

### Step 3: 결과 확인

```bash
ls -la inference_results_robust/
cat inference_results_robust/inference_summary.txt
```

---

## 파일 구조

```
VLM_PPO_journal/
├── inference_robust.py              # 강력한 inference 스크립트 (NEW)
├── run_inference_robust.sh          # 실행 스크립트 (NEW)
├── diagnose_segfault.py             # 진단 도구 (NEW)
├── SEGFAULT_FIX_GUIDE.md           # 종합 가이드 (NEW)
├── INFERENCE_SOLUTIONS.md          # 이 파일 (NEW)
│
├── inference_safe.py                # FUTR-only baseline (기존)
├── inference_anticipation.py        # Full VLM+FUTR (기존)
├── INFERENCE_GUIDE.md              # 기본 가이드 (기존)
└── CHECKPOINT_GUIDE.md             # 체크포인트 가이드 (기존)
```

---

## 각 Inference 스크립트 비교

| Script | VLM | FUTR | Tokenizer Strategy | Robustness | Use Case |
|--------|-----|------|-------------------|------------|----------|
| `inference_robust.py` | ❌ | ✅ | 5-level fallback | ⭐⭐⭐⭐⭐ | **Segfault 문제 해결** |
| `inference_safe.py` | ❌ | ✅ | 3-level fallback | ⭐⭐⭐⭐ | FUTR baseline |
| `inference_anticipation.py` | ✅ | ✅ | Single strategy | ⭐⭐⭐ | Full pipeline (정상 작동 시) |

---

## 예상 결과

### FUTR-only Baseline (inference_robust.py)

```
================================================================================
Inference Complete!
================================================================================
Total samples: 100
Average MoC: 0.3245
Average First-Action Accuracy: 0.4100

Per-class First-Action Accuracy:
  carry: 0.4500 (20 samples)
  clap: 0.3800 (15 samples)
  pick: 0.4200 (18 samples)
  ...

Note: This is FUTR-only baseline (no VLM fine-grained descriptions)
================================================================================
```

### Full Pipeline (inference_anticipation.py)

VLM fine-grained descriptions를 사용하면 성능 향상 예상:
- MoC: 0.35~0.40 (예상)
- First-Action Acc: 0.45~0.50 (예상)

---

## 문제가 계속될 때

### 1. 환경 정보 수집

```bash
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python3 -c "import tokenizers; print(f'Tokenizers: {tokenizers.__version__}')"
```

### 2. 라이브러리 재설치

```bash
pip uninstall tokenizers transformers -y
pip install transformers==4.36.0 tokenizers==0.15.0
```

### 3. Tokenizer 수동 다운로드

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "liuhaotian/llava-v1.5-7b",
    use_fast=False,
    cache_dir="./tokenizer_cache"
)
tokenizer.save_pretrained("./local_tokenizer")
```

이후:
```bash
python3 inference_robust.py --model-path ./local_tokenizer ...
```

---

## 추가 도움말

### VLM 체크포인트 확인

```bash
# 필요한 파일들이 있는지 확인
ls -la vlm_checkpoints/epoch_4/

# 필수 파일:
# - config.json
# - adapter_config.json
# - adapter_model.bin
# - tokenizer_config.json
# - tokenizer.model (또는 vocab.json)
```

### 학습 시 Tokenizer 저장 확인

`main.py`에서 체크포인트 저장 시:

```python
# VLM 저장
vlm_model.save_pretrained(vlm_save_dir)
tokenizer.save_pretrained(vlm_save_dir)  # ← 이 줄이 있는지 확인!
```

---

## 요약

**가장 빠른 해결책:**

```bash
cd VLM_PPO_journal
sh run_inference_robust.sh
```

**문제 진단:**

```bash
python3 diagnose_segfault.py
```

**상세 가이드:**

```bash
cat SEGFAULT_FIX_GUIDE.md
```

---

## 성공 체크리스트

- [ ] `diagnose_segfault.py` 실행하여 문제 파악
- [ ] 환경 변수 설정 (`TOKENIZERS_PARALLELISM=false`)
- [ ] `inference_robust.py` 실행
- [ ] 결과 파일 생성 확인 (`inference_results_robust/`)
- [ ] MoC 및 First-Action Accuracy 확인
- [ ] (선택) Full VLM+FUTR pipeline 테스트

---

## 연락처

문제가 해결되지 않으면:
1. `diagnose_segfault.py` 출력 전체 복사
2. 에러 메시지 전체 복사
3. 환경 정보 (PyTorch, Transformers 버전) 포함

이 정보를 제공하면 더 정확한 해결책을 제시할 수 있습니다.
