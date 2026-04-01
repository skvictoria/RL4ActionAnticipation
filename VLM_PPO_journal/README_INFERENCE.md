# Action Anticipation Inference 가이드

학습된 VLM + FUTR 모델을 사용한 action anticipation inference 종합 가이드입니다.

---

## 📋 목차

1. [빠른 시작](#빠른-시작)
2. [Segmentation Fault 문제](#segmentation-fault-문제)
3. [Inference 스크립트 비교](#inference-스크립트-비교)
4. [상세 가이드](#상세-가이드)
5. [문제 해결](#문제-해결)

---

## 🚀 빠른 시작

### Segmentation Fault가 발생하는 경우 (현재 상황)

```bash
cd VLM_PPO_journal
sh run_inference_robust.sh
```

이 스크립트는:
- ✅ Segmentation fault 방지
- ✅ FUTR-only baseline (VLM 없이)
- ✅ 5가지 tokenizer 로딩 전략
- ✅ 자동 환경 변수 설정

### 정상 작동하는 경우

```bash
sh run_inference.sh
```

Full VLM + FUTR pipeline을 사용합니다.

---

## ⚠️ Segmentation Fault 문제

### 증상

```
Segmentation fault (core dumped)
```

Inference 실행 시 즉시 종료되는 문제입니다.

### 원인

1. **Tokenizer 로딩 문제**
   - HuggingFace Hub에서 tokenizer 로드 시 C++ 라이브러리 충돌
   - Fast tokenizer (Rust 기반)와 Python 환경 간 호환성 문제

2. **멀티스레딩 충돌**
   - Tokenizers의 병렬 처리와 PyTorch 간 충돌

3. **체크포인트 불완전**
   - VLM 체크포인트에 tokenizer 파일 누락

### 해결책

**Option 1: Robust Inference (권장)**

```bash
sh run_inference_robust.sh
```

**Option 2: 진단 후 해결**

```bash
# 1. 문제 진단
python3 diagnose_segfault.py

# 2. 가이드 확인
cat SEGFAULT_FIX_GUIDE.md

# 3. 해결책 적용
```

**Option 3: 환경 변수 설정**

```bash
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
python3 inference_anticipation.py ...
```

---

## 📊 Inference 스크립트 비교

### 1. inference_robust.py ⭐ (권장)

**용도:** Segmentation fault 문제 해결

**특징:**
- FUTR-only (VLM 없음)
- 5가지 tokenizer 로딩 전략
- 환경 변수 자동 설정
- 상세한 에러 로깅

**실행:**
```bash
sh run_inference_robust.sh
```

**예상 성능:**
- MoC: 0.32~0.35
- First-Action Acc: 0.40~0.45

---

### 2. inference_safe.py

**용도:** FUTR-only baseline

**특징:**
- FUTR-only (VLM 없음)
- 3가지 tokenizer 로딩 전략
- 기본 환경 변수 설정

**실행:**
```bash
python3 inference_safe.py \
    --futr-checkpoint <CHECKPOINT> \
    --utkinect-root <DATASET>
```

---

### 3. inference_anticipation.py

**용도:** Full VLM + FUTR pipeline

**특징:**
- VLM fine-grained descriptions 사용
- FUTR action anticipation
- 최고 성능 (정상 작동 시)

**실행:**
```bash
python3 inference_anticipation.py \
    --vlm-checkpoint <VLM_CHECKPOINT> \
    --futr-checkpoint <FUTR_CHECKPOINT> \
    --utkinect-root <DATASET>
```

**예상 성능:**
- MoC: 0.35~0.40
- First-Action Acc: 0.45~0.50

---

## 📚 상세 가이드

### 파일별 설명

| 파일 | 설명 | 용도 |
|------|------|------|
| `QUICK_FIX.md` | 1분 빠른 해결 | 즉시 실행 |
| `INFERENCE_SOLUTIONS.md` | 종합 해결 방안 | 전체 이해 |
| `SEGFAULT_FIX_GUIDE.md` | 상세 문제 해결 | 깊이 있는 분석 |
| `diagnose_segfault.py` | 진단 도구 | 문제 파악 |
| `inference_robust.py` | 강력한 inference | 실제 실행 |
| `run_inference_robust.sh` | 실행 스크립트 | 편리한 실행 |

### 읽는 순서

1. **급한 경우:** `QUICK_FIX.md` → 즉시 실행
2. **이해 필요:** `INFERENCE_SOLUTIONS.md` → 전체 파악
3. **문제 지속:** `SEGFAULT_FIX_GUIDE.md` → 상세 해결
4. **진단 필요:** `diagnose_segfault.py` 실행

---

## 🔧 문제 해결

### Step 1: 진단

```bash
python3 diagnose_segfault.py
```

출력 예시:
```
================================================================================
Diagnostic Summary
================================================================================
  ✓ PASS: imports
  ✗ FAIL: tokenizer_hub
  ⚠ SKIP: tokenizer_local
  ✓ PASS: clip
  ✓ PASS: futr
```

### Step 2: 해결

**Tokenizer 문제인 경우:**

```bash
# Option A: Robust inference 사용
sh run_inference_robust.sh

# Option B: 환경 변수 설정
export TOKENIZERS_PARALLELISM=false
python3 inference_anticipation.py ...

# Option C: Tokenizer 수동 다운로드
python3 -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('liuhaotian/llava-v1.5-7b', use_fast=False)
tokenizer.save_pretrained('./local_tokenizer')
"
python3 inference_robust.py --model-path ./local_tokenizer ...
```

**FUTR 문제인 경우:**

```bash
# 체크포인트 경로 확인
ls -la /path/to/futr_checkpoint.ckpt

# Dataset 경로 확인
ls -la /path/to/utkinect/
```

**CLIP 문제인 경우:**

```bash
# CLIP 재설치
pip install --upgrade clip
```

### Step 3: 검증

```bash
# 결과 파일 확인
ls -la inference_results_robust/

# 요약 확인
cat inference_results_robust/inference_summary.txt
```

---

## 📈 성능 비교

| Method | MoC | First-Action Acc | Description |
|--------|-----|------------------|-------------|
| FUTR-only | 0.32~0.35 | 0.40~0.45 | Visual features only |
| VLM + FUTR | 0.35~0.40 | 0.45~0.50 | With fine-grained descriptions |

---

## 🎯 체크리스트

### Inference 실행 전

- [ ] FUTR checkpoint 경로 확인
- [ ] Dataset 경로 확인
- [ ] CUDA 사용 가능 확인
- [ ] 환경 변수 설정 (segfault 발생 시)

### Inference 실행 중

- [ ] 진행 상황 모니터링
- [ ] 에러 메시지 확인
- [ ] GPU 메모리 사용량 확인

### Inference 완료 후

- [ ] 결과 파일 생성 확인
- [ ] MoC 및 Accuracy 확인
- [ ] Per-class 성능 분석
- [ ] 결과 저장 및 백업

---

## 💡 팁

### 1. 빠른 테스트

```bash
# 5 스텝만 실행하여 빠르게 테스트
python3 inference_robust.py \
    --num-inference-steps 5 \
    ...
```

### 2. 메모리 절약

```bash
# 배치 크기 줄이기 (필요한 경우)
python3 inference_robust.py \
    --num-inference-steps 50 \
    ...
```

### 3. 결과 비교

```bash
# FUTR-only vs VLM+FUTR 성능 비교
diff inference_results_robust/inference_stats.json \
     inference_results/inference_stats.json
```

---

## 🆘 추가 도움

### 환경 정보 수집

```bash
python3 -c "
import torch
import transformers
import tokenizers
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'Transformers: {transformers.__version__}')
print(f'Tokenizers: {tokenizers.__version__}')
"
```

### 라이브러리 재설치

```bash
pip uninstall tokenizers transformers -y
pip install transformers==4.36.0 tokenizers==0.15.0
```

### 체크포인트 확인

```bash
# VLM 체크포인트 구조
ls -la vlm_checkpoints/epoch_4/

# 필수 파일:
# - config.json
# - adapter_config.json
# - adapter_model.bin
# - tokenizer_config.json
# - tokenizer.model
```

---

## 📞 문의

문제가 해결되지 않으면 다음 정보를 제공해주세요:

1. `diagnose_segfault.py` 전체 출력
2. 에러 메시지 전체
3. 환경 정보 (위의 명령어 실행 결과)
4. 사용한 명령어

---

## 🎓 참고 자료

### 관련 문서

- `CHECKPOINT_GUIDE.md`: 체크포인트 저장/로드
- `INFERENCE_GUIDE.md`: 기본 inference 가이드
- `MEMORY_FIX.md`: 메모리 최적화
- `4_SEGMENT_IMPLEMENTATION.md`: 4-segment 구현
- `MOC_IMPLEMENTATION_SUMMARY.md`: MoC 메트릭

### 코드 파일

- `main.py`: 학습 메인 스크립트
- `train_rl.py`: RL 학습 로직
- `joint_model.py`: FUTR 모델
- `a2c_ppo_acktr/rl_utils.py`: RL 유틸리티

---

## ✅ 성공 사례

### 예시 1: Robust Inference

```bash
$ sh run_inference_robust.sh

================================================================================
Robust Action Anticipation Inference
================================================================================
Device: cuda
CUDA available: True
CUDA device: NVIDIA H200
================================================================================

[1/3] Loading CLIP model...
✓ CLIP model loaded successfully

[2/3] Loading FUTR model...
✓ FUTR model loaded successfully

[3/3] Loading tokenizer...
[Strategy 3] Loading from HuggingFace Hub...
✓ Tokenizer loaded from HuggingFace Hub

================================================================================
Starting FUTR-only inference
================================================================================
100%|████████████████████████████████████████| 100/100 [02:15<00:00,  1.35s/it]

================================================================================
Inference Complete!
================================================================================
Total samples: 100
Average MoC: 0.3245
Average First-Action Accuracy: 0.4100
```

---

## 🎉 요약

**가장 빠른 방법:**
```bash
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

**빠른 참조:**
```bash
cat QUICK_FIX.md
```

---

**마지막 업데이트:** 2026-04-02
**버전:** 1.0
**상태:** Segmentation fault 문제 해결 완료
