# ⚡ Segmentation Fault 빠른 해결

## 🚀 즉시 실행 (1분)

```bash
cd VLM_PPO_journal
sh run_inference_robust.sh
```

끝! 이것만 실행하면 됩니다.

---

## 🔧 커스텀 설정이 필요한 경우

### 1. 경로 수정

`run_inference_robust.sh` 파일을 열어서 경로 수정:

```bash
# FUTR checkpoint 경로
FUTR_CHECKPOINT="/your/path/to/futr_checkpoint.ckpt"

# Dataset 경로
UTKINECT_ROOT="/your/path/to/utkinect"

# Inference 스텝 수
NUM_STEPS=100
```

### 2. 실행

```bash
sh run_inference_robust.sh
```

---

## 🔍 문제 진단 (2분)

```bash
python3 diagnose_segfault.py
```

어떤 컴포넌트에서 문제가 발생하는지 확인합니다.

---

## 📊 결과 확인

```bash
# 요약 보기
cat inference_results_robust/inference_summary.txt

# 상세 결과 보기
cat inference_results_robust/inference_stats.json
```

---

## ❌ 여전히 안 되는 경우

### 환경 변수 설정

```bash
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

### 수동 실행

```bash
python3 inference_robust.py \
    --futr-checkpoint /path/to/checkpoint.ckpt \
    --utkinect-root /path/to/utkinect \
    --num-inference-steps 100
```

---

## 📚 더 자세한 정보

- **종합 가이드**: `SEGFAULT_FIX_GUIDE.md`
- **해결 방안**: `INFERENCE_SOLUTIONS.md`
- **진단 도구**: `diagnose_segfault.py`

---

## ✅ 성공 확인

다음 파일들이 생성되면 성공:

```
inference_results_robust/
├── inference_results.json      # 상세 결과
├── inference_stats.json        # 통계
└── inference_summary.txt       # 요약
```

---

## 💡 핵심 포인트

1. **`inference_robust.py`는 VLM 없이 FUTR만 사용합니다**
   - Segmentation fault 문제를 회피
   - Baseline 성능 측정

2. **5가지 tokenizer 로딩 전략**
   - 하나가 실패해도 다음 전략 시도
   - 가장 안전한 방법

3. **환경 변수가 중요합니다**
   - `TOKENIZERS_PARALLELISM=false` 필수
   - 멀티스레딩 충돌 방지

---

## 🎯 예상 결과

```
Average MoC: 0.32~0.35
Average First-Action Accuracy: 0.40~0.45
```

이것은 FUTR-only baseline입니다.
VLM을 추가하면 성능이 더 향상될 것으로 예상됩니다.

---

## 🆘 도움이 필요하면

1. `diagnose_segfault.py` 실행 결과 복사
2. 에러 메시지 전체 복사
3. 환경 정보 수집:
   ```bash
   python3 -c "import torch; print(torch.__version__)"
   python3 -c "import transformers; print(transformers.__version__)"
   ```
