# Action Anticipation Inference Guide

학습된 VLM + FUTR 모델을 사용하여 action anticipation 수행하는 가이드입니다.

---

## 📋 준비사항

### 1. 학습된 체크포인트

학습 중 저장된 체크포인트가 필요합니다:

```
checkpoints/
├── vlm_epoch_10.pt          # VLM (LLaVA + LoRA) 체크포인트
└── futr_joint_epoch_10.ckpt # FUTR 체크포인트
```

**체크포인트 저장 위치**:
- VLM: `main.py`에서 `--save_interval` 마다 저장
- FUTR: `train_rl.py`에서 매 iteration마다 저장

### 2. 데이터셋

UTKinect 데이터셋의 test split:
```
utkinect/
├── groundTruth/
├── features_img/
├── splits/
│   └── test_split.txt
└── mapping_l2_changed.txt
```

---

## 🚀 실행 방법

### Option 1: 스크립트 사용 (권장)

```bash
# 1. 스크립트 편집 (체크포인트 경로 수정)
vim run_inference.sh

# 2. 실행 권한 부여
chmod +x run_inference.sh

# 3. 실행
sh run_inference.sh
```

### Option 2: 직접 실행

```bash
python inference_anticipation.py \
    --model-path liuhaotian/llava-v1.5-7b \
    --vlm-checkpoint ./checkpoints/vlm_epoch_10.pt \
    --futr-checkpoint ./checkpoints/futr_joint_epoch_10.ckpt \
    --env-name utkinect/test \
    --utkinect-root /path/to/utkinect \
    --utkinect-split test \
    --num-inference-steps 100 \
    --output-dir ./inference_results
```

---

## 📊 출력 결과

### 1. 콘솔 출력

```
================================================================================
Loading trained models...
================================================================================
Loading VLM checkpoint from: ./checkpoints/vlm_epoch_10.pt
✓ VLM checkpoint loaded successfully
Loading CLIP model...
✓ CLIP model loaded
Loading FUTR model...
✓ FUTR model loaded

================================================================================
Starting inference...
================================================================================
Step 10/100
  Avg MoC: 0.6234
  Avg First-Action Acc: 0.7500

...

================================================================================
Inference Complete!
================================================================================
Total samples: 100
Average MoC: 0.6234
Average First-Action Accuracy: 0.7500
```

### 2. 저장 파일

**inference_results_YYYYMMDD_HHMMSS/**
```
├── inference_results.json    # 상세 결과 (각 샘플별)
├── inference_stats.json      # 통계 요약
└── inference_summary.txt     # 텍스트 요약
```

#### inference_results.json 예시

```json
[
  {
    "step": 0,
    "sequence_id": "seq_001",
    "frame_index": 42,
    "coarse_labels": ["walking", "reaching", "grasping"],
    "fine_grained_descriptions": [
      "The person walks forward with arms swinging naturally",
      "The person extends right arm toward an object",
      "The person's hand closes around the object",
      "The person lifts the grasped object"
    ],
    "predicted_future": [
      "grasping", "grasping", "placing", "placing",
      "placing", "placing", "releasing", "releasing",
      ...
    ],
    "target_future": [
      "grasping", "grasping", "placing", "placing",
      ...
    ],
    "moc_score": 0.8333,
    "first_action_accuracy": 1.0
  },
  ...
]
```

#### inference_stats.json 예시

```json
{
  "avg_moc": 0.6234,
  "avg_first_acc": 0.7500
}
```

---

## 🔧 주요 파라미터

### 모델 관련

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `--model-path` | Base LLaVA 모델 경로 | `liuhaotian/llava-v1.5-7b` |
| `--vlm-checkpoint` | 학습된 VLM 체크포인트 | **필수** |
| `--futr-checkpoint` | 학습된 FUTR 체크포인트 | **필수** |
| `--cache-dir` | 모델 캐시 디렉토리 | `None` |

### 데이터셋 관련

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `--utkinect-root` | UTKinect 데이터셋 경로 | **필수** |
| `--utkinect-split` | 데이터 split | `test` |
| `--utkinect-history` | History window 크기 | `6` |
| `--utkinect-frame-skip` | Frame skip | `1` |

### 추론 설정

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `--num-inference-steps` | 추론 스텝 수 | `100` |
| `--temperature` | Sampling temperature | `0.2` |
| `--max-new-tokens` | 최대 생성 토큰 수 | `256` |
| `--conv-mode` | Conversation mode | `vicuna_v1` |

### 출력 관련

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `--output-dir` | 결과 저장 디렉토리 | `./inference_results` |
| `--seed` | Random seed | `1` |

---

## 📈 평가 메트릭

### 1. MoC (Mean over Classes)

클래스별 recall의 평균:

```
MoC = (Recall_class1 + Recall_class2 + ... + Recall_classN) / N
```

**해석**:
- 0.0 ~ 1.0 범위
- 1.0에 가까울수록 좋음
- 불균형 데이터에서도 공평한 평가

### 2. First-Action Accuracy

첫 번째 미래 행동의 정확도:

```
Accuracy = (Correct predictions) / (Total predictions)
```

**해석**:
- 0.0 ~ 1.0 범위
- 바로 다음 행동 예측 성능
- 실용적 메트릭

---

## 🐛 문제 해결

### 1. 체크포인트 로드 실패

```
Error: No such file or directory: './checkpoints/vlm_epoch_10.pt'
```

**해결**:
- 체크포인트 경로 확인
- 학습이 완료되었는지 확인
- `--vlm-checkpoint` 경로 수정

### 2. CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**해결**:
```bash
# Batch size 줄이기 (코드 수정 필요)
# 또는 더 작은 모델 사용
--model-path liuhaotian/llava-v1.5-7b  # 대신 7b 사용
```

### 3. JSON 파싱 실패

```
⚠ JSON parsing failed: Invalid \escape
```

**해결**:
- 자동으로 fallback 모드로 전환됨
- VLM temperature 조정: `--temperature 0.1`

### 4. 데이터셋 경로 오류

```
FileNotFoundError: /path/to/utkinect not found
```

**해결**:
```bash
# 절대 경로 사용
--utkinect-root /absolute/path/to/utkinect

# 또는 환경 변수 설정
export UTKINECT_ROOT=/path/to/utkinect
```

---

## 💡 사용 예시

### 예시 1: 전체 테스트셋 평가

```bash
python inference_anticipation.py \
    --vlm-checkpoint ./checkpoints/best_model.pt \
    --futr-checkpoint ./checkpoints/futr_best.ckpt \
    --utkinect-root /data/utkinect \
    --num-inference-steps 500 \
    --output-dir ./results_full_test
```

### 예시 2: 빠른 검증 (10 샘플)

```bash
python inference_anticipation.py \
    --vlm-checkpoint ./checkpoints/vlm_epoch_5.pt \
    --futr-checkpoint ./checkpoints/futr_epoch_5.ckpt \
    --utkinect-root /data/utkinect \
    --num-inference-steps 10 \
    --output-dir ./results_quick_test
```

### 예시 3: 다른 split 평가

```bash
# Train split으로 overfitting 확인
python inference_anticipation.py \
    --vlm-checkpoint ./checkpoints/vlm_final.pt \
    --futr-checkpoint ./checkpoints/futr_final.ckpt \
    --utkinect-root /data/utkinect \
    --utkinect-split train \
    --num-inference-steps 100 \
    --output-dir ./results_train_split
```

---

## 📊 결과 분석

### Python으로 결과 분석

```python
import json
import numpy as np

# Load results
with open('inference_results/inference_results.json', 'r') as f:
    results = json.load(f)

# Per-class analysis
from collections import defaultdict
class_moc = defaultdict(list)

for r in results:
    for pred, target in zip(r['predicted_future'], r['target_future']):
        class_moc[target].append(1 if pred == target else 0)

# Print per-class accuracy
for cls, scores in class_moc.items():
    print(f"{cls}: {np.mean(scores):.4f}")
```

### 시각화

```python
import matplotlib.pyplot as plt

# MoC over time
moc_scores = [r['moc_score'] for r in results]
plt.plot(moc_scores)
plt.xlabel('Sample')
plt.ylabel('MoC Score')
plt.title('MoC Score over Samples')
plt.savefig('moc_over_time.png')
```

---

## 🔄 다음 단계

1. **앙상블**: 여러 체크포인트 결과 평균
2. **Beam Search**: 더 나은 생성 품질
3. **Post-processing**: 시간적 일관성 강제
4. **Visualization**: 예측 결과 시각화

---

## 📚 참고

- 학습 가이드: `README.md`
- 4-Segment 구현: `4_SEGMENT_IMPLEMENTATION.md`
- MoC 메트릭: `MOC_IMPLEMENTATION_SUMMARY.md`
- 문제 해결: `CUDA_VERSION_FIX.md`
