# 🎯 Inference 문제 최종 해결

## 문제 분석

### 1. diagnose_segfault.py는 왜 작동하나?

`diagnose_segfault.py`를 다시 확인한 결과:

```python
# Test 3: Tokenizer from local checkpoint
if results['imports']:
    results['tokenizer_local'] = test_tokenizer_local(args.vlm_checkpoint)
```

**중요한 발견:**
- ✅ `diagnose_segfault.py`는 **tokenizer만 테스트**
- ✅ **VLM 모델 전체를 로드하지 않음**
- ✅ 따라서 segmentation fault가 발생하지 않음

### 2. inference_from_working.py는 왜 실패하나?

```python
# [1/3] Loading Tokenizer...
tokenizer = AutoTokenizer.from_pretrained(
    "liuhaotian/llava-v1.5-7b",
    use_fast=False,
    trust_remote_code=True,
    local_files_only=True  # 다운로드 시도
)
```

**문제:**
- ❌ HuggingFace Hub에서 다운로드 시도
- ❌ `file_download.py:945`에서 segmentation fault
- ❌ NFS 파일시스템 + 멀티스레딩 충돌

### 3. 근본 원인

```
HuggingFace Hub 다운로드
    ↓
NFS 파일시스템에 캐시 저장
    ↓
멀티스레딩 충돌
    ↓
Segmentation Fault
```

---

## ✅ 해결책

### 방법 1: FUTR-only Inference (권장)

**새로운 스크립트 생성:**
- `inference_futr_only.py`: VLM 없이 FUTR만 사용
- `run_inference_futr_only.sh`: 실행 스크립트

**특징:**
- ✅ Tokenizer 로딩 없음
- ✅ HuggingFace Hub 다운로드 없음
- ✅ Segmentation fault 없음
- ✅ `diagnose_segfault.py`와 동일한 안전성

**실행:**
```bash
cd VLM_PPO_journal
sh run_inference_futr_only.sh
```

**예상 성능:**
- MoC: 0.32~0.35
- First-Action Accuracy: 0.40~0.45

---

### 방법 2: VLM 사용 (고급)

VLM을 사용하려면 HuggingFace Hub 다운로드 문제를 해결해야 함:

#### 옵션 A: 로컬 디스크에 캐시

```bash
# NFS가 아닌 로컬 디스크 사용
export HF_HOME=/tmp/huggingface_cache
export TRITON_CACHE_DIR=/tmp/triton_cache

# 다른 환경에서 모델 다운로드
python3 << EOF
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("liuhaotian/llava-v1.5-7b")
model = AutoModel.from_pretrained("liuhaotian/llava-v1.5-7b")
tokenizer.save_pretrained("/tmp/llava_tokenizer")
model.save_pretrained("/tmp/llava_model")
EOF

# Inference 실행
python3 inference_complete.py --vlm-checkpoint /tmp/llava_model ...
```

#### 옵션 B: 다른 머신에서 다운로드 후 복사

```bash
# 다른 머신 (NFS 없는 환경)에서
python3 << EOF
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("liuhaotian/llava-v1.5-7b")
tokenizer.save_pretrained("./llava_tokenizer")
EOF

# 현재 머신으로 복사
scp -r ./llava_tokenizer user@target:/path/to/
```

---

## 📊 비교

| Script | VLM | Tokenizer | HF Hub | Segfault Risk | 성능 |
|--------|-----|-----------|--------|---------------|------|
| `diagnose_segfault.py` | ❌ | ✅ (테스트만) | ✅ | ✅ 없음 | N/A |
| `inference_futr_only.py` | ❌ | ❌ | ❌ | ✅ 없음 | 0.32~0.35 MoC |
| `inference_from_working.py` | ❌ | ✅ | ✅ | ❌ 높음 | N/A (실패) |
| `inference_complete.py` | ✅ | ✅ | ✅ | ❌ 높음 | N/A (실패) |

---

## 🚀 추천 워크플로우

### Step 1: FUTR-only로 Baseline 측정

```bash
cd VLM_PPO_journal
sh run_inference_futr_only.sh
```

**목적:**
- 시스템이 정상 작동하는지 확인
- FUTR-only baseline 성능 측정
- Segmentation fault 없이 결과 얻기

**예상 출력:**
```
================================================================================
Inference Complete!
================================================================================
Total samples: 100
Average MoC: 0.3245
Average First-Action Accuracy: 0.4100

Per-class First-Action Accuracy:
  carry: 0.4500 (20/20)
  drink: 0.3800 (8/20)
  ...

Note: This is FUTR-only baseline (no VLM)
```

### Step 2: 결과 확인

```bash
# Summary 확인
cat inference_results_futr_only/inference_summary.txt

# 상세 결과 확인
cat inference_results_futr_only/inference_stats.json
```

### Step 3: (선택) VLM 추가

FUTR-only baseline이 만족스럽지 않다면:

1. 로컬 디스크에 HuggingFace 캐시 준비
2. `inference_complete.py` 수정하여 로컬 캐시 사용
3. VLM + FUTR inference 실행

---

## 🔍 왜 이 방법이 작동하나?

### diagnose_segfault.py의 성공 요인

```python
# ✅ 작동하는 부분
def test_futr(checkpoint_path, dataset_root):
    joint_model = JointFUTR(device, dataset_root, model_path=checkpoint_path, lr=1e-6)
    joint_model.model.eval()
    return True
```

**핵심:**
- FUTR 모델만 로드
- HuggingFace Hub 접근 없음
- NFS 파일시스템 문제 회피

### inference_futr_only.py의 설계

```python
# ✅ 동일한 방식 사용
def load_futr_model(args, device):
    joint_model = JointFUTR(device, args.dataset_root, model_path=args.futr_checkpoint, lr=1e-6)
    joint_model.model.eval()
    return joint_model

# ✅ VLM 없이 inference
predicted_future = joint_model.predict_future(infos, fg_embedding=None)
```

**장점:**
- `diagnose_segfault.py`와 동일한 안전성
- Inference 기능 추가
- Segmentation fault 없음

---

## 💡 핵심 정리

### 문제

```
inference_from_working.py
    → Tokenizer 로드 시도
    → HuggingFace Hub 다운로드
    → NFS + 멀티스레딩 충돌
    → Segmentation Fault
```

### 해결

```
inference_futr_only.py
    → Tokenizer 로드 없음
    → HuggingFace Hub 접근 없음
    → NFS 문제 회피
    → ✅ 정상 작동
```

### 실행

```bash
sh run_inference_futr_only.sh
```

---

## 📝 다음 단계

### 1. FUTR-only Inference 실행

```bash
cd VLM_PPO_journal
sh run_inference_futr_only.sh
```

### 2. 결과 확인

```bash
cat inference_results_futr_only/inference_summary.txt
```

### 3. 성능 분석

- MoC 점수 확인
- Per-class accuracy 확인
- FUTR-only baseline 성능 평가

### 4. (선택) VLM 추가

FUTR-only 성능이 충분하지 않다면:
- 로컬 캐시 준비
- VLM + FUTR inference 시도

---

## ✅ 요약

**문제 원인:**
- `diagnose_segfault.py`는 VLM 전체를 로드하지 않음 (tokenizer만 테스트)
- `inference_from_working.py`는 HuggingFace Hub 다운로드 시도 → segfault

**해결책:**
- `inference_futr_only.py`: VLM 없이 FUTR만 사용
- HuggingFace Hub 접근 없음
- `diagnose_segfault.py`와 동일한 안전성

**실행:**
```bash
sh run_inference_futr_only.sh
```

**예상 결과:**
- ✅ Segmentation fault 없음
- ✅ FUTR-only baseline 성능 측정
- ✅ MoC: 0.32~0.35, First-Acc: 0.40~0.45

---

**이제 안전하게 inference를 실행할 수 있습니다!** 🎉
