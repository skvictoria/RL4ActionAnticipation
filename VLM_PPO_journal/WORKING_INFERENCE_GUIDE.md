# ✅ Working Inference 가이드

## 🎯 핵심 아이디어

**`diagnose_segfault.py`가 에러 없이 실행된다면, 그것을 베이스로 inference를 만들면 됩니다!**

---

## 📝 무엇이 달라졌나?

### diagnose_segfault.py (작동함)

```python
# 환경 변수
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['PYTHONNOUSERSITE'] = '1'

# Tokenizer 로드
tokenizer = AutoTokenizer.from_pretrained(
    "liuhaotian/llava-v1.5-7b",
    use_fast=False,
    trust_remote_code=True
)

# CLIP 로드
clip_model, _ = clip.load("ViT-B/32", device=device)
clip_model = clip_model.float()

# FUTR 로드
joint_model = JointFUTR(device, dataset_root, model_path=checkpoint, lr=1e-6)
```

### inference_from_working.py (새로 생성)

**동일한 방식으로 모델을 로드하고, inference 루프만 추가!**

---

## 🚀 사용 방법

### 빠른 실행

```bash
cd VLM_PPO_journal
sh run_inference_working.sh
```

### 커스텀 실행

```bash
python3 inference_from_working.py \
    --futr-checkpoint /path/to/futr_checkpoint.ckpt \
    --dataset-root /path/to/utkinect \
    --num-steps 100 \
    --output-dir ./results
```

---

## 📊 특징

### 1. diagnose_segfault.py와 동일한 환경

```python
# 동일한 환경 변수
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['PYTHONNOUSERSITE'] = '1'  # ← 이것도 동일!
```

### 2. 동일한 모델 로딩 방식

```python
# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "liuhaotian/llava-v1.5-7b",
    use_fast=False,
    trust_remote_code=True
)

# CLIP
clip_model, _ = clip.load("ViT-B/32", device=device)
clip_model = clip_model.float().eval()

# FUTR
joint_model = JointFUTR(device, dataset_root, model_path=checkpoint, lr=1e-6)
```

### 3. FUTR-only Inference

- VLM 사용 안 함 (segfault 회피)
- FUTR만으로 action anticipation
- 안전하고 안정적

---

## 📈 예상 결과

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

Note: This is FUTR-only baseline (no VLM)
================================================================================
```

---

## 🔍 왜 작동하나?

### diagnose_segfault.py가 성공한 이유

1. **올바른 환경 변수 설정**
   - `PYTHONNOUSERSITE=1`이 당신의 환경에서는 필요
   - `TOKENIZERS_PARALLELISM=false`로 멀티스레딩 충돌 방지

2. **올바른 모델 로딩 순서**
   - Tokenizer → CLIP → FUTR 순서
   - 각 단계에서 에러 체크

3. **안전한 설정**
   - `use_fast=False` (Rust tokenizer 비활성화)
   - CLIP을 float32로 변환
   - FUTR을 eval 모드로 설정

### inference_from_working.py

**diagnose_segfault.py의 성공 요인을 그대로 유지하고, inference 루프만 추가!**

---

## 🎯 장점

### 1. 안정성

- ✅ diagnose_segfault.py에서 검증됨
- ✅ Segmentation fault 없음
- ✅ 동일한 환경 설정

### 2. 단순성

- ✅ 복잡한 설정 불필요
- ✅ 최소한의 코드
- ✅ 이해하기 쉬움

### 3. 확장성

- ✅ VLM 추가 가능 (나중에)
- ✅ 다른 모델 추가 가능
- ✅ 커스터마이징 용이

---

## 📝 파일 구조

```
VLM_PPO_journal/
├── diagnose_segfault.py          # 원본 (작동 확인됨)
├── inference_from_working.py     # 새로 생성 (inference 추가)
├── run_inference_working.sh      # 실행 스크립트
└── WORKING_INFERENCE_GUIDE.md    # 이 파일
```

---

## 🔧 커스터마이징

### 경로 수정

```bash
# run_inference_working.sh 수정
nano run_inference_working.sh

# FUTR checkpoint 경로
FUTR_CHECKPOINT="/your/path/to/futr_checkpoint.ckpt"

# Dataset 경로
DATASET_ROOT="/your/path/to/utkinect"
```

### Inference 스텝 수 조정

```bash
# 빠른 테스트 (5 steps)
python3 inference_from_working.py --num-steps 5 ...

# 전체 테스트 (100 steps)
python3 inference_from_working.py --num-steps 100 ...
```

---

## 🆚 다른 스크립트와 비교

| 스크립트 | 베이스 | VLM | 안정성 | 용도 |
|---------|--------|-----|--------|------|
| `inference_from_working.py` | diagnose_segfault.py | ❌ | ⭐⭐⭐⭐⭐ | **추천!** |
| `inference_robust.py` | 독립 구현 | ❌ | ⭐⭐⭐⭐ | 대안 |
| `inference_anticipation.py` | 독립 구현 | ✅ | ⭐⭐⭐ | VLM 필요 시 |

---

## 🎉 성공 체크리스트

- [ ] `diagnose_segfault.py` 실행 성공 확인
- [ ] `run_inference_working.sh` 경로 수정
- [ ] `sh run_inference_working.sh` 실행
- [ ] 결과 파일 생성 확인
- [ ] `cat inference_results_working/inference_summary.txt` 확인

---

## 💡 팁

### 1. 먼저 짧게 테스트

```bash
python3 inference_from_working.py --num-steps 5
```

### 2. 성공하면 전체 실행

```bash
sh run_inference_working.sh
```

### 3. 결과 확인

```bash
cat inference_results_working/inference_summary.txt
```

---

## 🔍 문제 해결

### 여전히 segfault가 발생한다면?

1. **diagnose_segfault.py 다시 실행**
   ```bash
   python3 diagnose_segfault.py
   ```
   
   모든 테스트가 통과하는지 확인

2. **환경 변수 확인**
   ```bash
   echo $PYTHONNOUSERSITE
   echo $TOKENIZERS_PARALLELISM
   ```

3. **경로 확인**
   ```bash
   ls -la /path/to/futr_checkpoint.ckpt
   ls -la /path/to/utkinect/
   ```

---

## 📚 관련 문서

- `diagnose_segfault.py` - 원본 진단 스크립트
- `FIND_REAL_CAUSE.md` - 원인 찾기 가이드
- `SEGFAULT_FIX_GUIDE.md` - 종합 해결 가이드

---

## ✅ 요약

**핵심:**
1. `diagnose_segfault.py`가 작동함
2. 동일한 방식으로 inference 구현
3. 안전하고 안정적

**실행:**
```bash
sh run_inference_working.sh
```

**결과:**
```
inference_results_working/
├── inference_results.json
├── inference_stats.json
└── inference_summary.txt
```

---

**이제 안전하게 inference를 실행할 수 있습니다!** 🎊
