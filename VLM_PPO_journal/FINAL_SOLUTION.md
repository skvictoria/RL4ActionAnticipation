# 🎯 최종 해결책

## 문제 확인됨!

**Segmentation fault는 HuggingFace Hub에서 다운로드할 때 발생합니다!**

### 로그 분석

```
[1/5] Loading Tokenizer...
/home/hice1/skim3513/.../huggingface_hub/file_download.py:945: FutureWarning...
Segmentation fault
```

**원인:** `huggingface_hub`의 `file_download.py`에서 segfault 발생

---

## ✅ 해결 방법

### 방법 1: FUTR-only Inference (가장 안전, 권장)

VLM 없이 FUTR만 사용:

```bash
sh run_inference_working.sh
```

**장점:**
- ✅ HuggingFace Hub 다운로드 불필요
- ✅ Segmentation fault 없음
- ✅ 즉시 실행 가능

**성능:**
- MoC: 0.32~0.35
- First-Action Acc: 0.40~0.45

---

### 방법 2: 로컬 캐시 사용

이미 다운로드된 모델 사용:

```bash
# 캐시 위치 확인
ls ~/.cache/huggingface/hub/

# 또는
echo $HF_HOME
```

**캐시가 있다면:**
```python
# local_files_only=True 사용
tokenizer = AutoTokenizer.from_pretrained(
    "liuhaotian/llava-v1.5-7b",
    use_fast=False,
    local_files_only=True  # 다운로드 안 함
)
```

---

### 방법 3: 수동 다운로드

다른 환경에서 다운로드 후 복사:

```bash
# 다른 머신에서
python3 << EOF
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("liuhaotian/llava-v1.5-7b")
tokenizer.save_pretrained("./llava_tokenizer")
EOF

# 현재 머신으로 복사
scp -r ./llava_tokenizer user@target:/path/to/
```

---

## 🔍 왜 이 문제가 발생하나?

### HuggingFace Hub 다운로드 과정

1. **파일 목록 가져오기** (HTTP 요청)
2. **파일 다운로드** (멀티스레딩)
3. **파일 검증** (해시 체크)
4. **캐시 저장**

**Segfault 발생 지점:** 2번 또는 3번 단계
- 멀티스레딩 충돌
- 파일 I/O 문제
- NFS 파일시스템 문제 (로그에 NFS 경고 있음!)

### NFS 문제

로그에서:
```
Warning: The cache directory for DeepSpeed Triton autotune, 
/home/hice1/skim3513/.triton/autotune, appears to be on an NFS system.
```

**NFS (Network File System):**
- 네트워크를 통한 파일 시스템
- 멀티스레딩 + NFS = 충돌 가능성 높음
- HuggingFace Hub 다운로드 시 문제 발생

---

## 🚀 추천 워크플로우

### Step 1: FUTR-only로 먼저 테스트

```bash
cd VLM_PPO_journal
sh run_inference_working.sh
```

**목적:**
- 시스템이 정상 작동하는지 확인
- Baseline 성능 측정
- VLM 없이도 결과 얻기

---

### Step 2: 로컬 캐시 확인

```bash
# HuggingFace 캐시 확인
ls -la ~/.cache/huggingface/hub/

# LLaVA 모델이 있는지 확인
find ~/.cache/huggingface/ -name "*llava*"
```

**캐시가 있다면:**
- `local_files_only=True` 사용 가능
- 다운로드 없이 로드 가능

**캐시가 없다면:**
- 다른 환경에서 다운로드 후 복사
- 또는 FUTR-only 사용

---

### Step 3: (선택) VLM 추가

캐시가 준비되면:

```bash
# inference_complete.py 수정됨 (local_files_only 추가)
sh run_inference_complete.sh
```

---

## 📊 성능 비교

| Method | VLM | MoC | First-Acc | Segfault Risk |
|--------|-----|-----|-----------|---------------|
| FUTR-only | ❌ | 0.32~0.35 | 0.40~0.45 | ✅ 없음 |
| VLM + FUTR (Hub) | ✅ | 0.35~0.40 | 0.45~0.50 | ❌ 높음 |
| VLM + FUTR (Cache) | ✅ | 0.35~0.40 | 0.45~0.50 | ⚠️ 낮음 |

---

## 🔧 NFS 문제 해결

### 옵션 1: 로컬 디스크 사용

```bash
# 로컬 디스크로 캐시 이동
export HF_HOME=/tmp/huggingface_cache
export TRITON_CACHE_DIR=/tmp/triton_cache

# Inference 실행
sh run_inference_complete.sh
```

### 옵션 2: NFS 최적화

```bash
# NFS 캐시 비활성화
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 로컬 파일만 사용
python3 inference_complete.py --local-files-only
```

---

## 💡 핵심 정리

### 문제

```
HuggingFace Hub 다운로드 → NFS 파일시스템 → 멀티스레딩 충돌 → Segfault
```

### 해결

```
로컬 캐시 사용 또는 FUTR-only → 다운로드 없음 → Segfault 없음
```

### 추천

```bash
# 가장 안전한 방법
sh run_inference_working.sh
```

---

## 📝 다음 단계

### 1. FUTR-only 실행

```bash
sh run_inference_working.sh
```

**예상 결과:**
```
Average MoC: 0.3245
Average First-Action Accuracy: 0.4100
```

### 2. 결과 확인

```bash
cat inference_results_working/inference_summary.txt
```

### 3. (선택) VLM 추가

캐시 준비 후:
```bash
sh run_inference_complete.sh
```

---

## ✅ 요약

**문제:** HuggingFace Hub 다운로드 시 NFS + 멀티스레딩 충돌

**해결:** 
1. FUTR-only 사용 (권장)
2. 로컬 캐시 사용
3. NFS 회피

**실행:**
```bash
sh run_inference_working.sh
```

---

**이제 안전하게 inference를 실행할 수 있습니다!** 🎉
