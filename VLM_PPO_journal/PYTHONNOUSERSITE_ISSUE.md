# PYTHONNOUSERSITE 환경 변수 문제

## 🚨 중요 발견

`export PYTHONNOUSERSITE=1`이 segmentation fault의 원인일 수 있습니다!

---

## 문제 설명

### PYTHONNOUSERSITE란?

**용도:**
- Python이 user site-packages 디렉토리를 무시하도록 함
- `~/.local/lib/python3.x/site-packages/` 경로를 사용하지 않음

**일반적인 사용 이유:**
- 시스템 패키지와 사용자 패키지 충돌 방지
- 깨끗한 환경에서 테스트
- 특정 버전의 패키지만 사용

### 왜 Segmentation Fault를 유발하나?

**원인:**
1. **라이브러리 버전 불일치**
   - System site-packages의 transformers 버전
   - User site-packages의 tokenizers 버전
   - 두 버전이 호환되지 않을 때 C++ 라이브러리 충돌

2. **의존성 누락**
   - User site-packages에만 설치된 필수 의존성
   - PYTHONNOUSERSITE=1로 인해 접근 불가
   - 런타임에 라이브러리 로딩 실패 → segfault

3. **C++ 확장 모듈 충돌**
   - tokenizers는 Rust로 작성된 C++ 확장 모듈
   - 여러 경로에서 다른 버전 로드 시도
   - ABI 불일치 → segfault

---

## 해결 방법

### 방법 1: PYTHONNOUSERSITE 제거 (권장)

```bash
# 현재 세션에서 제거
unset PYTHONNOUSERSITE

# 또는
export -n PYTHONNOUSERSITE

# 확인
echo $PYTHONNOUSERSITE
# (아무것도 출력되지 않아야 함)

# Inference 실행
python3 inference_anticipation.py ...
```

---

### 방법 2: 환경 변수 확인 후 실행

```bash
# 진단 스크립트로 확인
python3 diagnose_segfault.py

# PYTHONNOUSERSITE가 설정되어 있으면 경고 표시됨
```

**출력 예시:**
```
================================================================================
6. Testing Environment Variables
================================================================================

Current environment variables:
    TOKENIZERS_PARALLELISM: false
    OMP_NUM_THREADS: 1
    MKL_NUM_THREADS: 1
    CUDA_LAUNCH_BLOCKING: Not set
    CUDA_VISIBLE_DEVICES: Not set
  ⚠ PYTHONNOUSERSITE: 1

================================================================================
⚠ WARNING: PYTHONNOUSERSITE is set!
================================================================================
This environment variable can cause segmentation faults with
certain Python packages, especially transformers and tokenizers.

Recommendation:
  1. Unset this variable: unset PYTHONNOUSERSITE
  2. Run inference without this variable
  3. If you must use it, test with: python3 diagnose_segfault.py --test-pythonnousersite
================================================================================
```

---

### 방법 3: PYTHONNOUSERSITE 테스트

```bash
# PYTHONNOUSERSITE=1로 테스트 (주의!)
python3 diagnose_segfault.py --test-pythonnousersite
```

**주의:** 이 옵션은 segfault를 재현하기 위한 것입니다.

---

## 영구적으로 제거

### .bashrc 또는 .bash_profile 확인

```bash
# 파일 확인
cat ~/.bashrc | grep PYTHONNOUSERSITE
cat ~/.bash_profile | grep PYTHONNOUSERSITE

# 발견되면 해당 줄 삭제
nano ~/.bashrc
# 또는
nano ~/.bash_profile

# 변경 사항 적용
source ~/.bashrc
```

---

### 스크립트에서 제거

```bash
# run.sh 또는 다른 실행 스크립트 확인
grep -r "PYTHONNOUSERSITE" *.sh

# 발견되면 해당 줄 삭제 또는 주석 처리
```

---

## 올바른 환경 변수 설정

### Inference 실행 시 권장 설정

```bash
# 필수 환경 변수
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# PYTHONNOUSERSITE는 설정하지 않음!
# export PYTHONNOUSERSITE=1  # ← 이것을 제거!

# Inference 실행
python3 inference_anticipation.py ...
```

---

## 패키지 설치 위치 확인

### 현재 설치된 패키지 위치 확인

```bash
# transformers 위치
python3 -c "import transformers; print(transformers.__file__)"

# tokenizers 위치
python3 -c "import tokenizers; print(tokenizers.__file__)"

# torch 위치
python3 -c "import torch; print(torch.__file__)"
```

**예상 출력:**
```
# User site-packages (정상)
/home/user/.local/lib/python3.10/site-packages/transformers/__init__.py
/home/user/.local/lib/python3.10/site-packages/tokenizers/__init__.py

# 또는 System site-packages (정상)
/usr/local/lib/python3.10/site-packages/transformers/__init__.py

# 또는 가상환경 (정상)
/home/user/venv/lib/python3.10/site-packages/transformers/__init__.py
```

---

### PYTHONNOUSERSITE=1일 때

```bash
export PYTHONNOUSERSITE=1
python3 -c "import transformers; print(transformers.__file__)"
```

**문제 발생 시:**
- User site-packages의 패키지를 찾지 못함
- 다른 버전의 패키지 로드
- 의존성 불일치 → segfault

---

## 대안: 가상환경 사용

PYTHONNOUSERSITE를 사용하는 이유가 패키지 격리라면, 가상환경을 사용하세요.

### 가상환경 생성

```bash
# venv 생성
python3 -m venv ~/inference_env

# 활성화
source ~/inference_env/bin/activate

# 필요한 패키지 설치
pip install torch transformers tokenizers clip

# Inference 실행 (PYTHONNOUSERSITE 불필요)
python3 inference_anticipation.py ...
```

**장점:**
- 완전히 격리된 환경
- PYTHONNOUSERSITE 불필요
- 패키지 버전 관리 용이
- Segfault 위험 감소

---

## 진단 및 테스트

### 1. 현재 환경 확인

```bash
# 환경 변수 확인
python3 diagnose_segfault.py
```

---

### 2. PYTHONNOUSERSITE 없이 테스트

```bash
# 제거
unset PYTHONNOUSERSITE

# 진단
python3 diagnose_segfault.py

# Inference
python3 inference_anticipation.py ...
```

---

### 3. PYTHONNOUSERSITE=1로 테스트 (재현용)

```bash
# 주의: segfault 재현 목적
python3 diagnose_segfault.py --test-pythonnousersite
```

---

## 업데이트된 실행 스크립트

### run_inference_robust.sh (수정됨)

```bash
#!/bin/bash

# Segmentation fault 방지 환경 변수
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_LAUNCH_BLOCKING=1

# PYTHONNOUSERSITE 제거 (중요!)
unset PYTHONNOUSERSITE

# 또는 명시적으로 0으로 설정
# export PYTHONNOUSERSITE=0

# Inference 실행
python3 inference_robust.py ...
```

---

## 자주 묻는 질문

### Q1: PYTHONNOUSERSITE를 꼭 사용해야 하나요?

**A:** 대부분의 경우 필요 없습니다.

**대안:**
- 가상환경 사용 (venv, conda)
- Docker 컨테이너 사용
- 특정 경로의 패키지만 사용: `PYTHONPATH` 설정

---

### Q2: PYTHONNOUSERSITE 없이도 패키지 격리가 가능한가요?

**A:** 네, 가상환경을 사용하세요.

```bash
# venv 사용
python3 -m venv myenv
source myenv/bin/activate

# conda 사용
conda create -n myenv python=3.10
conda activate myenv
```

---

### Q3: 시스템 관리자가 PYTHONNOUSERSITE를 설정했어요

**A:** 로컬 세션에서만 제거하세요.

```bash
# 현재 세션에서만 제거
unset PYTHONNOUSERSITE

# 또는 스크립트에서 명시적으로 제거
#!/bin/bash
unset PYTHONNOUSERSITE
python3 inference.py ...
```

---

## 요약

### 문제

```bash
export PYTHONNOUSERSITE=1
python3 inference.py
# → Segmentation fault (core dumped)
```

### 해결

```bash
unset PYTHONNOUSERSITE
python3 inference.py
# → 정상 작동
```

### 확인

```bash
# 진단 스크립트로 확인
python3 diagnose_segfault.py

# PYTHONNOUSERSITE가 설정되어 있으면 경고 표시
```

---

## 추가 리소스

- `diagnose_segfault.py`: 환경 변수 확인 및 진단
- `SEGFAULT_FIX_GUIDE.md`: 종합 해결 가이드
- `run_inference_robust.sh`: 올바른 환경 변수 설정 예시

---

## 핵심 포인트

1. **PYTHONNOUSERSITE=1은 segfault를 유발할 수 있음**
2. **대부분의 경우 이 변수는 필요 없음**
3. **패키지 격리가 필요하면 가상환경 사용**
4. **진단 스크립트로 환경 변수 확인**
5. **unset PYTHONNOUSERSITE로 제거**

---

**마지막 업데이트:** 2026-04-02
**발견:** PYTHONNOUSERSITE=1이 transformers/tokenizers와 충돌하여 segfault 유발
