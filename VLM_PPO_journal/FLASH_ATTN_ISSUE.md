# 🎯 진짜 원인: Flash Attention 버전 불일치

## 실제 문제 발견!

**Segmentation fault의 진짜 원인은 `flash_attn` 라이브러리였습니다!**

### 에러 메시지

```
RuntimeError: Failed to import transformers.models.llama.modeling_llama because of the following error:
/home/hice1/skim3513/.local/lib/python3.10/site-packages/flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so: 
undefined symbol: _ZNK3c105Error4whatEv
```

### 의미

- `flash_attn_2_cuda.so` (C++ 확장 모듈)이 PyTorch와 ABI 불일치
- `_ZNK3c105Error4whatEv`는 C++ 심볼 (PyTorch의 Error 클래스)
- Flash Attention이 다른 버전의 PyTorch로 컴파일됨
- **이것이 segmentation fault를 유발!**

---

## PYTHONNOUSERSITE=1의 역할

### 왜 시도했나?

**의도:**
- `~/.local/lib/python3.10/site-packages/flash_attn`을 무시
- System site-packages의 패키지만 사용

### 왜 실패했나?

**문제:**
1. Flash Attention은 여전히 로드됨 (import 순서 때문)
2. 다른 필수 패키지들이 누락됨
3. 더 많은 에러 발생

**결론:**
- PYTHONNOUSERSITE=1은 임시방편일 뿐
- 근본 원인(flash_attn)을 해결해야 함

---

## 해결 방법

### 방법 1: Flash Attention 완전히 비활성화 (권장)

코드에서 xformers를 사용하지 않도록 수정:

#### 1-1. main.py 수정

```python
# 기존 코드 (문제 있음)
from patch import replace_llama_attn_with_xformers_attn
if replace_llama_attn_with_xformers_attn():
    print("using xformers")
else:
    print("using native attention")

# 수정된 코드 (안전함)
try:
    from patch import replace_llama_attn_with_xformers_attn
    if replace_llama_attn_with_xformers_attn():
        print("using xformers")
    else:
        print("using native attention")
except Exception as e:
    print(f"xformers not available, using native attention: {e}")
```

#### 1-2. 또는 완전히 비활성화

```python
# patch import를 주석 처리
# from patch import replace_llama_attn_with_xformers_attn
# if replace_llama_attn_with_xformers_attn():
#     print("using xformers")
# else:
#     print("using native attention")

print("using native attention (xformers disabled)")
```

---

### 방법 2: Flash Attention 재설치

현재 PyTorch 버전에 맞게 flash_attn 재설치:

```bash
# 1. 현재 PyTorch 버전 확인
python3 -c "import torch; print(torch.__version__)"
# 예: 2.0.1+cu121

# 2. 기존 flash_attn 제거
pip uninstall flash-attn -y

# 3. 현재 PyTorch에 맞게 재설치
pip install flash-attn --no-build-isolation

# 또는 특정 버전
pip install flash-attn==2.3.0 --no-build-isolation
```

**주의:** 컴파일에 시간이 오래 걸림 (10-30분)

---

### 방법 3: Flash Attention 제거 (가장 빠름)

Flash Attention을 완전히 제거하고 native attention 사용:

```bash
# Flash Attention 제거
pip uninstall flash-attn -y

# Inference 실행
python3 inference_anticipation.py ...
```

**장점:**
- 즉시 해결
- 호환성 문제 없음
- Inference 속도는 거의 동일

**단점:**
- Training 시 약간 느려질 수 있음 (inference는 영향 없음)

---

## 추천 해결 순서

### Step 1: Flash Attention 제거 (가장 빠름)

```bash
pip uninstall flash-attn -y
```

### Step 2: Inference 테스트

```bash
python3 diagnose_segfault.py
```

### Step 3: 성공하면 Inference 실행

```bash
sh run_inference_robust.sh
```

---

## 코드 수정 (안전한 버전)

### inference_robust.py 수정

```python
import os
import sys

# Segmentation fault 방지 환경 변수
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

# Flash Attention 비활성화 (중요!)
os.environ['DISABLE_FLASH_ATTN'] = '1'

# PYTHONNOUSERSITE 제거
if 'PYTHONNOUSERSITE' in os.environ:
    print("⚠ WARNING: PYTHONNOUSERSITE detected! Removing...")
    del os.environ['PYTHONNOUSERSITE']

# Add project paths
sys.path.insert(0, os.path.dirname(__file__))

# ... 나머지 코드
```

---

## 진단 방법

### 1. Flash Attention 설치 확인

```bash
python3 -c "import flash_attn; print(flash_attn.__version__)"
```

**결과:**
- 설치됨: 버전 출력 (예: 2.3.0)
- 미설치: ImportError

---

### 2. PyTorch 버전 확인

```bash
python3 -c "import torch; print(torch.__version__); print(torch.version.cuda)"
```

**예상 출력:**
```
2.0.1+cu121
12.1
```

---

### 3. Flash Attention과 PyTorch 호환성 확인

```bash
python3 << EOF
import torch
try:
    import flash_attn
    print(f"PyTorch: {torch.__version__}")
    print(f"Flash Attn: {flash_attn.__version__}")
    print("✓ Flash Attention loaded successfully")
except Exception as e:
    print(f"✗ Flash Attention error: {e}")
EOF
```

---

## 왜 이 문제가 발생했나?

### 시나리오

1. **PyTorch 업데이트**
   - PyTorch 2.0.0 → 2.0.1로 업데이트
   - ABI 변경

2. **Flash Attention은 그대로**
   - 이전 PyTorch 버전으로 컴파일된 flash_attn 사용
   - C++ 심볼 불일치

3. **결과**
   - Import 시 segmentation fault
   - 또는 undefined symbol 에러

---

## 장기 해결책

### 옵션 1: Flash Attention 사용 안 함

```python
# main.py
# from patch import replace_llama_attn_with_xformers_attn
# ... (주석 처리)

print("using native attention")
```

**장점:**
- 안정적
- 호환성 문제 없음

**단점:**
- Training 시 약간 느림 (inference는 영향 없음)

---

### 옵션 2: 가상환경 사용

```bash
# 새 가상환경 생성
python3 -m venv ~/inference_env

# 활성화
source ~/inference_env/bin/activate

# 패키지 설치 (flash_attn 제외)
pip install torch transformers tokenizers clip

# Inference 실행
python3 inference_anticipation.py ...
```

---

### 옵션 3: Docker 사용

```dockerfile
FROM pytorch/pytorch:2.0.1-cuda12.1-cudnn8-runtime

RUN pip install transformers tokenizers clip
# flash_attn 설치 안 함

COPY . /workspace
WORKDIR /workspace

CMD ["python3", "inference_anticipation.py"]
```

---

## 요약

### 문제

```
flash_attn_2_cuda.so: undefined symbol: _ZNK3c105Error4whatEv
→ Segmentation fault
```

### 원인

- Flash Attention과 PyTorch 버전 불일치
- C++ ABI 호환성 문제

### 해결 (빠른 순서)

1. **Flash Attention 제거** (가장 빠름)
   ```bash
   pip uninstall flash-attn -y
   ```

2. **코드 수정** (안전함)
   ```python
   # main.py에서 patch import 주석 처리
   ```

3. **재설치** (시간 걸림)
   ```bash
   pip install flash-attn --no-build-isolation
   ```

---

## 다음 단계

```bash
# 1. Flash Attention 제거
pip uninstall flash-attn -y

# 2. 진단
python3 diagnose_segfault.py

# 3. Inference
sh run_inference_robust.sh
```

---

**핵심:** Flash Attention이 문제였고, PYTHONNOUSERSITE=1은 임시방편이었습니다!
