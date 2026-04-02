# 🔍 Segmentation Fault 진짜 원인 찾기

## 당신의 관찰이 맞을 수 있습니다!

Flash Attention이 문제가 아닐 수도 있습니다. 정확한 원인을 찾아봅시다.

---

## 🧪 단계별 테스트

### Step 1: 시나리오 테스트 실행

```bash
cd VLM_PPO_journal
python3 test_segfault_scenarios.py
```

이 스크립트는 각 단계를 개별적으로 테스트합니다:

1. ✓ Basic imports (torch, transformers, clip)
2. ✓ Tokenizer loading
3. ✓ LLaVA imports
4. ✓ Base model loading
5. ✓ LoRA loading
6. ✓ Forward pass

**어느 단계에서 실패하는지 확인하세요!**

---

## 📊 가능한 원인들

### 원인 1: Tokenizers 라이브러리 (Rust 확장)

**증상:**
- Test 2 (Tokenizer loading)에서 실패
- `tokenizers` import 시 segfault

**확인:**
```bash
python3 -c "from transformers import AutoTokenizer; t = AutoTokenizer.from_pretrained('liuhaotian/llava-v1.5-7b', use_fast=False)"
```

**해결:**
```bash
pip uninstall tokenizers -y
pip install tokenizers==0.15.0
```

---

### 원인 2: Transformers 버전 불일치

**증상:**
- Test 3 (LLaVA imports)에서 실패
- `from llava.model import ...` 시 segfault

**확인:**
```bash
python3 -c "from llava.model import LlavaLlamaForCausalLM"
```

**해결:**
```bash
pip install transformers==4.36.0
```

---

### 원인 3: CUDA/PyTorch 호환성

**증상:**
- Test 4 (Base model loading)에서 실패
- Model을 GPU로 로드할 때 segfault

**확인:**
```bash
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```

**해결:**
```bash
# PyTorch 재설치
pip install torch==2.0.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

---

### 원인 4: LoRA Checkpoint 문제

**증상:**
- Test 5 (LoRA loading)에서 실패
- PEFT 로드 시 segfault

**확인:**
```bash
ls -la vlm_checkpoints/epoch_4/
cat vlm_checkpoints/epoch_4/adapter_config.json
```

**해결:**
- 다른 epoch checkpoint 시도
- Checkpoint 재생성

---

### 원인 5: Flash Attention

**증상:**
- Test 3에서 특정 에러 메시지
- `flash_attn_2_cuda.so: undefined symbol`

**확인:**
```bash
python3 -c "import flash_attn; print(flash_attn.__version__)"
```

**해결:**
```bash
pip uninstall flash-attn -y
```

---

## 🎯 정확한 진단 방법

### 방법 1: 상세 에러 로그

```bash
# 상세 에러 출력
python3 -u test_segfault_scenarios.py 2>&1 | tee segfault_test.log

# 로그 확인
cat segfault_test.log
```

---

### 방법 2: GDB로 디버깅

```bash
# GDB 설치 (필요시)
# sudo apt-get install gdb

# GDB로 실행
gdb python3
(gdb) run test_segfault_scenarios.py
# Segfault 발생 시
(gdb) backtrace
```

**Backtrace에서 확인할 것:**
- 어떤 라이브러리에서 crash?
- 어떤 함수 호출 중?
- C++ 심볼 이름?

---

### 방법 3: Strace로 시스템 콜 추적

```bash
strace -o segfault_trace.txt python3 test_segfault_scenarios.py

# 마지막 부분 확인
tail -100 segfault_trace.txt
```

---

## 📝 정보 수집

### 1. 환경 정보

```bash
python3 << EOF
import sys
import torch
import transformers
import tokenizers

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
print(f"Transformers: {transformers.__version__}")
print(f"Tokenizers: {tokenizers.__version__}")

try:
    import flash_attn
    print(f"Flash Attn: {flash_attn.__version__}")
except:
    print("Flash Attn: Not installed")

try:
    import peft
    print(f"PEFT: {peft.__version__}")
except:
    print("PEFT: Not installed")
EOF
```

---

### 2. 실제 에러 메시지

```bash
# Inference 실행하여 정확한 에러 확인
python3 inference_anticipation.py \
    --vlm-checkpoint ./vlm_checkpoints/epoch_4 \
    --futr-checkpoint /path/to/futr.ckpt \
    --utkinect-root /path/to/utkinect \
    --num-inference-steps 1 \
    2>&1 | tee inference_error.log
```

**에러 로그에서 확인:**
- Segmentation fault 직전 메시지
- Import 에러
- 라이브러리 로딩 메시지

---

### 3. 패키지 설치 위치

```bash
python3 << EOF
import transformers
import tokenizers
import torch

print(f"Transformers: {transformers.__file__}")
print(f"Tokenizers: {tokenizers.__file__}")
print(f"Torch: {torch.__file__}")

try:
    import flash_attn
    print(f"Flash Attn: {flash_attn.__file__}")
except:
    print("Flash Attn: Not installed")
EOF
```

**확인할 것:**
- 모두 같은 환경에 설치되어 있나?
- User site-packages vs System site-packages
- 가상환경 사용 중인가?

---

## 🔬 실험적 테스트

### 테스트 A: 최소 환경

```bash
# 새 가상환경
python3 -m venv ~/test_env
source ~/test_env/bin/activate

# 최소 패키지만 설치
pip install torch transformers tokenizers

# 테스트
python3 -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('liuhaotian/llava-v1.5-7b', use_fast=False)"
```

**성공하면:** 기존 환경의 패키지 충돌 문제
**실패하면:** 시스템 레벨 문제

---

### 테스트 B: Fast vs Slow Tokenizer

```bash
# Fast tokenizer (Rust)
python3 -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('liuhaotian/llava-v1.5-7b', use_fast=True)"

# Slow tokenizer (Python)
python3 -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('liuhaotian/llava-v1.5-7b', use_fast=False)"
```

**Fast 실패, Slow 성공:** Rust 확장 문제
**둘 다 실패:** Transformers 자체 문제

---

### 테스트 C: PYTHONNOUSERSITE 효과

```bash
# PYTHONNOUSERSITE 없이
python3 test_segfault_scenarios.py

# PYTHONNOUSERSITE=1로
export PYTHONNOUSERSITE=1
python3 test_segfault_scenarios.py
unset PYTHONNOUSERSITE
```

**차이가 있으면:** User site-packages의 특정 패키지 문제
**차이가 없으면:** PYTHONNOUSERSITE 무관

---

## 📋 체크리스트

실제 원인을 찾기 위해 다음을 확인하세요:

- [ ] `test_segfault_scenarios.py` 실행 결과
- [ ] 어느 단계에서 실패하는가?
- [ ] 정확한 에러 메시지 (있다면)
- [ ] 환경 정보 (Python, PyTorch, CUDA 버전)
- [ ] 패키지 설치 위치
- [ ] PYTHONNOUSERSITE 효과
- [ ] Fast vs Slow tokenizer 차이
- [ ] 가상환경에서 테스트 결과

---

## 💬 결과 공유

테스트 결과를 공유해주세요:

```bash
# 1. 시나리오 테스트
python3 test_segfault_scenarios.py > test_results.txt 2>&1

# 2. 환경 정보
python3 -c "import torch, transformers, tokenizers; print(f'PyTorch: {torch.__version__}'); print(f'Transformers: {transformers.__version__}'); print(f'Tokenizers: {tokenizers.__version__}')" >> test_results.txt

# 3. 결과 확인
cat test_results.txt
```

**특히 중요한 정보:**
- 어느 테스트에서 실패?
- 에러 메시지 (있다면)
- Traceback 전체

---

## 🎯 다음 단계

### 테스트 결과에 따라

1. **Test 2 실패** → Tokenizers 문제
2. **Test 3 실패** → LLaVA/Transformers 문제
3. **Test 4 실패** → CUDA/Model loading 문제
4. **Test 5 실패** → LoRA checkpoint 문제
5. **모두 성공** → Inference 코드 자체 문제

---

## 🔧 임시 해결책

정확한 원인을 찾는 동안:

```bash
# FUTR-only로 테스트
sh run_inference_robust.sh
```

이것이 성공하면:
- FUTR는 문제없음
- VLM 로딩이 문제
- Tokenizer 또는 Model loading 관련

---

**핵심:** `test_segfault_scenarios.py`를 실행하여 정확히 어느 단계에서 문제가 발생하는지 확인하세요!
