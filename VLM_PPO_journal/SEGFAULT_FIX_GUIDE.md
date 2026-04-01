# Segmentation Fault 해결 가이드

## 문제 상황

Inference 실행 시 segmentation fault가 발생하는 문제:
```
Segmentation fault (core dumped)
```

## 원인 분석

1. **Tokenizer 로딩 문제**
   - HuggingFace Hub에서 tokenizer를 로드할 때 C++ 라이브러리 충돌
   - Fast tokenizer (Rust 기반)와 Python 환경 간 호환성 문제
   - 멀티스레딩 관련 충돌

2. **메모리 관련 문제**
   - 모델 로딩 시 메모리 할당 오류
   - CUDA 메모리와 CPU 메모리 간 전송 문제

3. **라이브러리 버전 충돌**
   - Transformers, Tokenizers, PyTorch 버전 불일치
   - CUDA 버전 불일치

## 해결 방법

### 방법 1: Robust Inference Script 사용 (권장)

가장 안전한 방법으로, 여러 fallback 전략을 사용합니다.

```bash
cd VLM_PPO_journal
sh run_inference_robust.sh
```

**특징:**
- 5가지 tokenizer 로딩 전략 (순차적 fallback)
- FUTR-only inference (VLM 없이 실행)
- 환경 변수 자동 설정
- 상세한 에러 로깅

**Tokenizer 로딩 전략:**
1. Local checkpoint (config.json 있는 경우)
2. LoRA checkpoint의 base model
3. HuggingFace Hub (liuhaotian/llava-v1.5-7b)
4. Llama tokenizer (huggyllama/llama-7b)
5. Llama-2 tokenizer (meta-llama/Llama-2-7b-hf)

### 방법 2: 환경 변수 설정

Segmentation fault를 방지하는 환경 변수를 설정합니다.

```bash
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_LAUNCH_BLOCKING=1
```

이후 inference 실행:
```bash
python3 inference_robust.py \
    --futr-checkpoint <FUTR_CHECKPOINT> \
    --utkinect-root <DATASET_PATH> \
    --num-inference-steps 100
```

### 방법 3: Tokenizer 수동 다운로드

HuggingFace Hub 접근 문제가 있는 경우, tokenizer를 미리 다운로드합니다.

```python
from transformers import AutoTokenizer

# 미리 다운로드
tokenizer = AutoTokenizer.from_pretrained(
    "liuhaotian/llava-v1.5-7b",
    use_fast=False,
    cache_dir="./tokenizer_cache"
)
tokenizer.save_pretrained("./local_tokenizer")
```

이후 inference 실행 시:
```bash
python3 inference_robust.py \
    --model-path ./local_tokenizer \
    --futr-checkpoint <FUTR_CHECKPOINT> \
    --utkinect-root <DATASET_PATH>
```

### 방법 4: VLM 체크포인트에서 Tokenizer 로드

학습된 VLM 체크포인트에 tokenizer가 포함되어 있는 경우:

```bash
python3 inference_robust.py \
    --model-path ./vlm_checkpoints/epoch_4 \
    --futr-checkpoint <FUTR_CHECKPOINT> \
    --utkinect-root <DATASET_PATH>
```

**주의:** `vlm_checkpoints/epoch_4` 디렉토리에 다음 파일들이 있어야 합니다:
- `config.json`
- `tokenizer_config.json`
- `tokenizer.model` 또는 `vocab.json`

## 체크포인트 구조 확인

VLM 체크포인트가 올바르게 저장되었는지 확인:

```bash
ls -la vlm_checkpoints/epoch_4/
```

필요한 파일들:
```
config.json              # 모델 설정
adapter_config.json      # LoRA 설정
adapter_model.bin        # LoRA 가중치
tokenizer_config.json    # Tokenizer 설정
tokenizer.model          # Tokenizer 모델
special_tokens_map.json  # 특수 토큰 매핑
```

만약 tokenizer 파일들이 없다면, 학습 시 저장 코드를 수정해야 합니다.

## 디버깅 방법

### 1. Tokenizer 로딩 테스트

```python
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from transformers import AutoTokenizer

# 테스트 1: HuggingFace Hub
try:
    tokenizer = AutoTokenizer.from_pretrained(
        "liuhaotian/llava-v1.5-7b",
        use_fast=False
    )
    print("✓ HuggingFace Hub tokenizer works")
except Exception as e:
    print(f"✗ HuggingFace Hub failed: {e}")

# 테스트 2: Local checkpoint
try:
    tokenizer = AutoTokenizer.from_pretrained(
        "./vlm_checkpoints/epoch_4",
        use_fast=False,
        local_files_only=True
    )
    print("✓ Local checkpoint tokenizer works")
except Exception as e:
    print(f"✗ Local checkpoint failed: {e}")
```

### 2. FUTR 로딩 테스트

```python
import torch
from joint_model import JointFUTR

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset_root = "/path/to/utkinect"
checkpoint = "/path/to/futr_checkpoint.ckpt"

try:
    joint_model = JointFUTR(device, dataset_root, model_path=checkpoint, lr=1e-6)
    print("✓ FUTR model loaded successfully")
except Exception as e:
    print(f"✗ FUTR loading failed: {e}")
```

### 3. 전체 파이프라인 테스트

```bash
# 최소한의 스텝으로 테스트
python3 inference_robust.py \
    --futr-checkpoint <CHECKPOINT> \
    --utkinect-root <DATASET> \
    --num-inference-steps 5 \
    --output-dir ./test_output
```

## 일반적인 에러와 해결책

### 에러 1: "Segmentation fault" (즉시 종료)

**원인:** Tokenizer 로딩 시 C++ 라이브러리 충돌

**해결:**
```bash
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
python3 inference_robust.py ...
```

### 에러 2: "Cannot load tokenizer"

**원인:** HuggingFace Hub 접근 불가 또는 로컬 파일 없음

**해결:**
1. 인터넷 연결 확인
2. HuggingFace 토큰 설정 (필요한 경우)
3. Tokenizer 수동 다운로드 (방법 3 참조)

### 에러 3: "CUDA out of memory"

**원인:** GPU 메모리 부족

**해결:**
```bash
# 배치 크기 줄이기
python3 inference_robust.py \
    --num-inference-steps 50 \
    ...
```

### 에러 4: "FileNotFoundError: config.json"

**원인:** VLM 체크포인트가 불완전하게 저장됨

**해결:**
1. 학습 시 `tokenizer.save_pretrained()` 호출 확인
2. Base model path 사용: `--model-path liuhaotian/llava-v1.5-7b`

## 추천 워크플로우

### 1단계: FUTR-only Baseline

VLM 없이 FUTR만으로 inference (가장 안전):

```bash
sh run_inference_robust.sh
```

### 2단계: VLM 체크포인트 확인

체크포인트 구조 확인:
```bash
ls -la vlm_checkpoints/epoch_4/
```

### 3단계: Full Pipeline Inference

VLM + FUTR 전체 파이프라인 (tokenizer 문제 해결 후):

```bash
python3 inference_anticipation.py \
    --vlm-checkpoint ./vlm_checkpoints/epoch_4 \
    --futr-checkpoint <FUTR_CHECKPOINT> \
    --utkinect-root <DATASET> \
    --num-inference-steps 100
```

## 성능 비교

| Method | MoC | First-Action Acc | Notes |
|--------|-----|------------------|-------|
| FUTR-only | ~0.XX | ~0.XX | Baseline (no VLM) |
| VLM + FUTR | ~0.XX | ~0.XX | Full pipeline |

## 추가 리소스

- **inference_robust.py**: 가장 안전한 inference 스크립트
- **inference_safe.py**: FUTR-only baseline
- **inference_anticipation.py**: Full VLM + FUTR pipeline
- **CHECKPOINT_GUIDE.md**: 체크포인트 저장/로드 가이드

## 문제 해결이 안 될 때

1. **환경 정보 수집:**
```bash
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')"
python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python3 -c "import tokenizers; print(f'Tokenizers: {tokenizers.__version__}')"
```

2. **최소 재현 코드 작성:**
```python
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "liuhaotian/llava-v1.5-7b",
    use_fast=False
)
print("Success!")
```

3. **라이브러리 재설치:**
```bash
pip uninstall tokenizers transformers -y
pip install transformers==4.36.0 tokenizers==0.15.0
```

## 요약

**가장 빠른 해결책:**
```bash
cd VLM_PPO_journal
sh run_inference_robust.sh
```

이 스크립트는:
- ✓ 환경 변수 자동 설정
- ✓ 5가지 tokenizer 로딩 전략
- ✓ FUTR-only inference (VLM 없이)
- ✓ 상세한 에러 로깅
- ✓ Per-class 성능 분석

**문제가 계속되면:**
1. `inference_robust.py` 실행 로그 확인
2. Tokenizer 로딩 전략 중 어디서 실패하는지 확인
3. 해당 전략에 맞는 해결책 적용
