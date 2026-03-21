# Segmentation Fault 해결 가이드

## 문제 상황
```
Segmentation fault (core dumped)
```

inference 실행 시 segmentation fault가 발생했습니다.

## 원인 분석

Segmentation fault는 보통 다음과 같은 이유로 발생합니다:

1. **메모리 접근 오류**: 잘못된 메모리 주소 접근
2. **모델 로딩 문제**: device 이동 시 충돌
3. **CUDA 버전 불일치**: PyTorch와 CUDA 버전 문제
4. **메모리 부족**: GPU 메모리 초과

## 해결 방법

### 1. 안전한 inference 스크립트 사용

기존 `inference_anticipation.py` 대신 **`inference_simple.py`**를 사용하세요.

```bash
sh run_inference_simple.sh
```

### 2. 주요 수정 사항

#### A. 모델 로딩 개선
```python
# Before (문제 발생)
base = LlavaLlamaForCausalLM.from_pretrained(model_path)
base = base.to(device)

# After (안전)
base = LlavaLlamaForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)
base = base.to(device)
```

#### B. CLIP 모델 float 변환
```python
# CLIP 모델은 반드시 float32로 변환
clip_model = clip_model.float()
```

#### C. Device 관리 개선
```python
# 모델의 실제 device 확인
model_device = next(vlm_model.parameters()).device
INPUT_IDS = INPUT_IDS.to(model_device)
```

#### D. 메모리 관리
```python
import gc

# 모델 로딩 후 캐시 정리
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

### 3. 단계별 디버깅

만약 여전히 문제가 발생한다면:

#### Step 1: 모델 로딩만 테스트
```python
# 간단한 테스트 스크립트
import torch
from llava.model import LlavaLlamaForCausalLM

device = torch.device('cuda')
model = LlavaLlamaForCausalLM.from_pretrained(
    'liuhaotian/llava-v1.5-7b',
    torch_dtype=torch.float16,
)
model = model.to(device)
print("✓ Model loaded successfully")
```

#### Step 2: CLIP 로딩 테스트
```python
import clip
import torch

device = torch.device('cuda')
clip_model, _ = clip.load("ViT-B/32", device=device)
clip_model = clip_model.float()
print("✓ CLIP loaded successfully")
```

#### Step 3: 환경 생성 테스트
```python
from a2c_ppo_acktr.envs import make_vec_envs

utkinect_config = {
    "dataset_root": "/path/to/utkinect",
    "split": "test",
    "history_window": 6,
    "frame_skip": 1,
}

envs = make_vec_envs(
    'utkinect/test', 1, 1,
    0.99, None, device, False, 1,
    utkinect_config=utkinect_config
)
print("✓ Environment created successfully")
```

### 4. 환경 변수 설정

CUDA 관련 환경 변수를 설정하면 도움이 될 수 있습니다:

```bash
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### 5. 메모리 사용량 확인

```bash
# GPU 메모리 모니터링
watch -n 1 nvidia-smi
```

## 새로운 스크립트 사용법

### inference_simple.py 특징

1. **단계별 로딩**: 각 모델을 순차적으로 로드하고 확인
2. **에러 핸들링**: 각 단계에서 에러 발생 시 상세 정보 출력
3. **메모리 관리**: 모델 로딩 후 캐시 정리
4. **안전한 device 관리**: 모델의 실제 device 확인 후 사용

### 실행 방법

```bash
# 1. 간단한 테스트 (10 steps)
sh run_inference_simple.sh

# 2. 전체 테스트 (100 steps)
CUDA_VISIBLE_DEVICES=0 python inference_simple.py \
    --model-path liuhaotian/llava-v1.5-7b \
    --vlm-checkpoint vlm_checkpoints/epoch_4 \
    --futr-checkpoint /path/to/futr_checkpoint.ckpt \
    --utkinect-root /path/to/utkinect \
    --num-inference-steps 100 \
    --output-dir ./inference_results
```

### 체크포인트 형식

두 가지 형식 모두 지원:

1. **LoRA 디렉토리**: `vlm_checkpoints/epoch_4/`
2. **단일 .pt 파일**: `vlm_checkpoints/vlm_epoch_4.pt`

## 추가 팁

### 1. 작은 배치로 시작
```bash
--num-inference-steps 10  # 먼저 10 step만 테스트
```

### 2. 로그 확인
스크립트는 각 단계마다 상세한 로그를 출력합니다:
- ✓ 성공
- ⚠ 경고
- ✗ 실패

### 3. PyTorch 버전 확인
```python
import torch
print(torch.__version__)
print(torch.version.cuda)
```

권장 버전:
- PyTorch: 2.0+
- CUDA: 12.1+

## 문제가 계속되면

1. **Python 환경 재생성**
```bash
conda create -n vrenv_new python=3.10
conda activate vrenv_new
# 필요한 패키지 재설치
```

2. **CUDA 캐시 정리**
```bash
rm -rf ~/.cache/torch_extensions/
```

3. **시스템 로그 확인**
```bash
dmesg | tail -50
```

## 요약

- ✅ `inference_simple.py` 사용 (안전한 버전)
- ✅ `run_inference_simple.sh` 실행
- ✅ 10 steps로 먼저 테스트
- ✅ 각 단계별 로그 확인
- ✅ GPU 메모리 모니터링

문제가 해결되면 `--num-inference-steps`를 늘려서 전체 평가를 진행하세요.
