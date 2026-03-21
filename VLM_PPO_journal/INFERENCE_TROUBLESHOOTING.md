# Inference Segmentation Fault 문제 해결

## 문제 상황

```bash
Segmentation fault (core dumped)
```

inference 실행 시 tokenizer 로딩 단계에서 segmentation fault 발생

## 원인

Segmentation fault는 다음과 같은 이유로 발생할 수 있습니다:

1. **HuggingFace transformers 라이브러리 버전 충돌**
2. **Tokenizer 로딩 시 메모리 문제**
3. **LLaVA 모델 로딩 시 CUDA 메모리 문제**
4. **라이브러리 간 호환성 문제**

## 해결 방법

### 방법 1: 테스트 스크립트로 문제 진단

먼저 어느 단계에서 문제가 발생하는지 확인:

```bash
sh run_test_loading.sh
```

이 스크립트는 각 단계를 개별적으로 테스트합니다:
1. transformers import
2. tokenizer 로딩
3. LLaVA import
4. LLaVA 모델 로딩
5. Device 이동
6. Eval 모드 설정

### 방법 2: Minimal Inference (권장)

VLM 생성 없이 FUTR만 사용하는 간단한 버전:

```bash
sh run_inference_minimal.sh
```

이 방법은:
- 학습 코드와 동일한 방식으로 모델 로딩
- VLM 텍스트 생성 단계 생략 (coarse labels 직접 사용)
- Segfault 발생 가능성 최소화

### 방법 3: 환경 변수 설정

```bash
export TOKENIZERS_PARALLELISM=false
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

그 후 다시 실행:

```bash
sh run_inference_simple.sh
```

### 방법 4: 라이브러리 재설치

```bash
pip uninstall transformers tokenizers -y
pip install transformers==4.36.0 tokenizers==0.15.0
```

### 방법 5: 새로운 환경 생성

```bash
conda create -n vrenv_inference python=3.10
conda activate vrenv_inference

# 필수 패키지 설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.36.0
pip install accelerate peft
pip install clip
pip install einops
pip install gym
```

## 스크립트 비교

### 1. inference_minimal.py (가장 안전)
- ✅ 학습 코드와 동일한 모델 로딩 방식
- ✅ VLM 생성 단계 생략
- ✅ Segfault 발생 가능성 낮음
- ⚠️ Fine-grained text 대신 coarse labels 사용

```bash
sh run_inference_minimal.sh
```

### 2. inference_simple.py (중간)
- ✅ 단계별 에러 핸들링
- ✅ 상세한 로그
- ⚠️ VLM 로딩 시 segfault 가능

```bash
sh run_inference_simple.sh
```

### 3. inference_anticipation.py (전체 기능)
- ✅ 모든 기능 포함
- ✅ VLM 텍스트 생성
- ⚠️ 가장 복잡하여 segfault 가능성 높음

```bash
sh run_inference.sh
```

## 권장 순서

1. **먼저 테스트**: `sh run_test_loading.sh`
   - 어느 단계에서 문제가 발생하는지 확인

2. **Minimal 버전 실행**: `sh run_inference_minimal.sh`
   - VLM 없이 FUTR만 테스트
   - 이것이 작동하면 FUTR는 정상

3. **환경 변수 설정 후 Simple 버전**: 
   ```bash
   export TOKENIZERS_PARALLELISM=false
   sh run_inference_simple.sh
   ```

4. **라이브러리 재설치 후 재시도**

## 디버깅 팁

### 1. 메모리 모니터링
```bash
watch -n 1 nvidia-smi
```

### 2. 시스템 로그 확인
```bash
dmesg | tail -50
```

### 3. Python 버전 확인
```bash
python --version  # 3.10 권장
```

### 4. PyTorch 버전 확인
```python
import torch
print(torch.__version__)  # 2.0+ 권장
print(torch.version.cuda)  # 12.1+ 권장
```

### 5. Transformers 버전 확인
```python
import transformers
print(transformers.__version__)  # 4.36.0 권장
```

## 체크포인트 형식

모든 스크립트는 두 가지 형식 지원:

1. **LoRA 디렉토리**: `vlm_checkpoints/epoch_4/`
2. **단일 .pt 파일**: `vlm_checkpoints/vlm_epoch_4.pt`

## 결과 파일

성공 시 다음 파일이 생성됩니다:

```
inference_results/
├── inference_results.json    # 상세 결과
└── inference_stats.json      # 통계 (MoC, First-Acc)
```

## 여전히 문제가 발생하면

1. **CUDA 캐시 정리**:
   ```bash
   rm -rf ~/.cache/torch_extensions/
   ```

2. **HuggingFace 캐시 정리**:
   ```bash
   rm -rf ~/.cache/huggingface/
   ```

3. **완전히 새로운 환경**:
   ```bash
   conda create -n fresh_env python=3.10
   conda activate fresh_env
   # 처음부터 설치
   ```

4. **다른 노드에서 실행**:
   - 현재 노드에 하드웨어 문제가 있을 수 있음

## 요약

**가장 빠른 해결책**:
```bash
# 1. Minimal 버전 실행 (VLM 없이)
sh run_inference_minimal.sh

# 2. 작동하면 환경 변수 설정 후 Simple 버전
export TOKENIZERS_PARALLELISM=false
sh run_inference_simple.sh
```

이 방법으로 최소한 FUTR의 action anticipation 성능은 평가할 수 있습니다.
