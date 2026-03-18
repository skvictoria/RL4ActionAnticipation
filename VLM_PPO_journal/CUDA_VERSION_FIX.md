# CUDA 버전 불일치 문제 해결

## 문제 상황

```
CUDAMismatchException: Installed CUDA version 13.0 does not match 
the version torch was compiled with 12.1
```

DeepSpeed의 CPU Adam optimizer가 CUDA 버전 불일치로 컴파일 실패.

---

## 해결 방법 (3가지)

### ✅ Option 1: DeepSpeed 비활성화 (권장 - 가장 빠름)

H200 GPU는 메모리가 충분하므로 DeepSpeed 없이도 학습 가능합니다.

```bash
# 새로운 스크립트 사용
chmod +x run_no_deepspeed.sh
sh run_no_deepspeed.sh
```

**장점**:
- ✅ CUDA 버전 문제 완전 회피
- ✅ 설정 간단
- ✅ H200 (141GB VRAM)에서 충분

**단점**:
- ❌ 메모리 최적화 없음 (하지만 H200에서는 문제 없음)

---

### ✅ Option 2: CPU Offload 비활성화

DeepSpeed는 사용하되 CPU offload만 비활성화:

```bash
# 수정된 설정 파일 사용
sh run.sh  # 자동으로 수정된 main.py 사용
```

또는 직접 설정 파일 변경:

```bash
accelerate launch \
    --config_file scripts/config_zero2_fixed.yaml \
    ...
```

**config_zero2_fixed.yaml**:
```yaml
deepspeed_config:
  offload_optimizer_device: none  # CPU offload 비활성화
  zero_stage: 2
```

**장점**:
- ✅ DeepSpeed ZeRO-2 최적화 유지
- ✅ CUDA 버전 문제 회피

**단점**:
- ⚠️ CPU offload 없어서 메모리 사용량 약간 증가 (H200에서는 문제 없음)

---

### ⚠️ Option 3: PyTorch 재설치 (비권장 - 시간 소요)

PyTorch를 CUDA 13.0 버전으로 재설치:

```bash
# 현재 환경 확인
python -c "import torch; print(torch.version.cuda)"  # 12.1

# CUDA 13.0용 PyTorch 설치 (시간 소요)
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

**장점**:
- ✅ 모든 기능 사용 가능

**단점**:
- ❌ 시간 소요 (10-20분)
- ❌ 다른 패키지 호환성 문제 가능
- ❌ 불필요 (H200에서는 Option 1, 2로 충분)

---

## 권장 실행 방법

### 1단계: DeepSpeed 없이 실행 (가장 빠름)

```bash
cd VLM_PPO_journal
chmod +x run_no_deepspeed.sh
sh run_no_deepspeed.sh
```

### 2단계: 메모리 사용량 확인

```bash
# 다른 터미널에서
watch -n 1 nvidia-smi
```

**예상 메모리 사용량**:
- LLaVA-7B with LoRA: ~15GB
- Value model: ~15GB
- FUTR model: ~2GB
- Activations & gradients: ~10GB
- **Total: ~42GB** (H200 141GB의 30%)

→ H200에서 DeepSpeed 없이도 충분히 여유!

---

## 이미 적용된 수정 사항

`main.py`에 다음 코드가 추가되어 있습니다:

```python
# DeepSpeed 설정 수정 (CUDA 버전 불일치 문제 해결)
if hasattr(AcceleratorState(), 'deepspeed_plugin') and AcceleratorState().deepspeed_plugin is not None:
    # CPU offload 비활성화
    if 'offload_optimizer' in AcceleratorState().deepspeed_plugin.deepspeed_config:
        AcceleratorState().deepspeed_plugin.deepspeed_config['offload_optimizer'] = {
            "device": "none"
        }
```

이제 기존 `run.sh`를 실행해도 CPU offload가 자동으로 비활성화됩니다!

---

## 실행 및 검증

### 실행

```bash
# Option 1: DeepSpeed 없이 (권장)
sh run_no_deepspeed.sh

# Option 2: DeepSpeed with fixed config
sh run.sh  # main.py의 수정사항이 자동 적용됨
```

### 성공 확인

```
✅ 성공 시:
using xformers
Model max context length: 1024
[JointFUTR] Loading weights from ...
[Warmup] Starting FUTR Warmup for 500 steps...

❌ 실패 시:
CUDAMismatchException: ...
→ run_no_deepspeed.sh 사용
```

---

## 성능 비교

| 설정 | 메모리 | 속도 | 안정성 |
|------|--------|------|--------|
| DeepSpeed ZeRO-2 + CPU offload | 낮음 | 느림 | ❌ CUDA 에러 |
| DeepSpeed ZeRO-2 (no offload) | 중간 | 빠름 | ✅ |
| No DeepSpeed | 중간 | 빠름 | ✅ |

H200 (141GB)에서는 **No DeepSpeed**가 가장 간단하고 안정적!

---

## 추가 최적화 (선택사항)

메모리가 부족하다면:

### 1. Gradient Checkpointing 활성화

이미 `main.py`에 구현되어 있음:
```python
use_grad_ckpt = True
if use_grad_ckpt:
    base.enable_input_require_grads()
```

### 2. Batch Size 조정

```bash
# run_no_deepspeed.sh에서
--mini-batch-size 4  # → 2로 줄이기
--grad-accum-steps 16  # → 32로 늘리기 (동일한 effective batch size)
```

### 3. Mixed Precision 확인

이미 활성화되어 있음:
```yaml
mixed_precision: bf16
```

---

## 문제 해결 체크리스트

- [ ] `run_no_deepspeed.sh` 실행
- [ ] `nvidia-smi`로 메모리 확인 (~42GB 사용)
- [ ] WandB 로그 확인 (`step/anticipation_moc` 보이는지)
- [ ] 첫 iteration 완료 확인 (~5-10분)

---

## 결론

**H200 GPU에서는 DeepSpeed 없이 실행하는 것이 가장 간단하고 안정적입니다!**

```bash
sh run_no_deepspeed.sh
```

이미 `main.py`에 CPU offload 비활성화 코드가 추가되어 있으므로, 기존 `run.sh`도 동작할 것입니다.
