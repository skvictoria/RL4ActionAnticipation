# 체크포인트 경로 설명

## 헷갈리기 쉬운 개념 정리

### 1. `--cache-dir` (HuggingFace 캐시)

**용도:**
- HuggingFace Hub에서 다운로드한 **base model**을 저장하는 디렉토리
- 예: `liuhaotian/llava-v1.5-7b` 같은 pretrained model

**기본값:**
- `None` → 자동으로 `~/.cache/huggingface/` 사용
- 명시적으로 지정하지 않아도 됨

**예시:**
```bash
# cache-dir를 지정하지 않으면
python3 inference.py --model-path liuhaotian/llava-v1.5-7b
# → ~/.cache/huggingface/에 자동 다운로드

# cache-dir를 지정하면
python3 inference.py \
    --model-path liuhaotian/llava-v1.5-7b \
    --cache-dir /custom/cache/path
# → /custom/cache/path/에 다운로드
```

**언제 사용?**
- 디스크 공간이 부족한 경우 (다른 디스크로 변경)
- 여러 사용자가 공유하는 캐시 사용
- 대부분의 경우 **지정하지 않아도 됨**

---

### 2. `--vlm-checkpoint` (학습된 VLM)

**용도:**
- **당신이 학습한 LoRA weights**를 저장한 디렉토리
- `main.py`에서 저장한 체크포인트

**저장 위치 (main.py에서):**
```python
# main.py에서 저장되는 경로
vlm_checkpoint_dir = os.path.join(save_dir_base, "vlm_checkpoints")
vlm_save_dir = os.path.join(vlm_checkpoint_dir, f"epoch_{j}")

# 예시:
# /home/.../FUTR_proposed/save_dir/.../vlm_checkpoints/epoch_4/
```

**필수 파일들:**
```
vlm_checkpoints/epoch_4/
├── config.json              # 모델 설정
├── adapter_config.json      # LoRA 설정
├── adapter_model.bin        # LoRA 가중치 (중요!)
├── tokenizer_config.json    # Tokenizer 설정
├── tokenizer.model          # Tokenizer 모델
└── special_tokens_map.json  # 특수 토큰
```

**예시:**
```bash
python3 inference_anticipation.py \
    --vlm-checkpoint /home/.../vlm_checkpoints/epoch_4 \
    ...
```

---

### 3. `--futr-checkpoint` (학습된 FUTR)

**용도:**
- **학습된 FUTR 모델**의 체크포인트 파일
- `joint_model.py`에서 저장한 `.ckpt` 파일

**저장 위치 (main.py에서):**
```python
# main.py에서 저장되는 경로
futr_save_path = os.path.join(save_dir_base, f"futr_joint_epoch_{j}.ckpt")

# 예시:
# /home/.../FUTR_proposed/save_dir/.../futr_joint_epoch_66.ckpt
```

**예시:**
```bash
python3 inference_anticipation.py \
    --futr-checkpoint /home/.../futr_joint_epoch_66.ckpt \
    ...
```

---

## 실제 사용 예시

### 시나리오 1: 기본 설정 (cache-dir 지정 안 함)

```bash
python3 inference_anticipation.py \
    --model-path liuhaotian/llava-v1.5-7b \
    --vlm-checkpoint ./vlm_checkpoints/epoch_4 \
    --futr-checkpoint ./futr_joint_epoch_66.ckpt \
    --utkinect-root /path/to/utkinect
```

**동작:**
- Base model은 `~/.cache/huggingface/`에 자동 다운로드
- VLM LoRA weights는 `./vlm_checkpoints/epoch_4/`에서 로드
- FUTR weights는 `./futr_joint_epoch_66.ckpt`에서 로드

---

### 시나리오 2: 커스텀 캐시 디렉토리

```bash
python3 inference_anticipation.py \
    --model-path liuhaotian/llava-v1.5-7b \
    --cache-dir /scratch/cache \
    --vlm-checkpoint ./vlm_checkpoints/epoch_4 \
    --futr-checkpoint ./futr_joint_epoch_66.ckpt \
    --utkinect-root /path/to/utkinect
```

**동작:**
- Base model은 `/scratch/cache/`에 다운로드
- VLM/FUTR는 동일

---

### 시나리오 3: 절대 경로 사용

```bash
python3 inference_anticipation.py \
    --model-path liuhaotian/llava-v1.5-7b \
    --vlm-checkpoint /home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/save_dir/utkinects/long/model/transformer/1/i3d_transcript/runs0/_20_30_50_erank_40p_64_latent_20251226/vlm_checkpoints/epoch_4 \
    --futr-checkpoint /home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/save_dir/utkinects/long/model/transformer/1/i3d_transcript/runs0/_20_30_50_erank_40p_64_latent_20251226/futr_joint_epoch_66.ckpt \
    --utkinect-root /home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/datasets/utkinect
```

---

## 체크포인트 확인 방법

### 1. VLM 체크포인트 확인

```bash
# 디렉토리 존재 확인
ls -la /path/to/vlm_checkpoints/epoch_4/

# 필수 파일 확인
ls -la /path/to/vlm_checkpoints/epoch_4/adapter_model.bin
ls -la /path/to/vlm_checkpoints/epoch_4/config.json
```

**예상 출력:**
```
total 256M
-rw-r--r-- 1 user group  512 Apr  2 10:00 adapter_config.json
-rw-r--r-- 1 user group 256M Apr  2 10:00 adapter_model.bin
-rw-r--r-- 1 user group  2.0K Apr  2 10:00 config.json
-rw-r--r-- 1 user group  1.5K Apr  2 10:00 tokenizer_config.json
-rw-r--r-- 1 user group  500K Apr  2 10:00 tokenizer.model
```

---

### 2. FUTR 체크포인트 확인

```bash
# 파일 존재 확인
ls -lh /path/to/futr_joint_epoch_66.ckpt
```

**예상 출력:**
```
-rw-r--r-- 1 user group 1.2G Apr  2 10:00 futr_joint_epoch_66.ckpt
```

---

## 학습 시 저장 경로

### main.py에서 저장되는 구조

```
FUTR_MODEL_PATH (base directory)
├── futr_joint_epoch_0.ckpt
├── futr_joint_epoch_10.ckpt
├── futr_joint_epoch_20.ckpt
├── ...
├── futr_joint_epoch_66.ckpt
└── vlm_checkpoints/
    ├── epoch_0/
    │   ├── adapter_config.json
    │   ├── adapter_model.bin
    │   ├── config.json
    │   └── tokenizer_config.json
    ├── epoch_10/
    ├── epoch_20/
    ├── ...
    ├── epoch_66/
    ├── vlm_epoch_0.pt
    ├── vlm_epoch_10.pt
    └── vlm_epoch_66.pt
```

---

## 자주 묻는 질문

### Q1: cache-dir를 학습 폴더로 지정해야 하나요?

**A:** 아니요! `cache-dir`는 HuggingFace base model용이고, 학습된 체크포인트는 별도입니다.

```bash
# ✗ 잘못된 사용
--cache-dir /path/to/vlm_checkpoints/epoch_4

# ✓ 올바른 사용
--vlm-checkpoint /path/to/vlm_checkpoints/epoch_4
--cache-dir ~/.cache/huggingface  # 또는 생략
```

---

### Q2: VLM 체크포인트가 없으면?

**A:** FUTR-only inference를 사용하세요:

```bash
sh run_inference_robust.sh
```

이 스크립트는 VLM 없이 FUTR만 사용합니다.

---

### Q3: Tokenizer 파일이 VLM 체크포인트에 없으면?

**A:** Base model의 tokenizer를 자동으로 사용합니다:

```bash
# VLM 체크포인트에 tokenizer가 없어도
python3 inference_anticipation.py \
    --model-path liuhaotian/llava-v1.5-7b \  # 여기서 tokenizer 로드
    --vlm-checkpoint ./vlm_checkpoints/epoch_4 \
    ...
```

---

### Q4: 어떤 epoch를 사용해야 하나요?

**A:** 일반적으로:
- 마지막 epoch (가장 많이 학습됨)
- 또는 validation 성능이 가장 좋은 epoch

```bash
# 여러 epoch 테스트
for epoch in 50 60 66; do
    python3 inference_anticipation.py \
        --vlm-checkpoint ./vlm_checkpoints/epoch_$epoch \
        --futr-checkpoint ./futr_joint_epoch_$epoch.ckpt \
        --output-dir ./results_epoch_$epoch
done
```

---

## 요약

| 인자 | 용도 | 필수? | 기본값 |
|------|------|-------|--------|
| `--cache-dir` | HuggingFace base model 캐시 | ❌ | `~/.cache/huggingface/` |
| `--vlm-checkpoint` | 학습된 VLM LoRA weights | ✅ | 없음 |
| `--futr-checkpoint` | 학습된 FUTR weights | ✅ | 없음 |
| `--model-path` | Base model 이름/경로 | ✅ | 없음 |

**핵심:**
- `cache-dir`는 대부분 **지정하지 않아도 됨**
- `vlm-checkpoint`와 `futr-checkpoint`는 **학습 시 저장한 경로** 사용
- `model-path`는 보통 `liuhaotian/llava-v1.5-7b` 사용

---

## 실전 예시

### 당신의 경로 (예상)

```bash
# 학습 시 저장된 경로
BASE_DIR="/home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/save_dir/utkinects/long/model/transformer/1/i3d_transcript/runs0/_20_30_50_erank_40p_64_latent_20251226"

# VLM checkpoint
VLM_CHECKPOINT="$BASE_DIR/vlm_checkpoints/epoch_66"

# FUTR checkpoint
FUTR_CHECKPOINT="$BASE_DIR/futr_joint_epoch_66.ckpt"

# Inference 실행
python3 inference_anticipation.py \
    --model-path liuhaotian/llava-v1.5-7b \
    --vlm-checkpoint "$VLM_CHECKPOINT" \
    --futr-checkpoint "$FUTR_CHECKPOINT" \
    --utkinect-root /path/to/utkinect
```

**cache-dir는 지정하지 않음!** (자동으로 `~/.cache/huggingface/` 사용)
