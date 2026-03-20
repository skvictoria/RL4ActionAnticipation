# 체크포인트 저장 및 로드 가이드

## 📦 체크포인트 저장 (Training)

### 자동 저장

학습 중 `--save_interval` 마다 자동으로 체크포인트가 저장됩니다.

```bash
# run.sh 또는 run_no_deepspeed.sh에서
--save_interval 5  # 5 iteration마다 저장
```

### 저장 위치

```
FUTR_MODEL_PATH/
├── futr_joint_epoch_0.ckpt
├── futr_joint_epoch_5.ckpt
├── futr_joint_epoch_10.ckpt
├── ...
└── vlm_checkpoints/
    ├── epoch_0/
    │   ├── adapter_config.json
    │   ├── adapter_model.bin
    │   └── ...
    ├── epoch_5/
    │   └── ...
    ├── vlm_epoch_0.pt
    ├── vlm_epoch_5.pt
    └── vlm_epoch_10.pt
```

### 저장 내용

#### 1. FUTR 체크포인트 (`.ckpt`)

```python
# futr_joint_epoch_X.ckpt
{
    'model_state_dict': {...},  # FUTR 모델 가중치
}
```

#### 2. VLM 체크포인트 (2가지 형식)

**A. LoRA 디렉토리** (`epoch_X/`)
```
epoch_X/
├── adapter_config.json      # LoRA 설정
├── adapter_model.bin         # LoRA 가중치
├── tokenizer_config.json     # Tokenizer 설정
└── special_tokens_map.json   # 특수 토큰
```

**B. 단일 .pt 파일** (`vlm_epoch_X.pt`)
```python
{
    'iteration': X,
    'model_state_dict': {...},      # 전체 모델 가중치
    'optimizer_state_dict': {...},  # Optimizer 상태
    'lr_scheduler_state_dict': {...}, # LR scheduler 상태
}
```

---

## 📂 체크포인트 로드 (Inference)

### Option 1: .pt 파일 사용 (권장)

```bash
python inference_anticipation.py \
    --vlm-checkpoint ./checkpoints/vlm_checkpoints/vlm_epoch_10.pt \
    --futr-checkpoint ./checkpoints/futr_joint_epoch_10.ckpt \
    ...
```

**장점**:
- ✅ 단일 파일로 관리 용이
- ✅ Optimizer 상태 포함 (재학습 가능)
- ✅ 빠른 로드

### Option 2: LoRA 디렉토리 사용

```bash
python inference_anticipation.py \
    --vlm-checkpoint ./checkpoints/vlm_checkpoints/epoch_10 \
    --futr-checkpoint ./checkpoints/futr_joint_epoch_10.ckpt \
    ...
```

**장점**:
- ✅ HuggingFace 표준 형식
- ✅ 다른 도구와 호환성 좋음
- ✅ 설정 파일 포함

---

## 🔍 체크포인트 확인

### 저장된 체크포인트 목록 확인

```bash
# FUTR 체크포인트
ls -lh checkpoints/*.ckpt

# VLM 체크포인트 (.pt)
ls -lh checkpoints/vlm_checkpoints/*.pt

# VLM 체크포인트 (디렉토리)
ls -d checkpoints/vlm_checkpoints/epoch_*/
```

### 체크포인트 정보 확인

```python
import torch

# VLM .pt 파일 확인
checkpoint = torch.load('checkpoints/vlm_checkpoints/vlm_epoch_10.pt')
print(f"Iteration: {checkpoint['iteration']}")
print(f"Keys: {checkpoint.keys()}")

# FUTR .ckpt 파일 확인
futr_ckpt = torch.load('checkpoints/futr_joint_epoch_10.ckpt')
print(f"FUTR keys: {futr_ckpt.keys()}")
```

---

## 🔧 체크포인트 관리

### 디스크 공간 절약

오래된 체크포인트 삭제:

```bash
# 최신 3개만 유지
cd checkpoints/vlm_checkpoints
ls -t vlm_epoch_*.pt | tail -n +4 | xargs rm -f

# 또는 특정 epoch만 유지
rm vlm_epoch_{0..4}.pt  # epoch 0-4 삭제
```

### Best 모델 선택

WandB 로그를 보고 가장 성능이 좋은 epoch 선택:

```bash
# 예: epoch 15가 best MoC를 기록
cp checkpoints/vlm_checkpoints/vlm_epoch_15.pt checkpoints/vlm_best.pt
cp checkpoints/futr_joint_epoch_15.ckpt checkpoints/futr_best.ckpt
```

---

## 🚨 문제 해결

### 1. 체크포인트 파일이 없음

```
FileNotFoundError: checkpoints/vlm_epoch_10.pt not found
```

**원인**: 학습이 해당 epoch까지 진행되지 않음

**해결**:
```bash
# 저장된 체크포인트 확인
ls checkpoints/vlm_checkpoints/

# 가장 최근 체크포인트 사용
--vlm-checkpoint checkpoints/vlm_checkpoints/vlm_epoch_5.pt
```

### 2. 체크포인트 로드 실패

```
RuntimeError: Error(s) in loading state_dict
```

**원인**: LoRA 설정 불일치

**해결**:
```python
# inference_anticipation.py에서 target_modules 확인
base_lora_config = LoraConfig(
    r=128,
    lora_alpha=256,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # 학습 시와 동일하게
    ...
)
```

### 3. 메모리 부족

```
CUDA out of memory when loading checkpoint
```

**해결**:
```bash
# 더 작은 모델 사용
--model-path liuhaotian/llava-v1.5-7b  # 13b 대신 7b

# 또는 CPU로 로드 후 GPU로 이동
# inference_anticipation.py 수정 필요
```

---

## 📊 체크포인트 비교

여러 체크포인트 성능 비교:

```bash
# 스크립트 작성
for epoch in 5 10 15 20; do
    echo "Testing epoch $epoch..."
    python inference_anticipation.py \
        --vlm-checkpoint checkpoints/vlm_checkpoints/vlm_epoch_${epoch}.pt \
        --futr-checkpoint checkpoints/futr_joint_epoch_${epoch}.ckpt \
        --num-inference-steps 50 \
        --output-dir results_epoch_${epoch}
done

# 결과 비교
for epoch in 5 10 15 20; do
    echo "Epoch $epoch:"
    cat results_epoch_${epoch}/inference_summary.txt | grep "Average MoC"
done
```

---

## 💾 체크포인트 백업

중요한 체크포인트 백업:

```bash
# 로컬 백업
tar -czf checkpoints_backup_$(date +%Y%m%d).tar.gz checkpoints/

# 원격 서버로 복사
scp checkpoints_backup_*.tar.gz user@server:/backup/

# 복원
tar -xzf checkpoints_backup_20250120.tar.gz
```

---

## 🔄 재학습 (Resume Training)

저장된 체크포인트에서 재학습:

```python
# main.py에 추가 (이미 구현됨)
# FUTR는 자동으로 epoch 번호 파싱하여 재개
# VLM은 optimizer/scheduler 상태 로드 필요

# 예시:
if args.resume_from:
    checkpoint = torch.load(args.resume_from)
    actor_critic.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
    start_epoch = checkpoint['iteration'] + 1
```

---

## 📝 체크포인트 메타데이터

체크포인트에 추가 정보 저장:

```python
# main.py 수정 예시
torch.save({
    'iteration': j,
    'model_state_dict': vlm_model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'lr_scheduler_state_dict': lr_scheduler.state_dict(),
    # 추가 메타데이터
    'args': vars(args),
    'avg_moc': avg_moc,
    'avg_reward': avg_reward,
    'timestamp': time.time(),
}, vlm_checkpoint_file)
```

---

## 🎯 Best Practices

1. **정기적 저장**: `--save_interval 5` (너무 자주 저장하면 디스크 부족)
2. **명명 규칙**: `epoch_X` 형식 유지
3. **백업**: 중요한 체크포인트는 별도 백업
4. **정리**: 오래된 체크포인트 주기적 삭제
5. **검증**: 저장 후 로드 테스트
6. **문서화**: 각 체크포인트의 성능 기록

---

## 📚 참고

- 학습 가이드: `README.md`
- 인퍼런스 가이드: `INFERENCE_GUIDE.md`
- LoRA 문서: https://huggingface.co/docs/peft
