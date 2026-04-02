# 어떤 파일을 사용해야 할까?

## 빠른 답변

### Inference (추론)만 할 경우

```bash
# LoRA 디렉토리 사용 (256MB)
python3 inference_anticipation.py \
    --vlm-checkpoint ./vlm_checkpoints/epoch_66 \
    ...
```

**30GB .pt 파일은 필요 없음!** → 삭제 가능

---

### 학습을 더 하고 싶은 경우 (Resume)

```bash
# .pt 파일 사용 (30GB)
python3 main.py \
    --resume-from ./vlm_checkpoints/vlm_epoch_66.pt \
    ...
```

**LoRA 디렉토리는 선택사항** → 삭제 가능

---

## 파일 비교

| 파일 | 크기 | 용도 |
|------|------|------|
| `vlm_checkpoints/epoch_66/` | 256MB | ✅ Inference |
| `vlm_checkpoints/vlm_epoch_66.pt` | 30GB | ✅ Training resume |

**성능은 동일!** 차이는 용도와 크기뿐입니다.

---

## 디스크 공간 절약

### Inference만 한다면

```bash
# 30GB 절약!
rm vlm_checkpoints/vlm_epoch_*.pt

# LoRA 디렉토리만 유지
ls vlm_checkpoints/
# epoch_0/  epoch_10/  ...  epoch_66/
```

---

### 학습 재개가 필요하다면

```bash
# 최신 .pt만 유지
rm vlm_checkpoints/vlm_epoch_{0..60}.pt

# 최신 것만 남김
ls vlm_checkpoints/vlm_epoch_*.pt
# vlm_epoch_66.pt
```

---

## 현재 상황에서 추천

**당신의 경우:**
- `diagnose_segfault.py` 성공 → Inference 테스트 단계
- 학습은 이미 완료됨

**추천:**
1. LoRA 디렉토리로 inference 테스트
2. 성공하면 .pt 파일 삭제 (30GB 절약)
3. 나중에 학습을 더 하고 싶으면 .pt 파일 복원

---

## 자세한 설명

- `CHECKPOINT_FORMATS_EXPLAINED.md` 참고
