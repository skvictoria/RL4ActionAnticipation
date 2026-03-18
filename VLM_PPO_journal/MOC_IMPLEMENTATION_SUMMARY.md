# MoC (Mean over Classes) 구현 요약

## 변경 사항

### 1. Action Anticipation: 1개 → 16개 전체 사용 ✅

**이전**:
```python
# 첫 번째 행동만 반환
result_future[i] = "grasping"  # 단일 문자열
```

**수정 후**:
```python
# 16개 모든 미래 행동 반환
result_future[i] = ["grasping", "grasping", "placing", "placing", ...]  # 16개 리스트
```

---

### 2. 환경에서 16개 Ground Truth 제공 ✅

**파일**: `a2c_ppo_acktr/datasets/utkinect.py`

```python
def _build_info(self):
    # ... (기존 코드)
    
    # Get future action sequence (up to 16 future actions)
    n_future = 16
    future_actions = []
    for offset in range(1, n_future + 1):
        future_idx = self._current_index + offset
        if future_idx < len(self._current_sequence):
            future_actions.append(self._current_sequence[future_idx]["label"])
        else:
            future_actions.append("none")  # Padding
    
    return {
        # ... (기존 필드)
        "target_future_sequence": future_actions,  # 16개 GT
    }
```

---

### 3. MoC 계산 함수 추가 ✅

**파일**: `a2c_ppo_acktr/rl_utils.py`

```python
def _compute_moc(predicted_seq: List[str], target_seq: List[str]) -> float:
    """
    Compute Mean over Classes (MoC) metric.
    
    MoC = Average of per-class recall across all classes
    
    Example:
        Target:    ["walk", "walk", "reach", "reach", "grasp", "grasp"]
        Predicted: ["walk", "reach", "reach", "reach", "grasp", "grasp"]
        
        Class "walk":  TP=1, Total=2 → Recall = 1/2 = 0.5
        Class "reach": TP=2, Total=2 → Recall = 2/2 = 1.0
        Class "grasp": TP=2, Total=2 → Recall = 2/2 = 1.0
        
        MoC = (0.5 + 1.0 + 1.0) / 3 = 0.833
    """
    # Normalize labels
    pred_norm = [_normalize_label(p) for p in predicted_seq]
    target_norm = [_normalize_label(t) for t in target_seq]
    
    # Get unique classes
    unique_classes = set(target_norm) - {"none"}
    
    # Compute per-class recall
    class_recalls = []
    for cls in unique_classes:
        tp = sum(1 for p, t in zip(pred_norm, target_norm) if p == cls and t == cls)
        total_pos = sum(1 for t in target_norm if t == cls)
        
        if total_pos > 0:
            recall = tp / total_pos
            class_recalls.append(recall)
    
    # Mean over all classes
    return sum(class_recalls) / len(class_recalls) if class_recalls else 0.0
```

---

### 4. 리워드 계산에 MoC 적용 ✅

**파일**: `a2c_ppo_acktr/rl_utils.py`

```python
def semantic_reward_from_text(...):
    # ... (Alignment reward 계산)
    
    # Task Reward: MoC 기반
    if joint_model is not None:
        predicted_seq = info.get("predicted_future_sequence", ["none"] * 16)
        target_seq = info.get("target_future_sequence", ["none"] * 16)
        
        moc_score = _compute_moc(predicted_seq, target_seq)  # 0~1
        task_reward = moc_score
    
    # Combined
    reward = 0.3 * alignment_reward + 0.7 * task_reward
```

---

### 5. 로깅 개선 ✅

**파일**: `train_rl.py`

```python
# 매 스텝마다 MoC 계산 및 로깅
if joint_model is not None:
    moc_scores = []
    for info in infos:
        pred_seq = info.get('predicted_future_sequence', ["none"] * 16)
        target_seq = info.get('target_future_sequence', ["none"] * 16)
        moc = rl_utils._compute_moc(pred_seq, target_seq)
        moc_scores.append(moc)
    
    avg_moc = sum(moc_scores) / len(moc_scores)
    wandb.log({"step/anticipation_moc": avg_moc})
    
    # 첫 번째 행동 정확도도 함께 로깅 (비교용)
    first_correct = sum([
        1 for info in infos 
        if info.get('predicted_future_sequence', ["none"])[0] == 
           info.get('target_next_action', 'none')
    ])
    first_acc = first_correct / len(infos)
    wandb.log({"step/first_action_accuracy": first_acc})
```

---

## MoC vs 단순 Accuracy 비교

### 예시 1: 불균형 데이터

```
Target:    ["walk"]*10 + ["grasp"]*2 + ["place"]*4
Predicted: ["walk"]*16

단순 Accuracy: 10/16 = 0.625 (높음!)
MoC:
  - walk:  10/10 = 1.0
  - grasp: 0/2   = 0.0
  - place: 0/4   = 0.0
  - MoC = (1.0 + 0.0 + 0.0) / 3 = 0.333 (낮음!)
```

→ MoC는 소수 클래스도 공평하게 평가!

### 예시 2: 균형 데이터

```
Target:    ["walk"]*5 + ["reach"]*5 + ["grasp"]*6
Predicted: ["walk"]*5 + ["reach"]*5 + ["grasp"]*6

단순 Accuracy: 16/16 = 1.0
MoC:
  - walk:  5/5 = 1.0
  - reach: 5/5 = 1.0
  - grasp: 6/6 = 1.0
  - MoC = (1.0 + 1.0 + 1.0) / 3 = 1.0
```

→ 완벽한 예측 시 동일!

---

## 데이터 흐름 (최종)

```
┌─────────────────────────────────────────────────────────────┐
│ 1. 환경: 현재 프레임 + 16개 미래 GT                          │
│    current_action: "reaching"                               │
│    target_future_sequence: ["grasping", "grasping", ...]   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. FUTR: Visual → Coarse Labels (16개)                     │
│    ["walking", "walking", "reaching", ...]                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. VLM: Coarse + Images → Fine-grained Text (1개)          │
│    "The person walks forward with arms swinging..."         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. CLIP: Text → Embedding (1개 → 16개 복사)                │
│    [emb, emb, emb, ..., emb]  # [16, 512]                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. FUTR: Visual + Context → Future Actions (16개)          │
│    predicted_future_sequence: ["grasping", "grasping", ...] │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. MoC 계산                                                 │
│    Predicted: ["grasping", "grasping", "placing", ...]     │
│    Target:    ["grasping", "grasping", "placing", ...]     │
│    MoC = 0.85                                               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 7. Reward                                                   │
│    Alignment: 0.80                                          │
│    Task (MoC): 0.85                                         │
│    Final: 0.3 * 0.80 + 0.7 * 0.85 = 0.835                  │
└─────────────────────────────────────────────────────────────┘
```

---

## WandB 로그 (예상)

```
step/reward: 0.835
step/futr_loss: 1.234
step/anticipation_moc: 0.85          ← MoC 점수
step/first_action_accuracy: 0.92     ← 첫 번째 행동만 정확도
train/value_loss: 0.12
train/action_loss: 0.34
eval/mean_episode_reward: 0.78
eval/success_rate: 0.65
```

---

## Fine-grained Text 개선 (다음 단계)

현재는 **1개 통합 설명 → 16번 복사** 방식 사용.

개선 방안은 `FINE_GRAINED_TEXT_IMPROVEMENT.md` 참고:
- Option A-1: 16개 개별 설명 생성
- Option A-3: Hierarchical description (권장)

---

## 실행 및 검증

```bash
cd VLM_PPO_journal
python3 main.py \
  --env_name utkinect \
  --utkinect_root /path/to/dataset \
  --use_wandb \
  --wandb_project your_project
```

**확인 사항**:
1. `step/anticipation_moc` 로그가 보이는가?
2. MoC 값이 0~1 범위인가?
3. 학습이 진행되면서 MoC가 증가하는가?
4. First-action accuracy와 MoC의 차이는?

---

## 기대 효과

1. **공평한 평가**: 소수 클래스도 동등하게 평가
2. **더 나은 학습 신호**: 전체 시퀀스 예측 품질 반영
3. **표준 메트릭**: Action anticipation 논문에서 널리 사용
4. **디버깅 용이**: 클래스별 성능 분석 가능

---

## 참고 문헌

- FUTR 논문: "FUTR: Future Transformer for Action Anticipation"
- MoC 메트릭: Breakfast Actions, EPIC-Kitchens 등에서 사용
- Mean over Classes = Macro-averaged Recall
