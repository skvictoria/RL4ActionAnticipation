# RL 학습 개선 사항 (2025)

## 문제점 분석 및 해결

### 1. 리워드 신호 문제 ❌ → ✅

**이전 문제:**
- VLM이 생성한 fine-grained description을 **미래 행동(target_next_action)**과 비교
- 리워드가 항상 -1.0 ~ 0.0 범위 (음수만)
- 최종 목표(action anticipation)와 무관한 신호

**해결 방법:**
```python
# 새로운 리워드 = Alignment Reward (30%) + Task Reward (70%)

# 1. Alignment Reward: Fine-grained text가 현재 coarse label을 잘 설명하는가?
alignment_reward = CLIP_similarity(fine_grained_text, current_action)  # 0~1

# 2. Task Reward: Fine-grained context가 action anticipation을 개선하는가?
task_reward = 1.0 if predicted_next == target_next else 0.0

# Combined
reward = 0.3 * alignment_reward + 0.7 * task_reward  # 0~1 범위
```

**효과:**
- 리워드가 0~1 범위로 정규화되어 학습 안정성 향상
- 최종 목표(anticipation accuracy)에 직접적으로 기여하는 신호 제공

---

### 2. 프롬프트 개선 📝

**이전 문제:**
- 프롬프트가 모호하고 목표가 불명확
- "fine-grained description for each frame"이라는 리스트 형태 요구
- VLM이 무엇을 생성해야 하는지 불분명

**해결 방법:**
```python
# 새로운 프롬프트 (rl_utils.py)
qs = "You are analyzing a sequence of 3 representative RGB frames from a video showing human actions. "
qs += "The coarse-level action labels for the observed sequence are: [walking, reaching, grasping]. "
qs += "Your task is to generate a detailed, fine-grained description that explains HOW the person is performing these actions. "
qs += "Focus on body movements, posture, hand positions, and motion patterns that characterize each action. "
qs += "This description will be used to predict the next action the person will perform. "
```

**핵심 변경:**
- 목적 명시: "HOW the person is performing these actions"
- 구체적 가이드: "body movements, posture, hand positions, motion patterns"
- 용도 설명: "will be used to predict the next action"
- 단일 텍스트 출력 (리스트 아님)

---

### 3. Action Anticipation 평가 추가 📊

**이전 문제:**
- 최종 목표가 action anticipation인데 평가 메트릭이 없음
- `predict_future` 함수가 정의되어 있지만 사용되지 않음

**해결 방법:**
```python
# train_rl.py에 추가
# 1. VLM이 fine-grained text 생성
text_outputs = actor_critic.act(...)

# 2. CLIP embedding으로 변환
fg_embeddings = clip_model.encode_text(text_outputs)

# 3. FUTR로 다음 행동 예측
predicted_actions = joint_model.predict_future(infos, fg_embeddings)

# 4. 정확도 계산 및 로깅
anticipation_acc = (predicted == target).mean()
wandb.log({"step/anticipation_accuracy": anticipation_acc})
```

**효과:**
- 학습 중 실시간으로 anticipation 성능 모니터링
- 리워드 신호와 실제 성능의 상관관계 확인 가능

---

### 4. 멀티 프레임 샘플링 개선 🎞️

**이전 문제:**
- `step=0`일 때 `[0, 0, 0]` → 같은 프레임 3번 반복
- 초반 스텝에서 의미 없는 입력

**해결 방법:**
```python
# train_rl.py
if step < 3:
    # Early steps: use available frames with repetition if needed
    history_indices = [0, max(0, step-1), step]
else:
    # Later steps: sample at 0.5, 0.75, 1.0 positions
    history_indices = [int(step * 0.5), int(step * 0.75), step]
```

**효과:**
- 초반 스텝에서도 최소한의 시간적 다양성 확보
- 프레임 간 최소 간격 보장

---

### 5. 성공률 판단 기준 수정 ✓

**이전:**
```python
episode_success_rate.append(1 if avg_reward > -0.5 else 0)  # -1~0 범위에서 -0.5
```

**수정:**
```python
episode_success_rate.append(1 if avg_reward > 0.5 else 0)  # 0~1 범위에서 0.5
```

---

### 6. 로깅 개선 📈

**추가된 메트릭:**
- `step/anticipation_accuracy`: 매 스텝 anticipation 정확도
- `eval/min_episode_reward`: 에피소드 최소 리워드
- `eval/max_episode_reward`: 에피소드 최대 리워드
- `futr_detail/loss_segmentation`: FUTR segmentation loss
- `futr_detail/loss_action`: FUTR action loss
- `futr_detail/loss_duration`: FUTR duration loss

---

## 전체 파이프라인 (수정 후)

```
1. FUTR predicts coarse labels from visual features
   └─> ["walking", "reaching", "grasping"]

2. VLM generates fine-grained description (with RL)
   └─> "The person is walking forward with arms swinging naturally, 
        then extending their right arm toward an object..."

3. CLIP encodes fine-grained text
   └─> [512-dim embedding]

4. FUTR uses fine-grained context for action anticipation
   └─> Predicted next action: "grasping"

5. Reward calculation
   ├─> Alignment: CLIP_sim(fine_text, current_coarse) = 0.85
   ├─> Task: (predicted == target) = 1.0
   └─> Final reward: 0.3 * 0.85 + 0.7 * 1.0 = 0.955
```

---

## 실행 방법

```bash
cd VLM_PPO_journal
python3 main.py --env_name utkinect --use_wandb --wandb_project your_project
```

---

## 기대 효과

1. **학습 안정성 향상**: 리워드가 0~1 범위로 정규화
2. **목표 정렬**: 리워드가 최종 목표(anticipation)와 직접 연결
3. **해석 가능성**: Fine-grained text가 구체적이고 명확
4. **모니터링 개선**: Anticipation accuracy 실시간 추적
5. **수렴 속도**: 명확한 신호로 더 빠른 학습 가능

---

## 주요 파일 변경

- `train_rl.py`: 리워드 계산, anticipation 평가 추가
- `a2c_ppo_acktr/rl_utils.py`: 프롬프트 개선, 리워드 함수 재설계
- `joint_model.py`: `predict_future` 함수 수정
- `mmaam/model/mMant.py`: Context 처리 주석 개선

---

## 다음 단계 (선택적 개선)

1. **Curriculum Learning**: 초반에는 alignment reward 비중 높이고, 후반에 task reward 비중 증가
2. **Reward Shaping**: Anticipation이 틀렸을 때도 부분 점수 (예: top-3 accuracy)
3. **Fine-grained Text Quality**: BLEU/ROUGE 같은 메트릭으로 텍스트 품질 평가
4. **Ablation Study**: Alignment vs Task reward 비율 실험 (0.3/0.7 vs 0.5/0.5 등)
