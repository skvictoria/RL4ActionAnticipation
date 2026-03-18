# 최종 코드 검토 및 수정 사항

## 질문에 대한 답변

### 1. Coarse Label은 몇 개를 Prediction하는지?

**답변**: **가변 길이** (관찰된 프레임 수만큼)

```python
# joint_model.py - predict_coarse()
# 예시: 16개 프레임 샘플링 → 16개 coarse label 반환
result = [["walking", "walking", "reaching", "reaching", "grasping", ...]]  # 16개
```

**세부 사항**:
- `_prepare_batch()`에서 최대 16개 프레임을 uniform sampling
- 각 프레임마다 1개의 coarse label 예측
- Segmentation task이므로 프레임 수 = 레이블 수

---

### 2. Fine-grained Text는 똑같은 개수를 Prediction하는지?

**답변**: **아니오, 단 1개의 텍스트만 생성**

```python
# VLM 출력 예시
text_output = '''
{
  "thoughts": "The person is walking forward with arms swinging naturally, 
               then extending their right arm toward an object while bending knees..."
}
'''
```

**불일치 이유**:
- Coarse labels: 16개 (프레임별)
- Fine-grained text: 1개 (전체 시퀀스에 대한 통합 설명)
- 이는 의도된 설계: VLM이 여러 coarse label을 보고 **하나의 통합된 상세 설명** 생성

**프롬프트 설계**:
```python
# rl_utils.py
qs = f"The coarse-level action labels are: [{history_text}]."  # 여러 개 입력
qs += "Your task is to generate a detailed, fine-grained description..."  # 1개 출력
```

---

### 3. Action Anticipation Label은 몇 개를 Prediction하는지?

**답변**: **n_query=16개 예측하지만, 첫 번째만 사용**

```python
# joint_model.py - predict_future()
# FUTR 모델 출력: [B, 16, n_class]
# 하지만 반환값: 각 info당 1개 문자열

result = ["grasping", "grasping", ...]  # 각 프로세스당 1개 (첫 번째 쿼리만)
```

**수정 전 문제**:
```python
# ❌ 이전 코드: 16개 딕셔너리 리스트 반환
result_future[i] = [
    {"action": "grasping", "duration": 0.3},
    {"action": "placing", "duration": 0.2},
    ...  # 16개
]
```

**수정 후**:
```python
# ✅ 수정 코드: 첫 번째 행동만 문자열로 반환
n_act = act_preds[batch_idx][0]  # 첫 번째 쿼리만
pred_str = self.inverse_dict.get(n_act, "none")
result_future[i] = pred_str  # 단일 문자열
```

---

### 4. Action Anticipation은 어떻게 검증하는지?

**답변**: **환경의 target_next_action과 비교**

#### 검증 흐름:

```python
# Step 1: 환경에서 Ground Truth 제공
# utkinect.py - _build_info()
next_label = self._current_sequence[self._current_index + 1]["label"]
info["target_next_action"] = next_label  # 예: "grasping"

# Step 2: FUTR로 예측
# train_rl.py
predicted_actions = joint_model.predict_future(prev_infos, fg_batch)
# 예: ["grasping", "reaching", ...]

# Step 3: info에 저장
for i, pred_act in enumerate(predicted_actions):
    infos[i]['predicted_next_action'] = pred_act  # 단일 문자열

# Step 4: 리워드 계산 시 비교
# rl_utils.py - semantic_reward_from_text()
predicted_next = _normalize_label(info.get("predicted_next_action", "none"))
target_next = _normalize_label(info.get("target_next_action", "none"))

if predicted_next == target_next:
    task_reward = 1.0  # 정확!
else:
    task_reward = 0.0  # 틀림

# Step 5: 정확도 로깅
# train_rl.py
correct_count = sum([
    1 for info in infos 
    if _normalize_label(info.get('predicted_next_action')) == 
       _normalize_label(info.get('target_next_action'))
])
anticipation_acc = correct_count / len(infos)
wandb.log({"step/anticipation_accuracy": anticipation_acc})
```

---

## 발견된 버그 및 수정 사항

### 🐛 Bug 1: predict_future 반환 타입 불일치

**문제**:
```python
# ❌ 이전: 리스트의 리스트 반환
return [
    [{"action": "a1", "duration": 0.3}, ...],  # 16개 딕셔너리
    [{"action": "a2", "duration": 0.2}, ...],
    ...
]
```

**수정**:
```python
# ✅ 수정: 단일 문자열 리스트 반환
return ["grasping", "reaching", ...]  # 각 프로세스당 1개
```

---

### 🐛 Bug 2: 리워드 계산 시 타입 에러

**문제**:
```python
# ❌ 이전: 리스트와 문자열 비교
predicted_next = [{"action": "grasping", ...}, ...]  # 리스트
target_next = "grasping"  # 문자열
if predicted_next == target_next:  # TypeError!
```

**수정**:
```python
# ✅ 수정: 문자열끼리 비교
predicted_next = "grasping"  # 문자열
target_next = "grasping"  # 문자열
if _normalize_label(predicted_next) == _normalize_label(target_next):  # OK!
```

---

### 🐛 Bug 3: 정확도 로깅 시 타입 에러

**문제**:
```python
# ❌ 이전: 리스트와 문자열 직접 비교
correct_count = sum([
    1 for info in infos 
    if info.get('predicted_next_action') == info.get('target_next_action')
])
# predicted_next_action이 리스트면 항상 False
```

**수정**:
```python
# ✅ 수정: normalize 후 비교
correct_count = sum([
    1 for info in infos 
    if _normalize_label(info.get('predicted_next_action', 'none')) == 
       _normalize_label(info.get('target_next_action', 'none'))
])
```

---

### 🐛 Bug 4: Alignment Reward 음수 가능성

**문제**:
```python
# ❌ 이전: CLIP similarity가 음수일 수 있음 (-1~1 범위)
alignment_reward = alignment_sim  # -1~1
reward = 0.3 * alignment_reward + 0.7 * task_reward  # 음수 가능!
```

**수정**:
```python
# ✅ 수정: 음수 제거
alignment_reward = max(0.0, alignment_sim)  # 0~1 범위로 제한
```

---

## 최종 데이터 흐름

```
┌─────────────────────────────────────────────────────────────┐
│ 1. FUTR: Visual Features → Coarse Labels                   │
│    Input:  [B, 16, 2048] (16 sampled frames)               │
│    Output: [B, 16] → ["walking", "walking", "reaching"...] │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. VLM: Coarse Labels + Images → Fine-grained Text         │
│    Input:  Coarse labels (5개 최근 것만)                    │
│    Output: 1개 통합 설명 텍스트                             │
│    "The person is walking forward with arms swinging..."    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. CLIP: Fine-grained Text → Embedding                     │
│    Input:  Text string                                      │
│    Output: [512] embedding                                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. FUTR: Visual + FG Embedding → Next Action               │
│    Input:  Visual [B,16,2048] + Context [B,16,512]         │
│    Output: [B, 16, n_class] → 첫 번째만 사용               │
│    Result: ["grasping"] (1개 문자열)                        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. Reward Calculation                                       │
│    Alignment: CLIP_sim(FG_text, current_action) = 0.85     │
│    Task:      (predicted=="grasping") == (target=="grasping") = 1.0 │
│    Final:     0.3 * 0.85 + 0.7 * 1.0 = 0.955               │
└─────────────────────────────────────────────────────────────┘
```

---

## 실행 전 체크리스트

### ✅ 필수 확인 사항

1. **데이터셋 경로**
   ```python
   # main.py
   dataset_root = os.path.abspath(os.path.expanduser(args.utkinect_root))
   # 확인: 해당 경로에 groundTruth/, features_img/ 폴더 존재?
   ```

2. **FUTR 체크포인트**
   ```python
   FUTR_MODEL_PATH = "/home/.../futr_joint_epoch_57.ckpt"
   # 확인: 파일 존재? 없으면 랜덤 초기화로 시작
   ```

3. **환경 변수**
   ```bash
   # wandb 로그인 필요
   wandb login
   ```

4. **Arguments**
   ```bash
   python3 main.py \
     --env_name utkinect \
     --utkinect_root /path/to/dataset \
     --use_wandb \
     --wandb_project your_project
   ```

---

## 예상 로그 출력

```
[Warmup] Starting FUTR Warmup for 500 steps...
[Warmup] Step 50/500 | Avg Loss: 2.3456
...
[Warmup] FUTR Warmup Complete.

[collect] update 1/1000, step 1/128
****** iteration number: 0 | FUTR Loss: 1.2345 ******

WandB Logs:
- step/reward: 0.65
- step/futr_loss: 1.23
- step/anticipation_accuracy: 0.45  ← 이게 보여야 함!
- train/value_loss: 0.12
- train/action_loss: 0.34
- eval/mean_episode_reward: 0.68
- eval/success_rate: 0.60
```

---

## 남은 잠재적 이슈

### ⚠️ 1. Fine-grained Text 품질
- VLM이 실제로 유용한 설명을 생성하는지 확인 필요
- 초반에는 무의미한 텍스트 생성 가능 → Warmup 중요

### ⚠️ 2. Context Projector 학습
- `context_projector`가 랜덤 초기화 상태
- FUTR가 fine-grained context를 활용하는 법을 학습하는 데 시간 필요

### ⚠️ 3. 리워드 스케일
- Alignment reward가 너무 낮으면 (예: 0.1) task reward에 압도될 수 있음
- 필요시 가중치 조정: `0.5 * alignment + 0.5 * task`

---

## 결론

✅ **모든 타입 불일치 해결**
✅ **Anticipation 검증 로직 수정**
✅ **리워드 계산 안정화**
✅ **로깅 개선**

**이제 `python3 main.py`로 실행 가능합니다!**
