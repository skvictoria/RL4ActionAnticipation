# Fine-grained Text 개선 방안

## 현재 구현 (Option B)

### 문제점
- **16개 coarse label → 1개 통합 설명**
- 정보 손실: 각 프레임별 세밀한 변화를 포착하기 어려움
- 예시:
  ```
  Coarse: ["walking", "walking", "reaching", "reaching", "grasping", ...]
  Fine-grained: "The person walks forward then reaches for an object"
  → 어느 프레임에서 walking이 끝나고 reaching이 시작되는지 불명확
  ```

### 현재 해결 방법
- 1개 통합 설명을 CLIP embedding으로 변환
- 이를 16번 복사해서 시퀀스로 사용: `[emb, emb, emb, ..., emb]` (16개)
- FUTR의 context로 전달

### 장점
- ✅ 빠름: VLM 추론 1번만
- ✅ 구현 간단
- ✅ 전체적인 행동 흐름 파악 가능

### 단점
- ❌ 시간적 세밀함 부족
- ❌ 각 프레임별 차이 표현 불가
- ❌ FUTR가 시간 정보를 context에서 얻기 어려움

---

## 개선 방안 (Option A) - 권장

### 방법 1: 16개 개별 설명 생성

#### 구현
```python
# 프롬프트 수정
qs = "You are analyzing 3 frames. For EACH coarse label, provide a fine-grained description."
qs += f"Coarse labels: {limited_history}"
qs += "Output format: {\"descriptions\": [\"desc1\", \"desc2\", ...]}"

# VLM 출력 파싱
output = json.loads(text_output)
descriptions = output["descriptions"]  # 16개 문자열 리스트

# 각각 CLIP embedding으로 변환
fg_embeddings = []
for desc in descriptions:
    emb = clip_model.encode_text(clip.tokenize(desc))
    fg_embeddings.append(emb)
fg_tensor = torch.stack(fg_embeddings)  # [16, 512]
```

#### 장점
- ✅ 각 프레임별 세밀한 설명
- ✅ 시간적 변화 명확히 표현
- ✅ FUTR가 더 풍부한 context 활용 가능

#### 단점
- ❌ VLM 출력이 길어짐 (토큰 수 증가)
- ❌ JSON 파싱 실패 가능성
- ❌ 약간 느림 (하지만 여전히 1번 추론)

---

### 방법 2: Sliding Window 방식

#### 구현
```python
# 16개 프레임을 4개 그룹으로 나눔
groups = [
    coarse_labels[0:4],   # "walking, walking, walking, walking"
    coarse_labels[4:8],   # "walking, reaching, reaching, reaching"
    coarse_labels[8:12],  # "reaching, grasping, grasping, grasping"
    coarse_labels[12:16], # "grasping, grasping, grasping, grasping"
]

# 각 그룹마다 설명 생성 (4번 VLM 추론)
descriptions = []
for group in groups:
    desc = vlm.generate(f"Describe: {group}")
    descriptions.append(desc)

# 각 설명을 4번씩 복사
fg_embeddings = []
for desc in descriptions:
    emb = clip_model.encode_text(clip.tokenize(desc))
    fg_embeddings.extend([emb] * 4)  # 4번 복사
fg_tensor = torch.stack(fg_embeddings)  # [16, 512]
```

#### 장점
- ✅ 시간적 변화 어느 정도 포착
- ✅ JSON 파싱 불필요
- ✅ 구현 중간 난이도

#### 단점
- ❌ VLM 추론 4번 (느림)
- ❌ 여전히 그룹 내 세밀함 부족

---

### 방법 3: Hierarchical Description (추천!)

#### 구현
```python
# 프롬프트: 전체 + 세부 설명 동시 요구
qs = f"Coarse labels: {limited_history}"
qs += '''
Provide:
1. Overall description: What is the person doing across all frames?
2. Key transitions: At which points do actions change?
3. Fine-grained details: Describe body movements for each distinct action.

Output format:
{
  "overall": "The person walks forward, then reaches for an object and grasps it",
  "transitions": [4, 8, 12],  // Frame indices where action changes
  "details": {
    "walking": "Arms swinging naturally, steady pace",
    "reaching": "Right arm extends forward, body leans slightly",
    "grasping": "Fingers close around object, wrist rotates"
  }
}
'''

# 파싱 및 임베딩 생성
output = json.loads(text_output)
overall_emb = clip_encode(output["overall"])
detail_embs = {action: clip_encode(desc) for action, desc in output["details"].items()}

# 각 프레임에 맞는 임베딩 할당
fg_embeddings = []
for i, coarse_label in enumerate(coarse_labels):
    # Overall + Detail 결합
    detail_emb = detail_embs.get(coarse_label, overall_emb)
    combined_emb = (overall_emb + detail_emb) / 2  # 평균
    fg_embeddings.append(combined_emb)
fg_tensor = torch.stack(fg_embeddings)  # [16, 512]
```

#### 장점
- ✅ 전체 맥락 + 세부 정보 모두 포착
- ✅ VLM 추론 1번만
- ✅ 시간적 변화 명확 (transitions)
- ✅ 각 행동별 세밀한 설명

#### 단점
- ❌ 프롬프트 복잡
- ❌ JSON 파싱 실패 가능성
- ❌ VLM이 복잡한 구조 생성 못할 수도

---

## 구현 우선순위

### Phase 1: 현재 유지 (Option B) ✅ 완료
- 1개 통합 설명 → 16번 복사
- 빠르게 전체 파이프라인 검증

### Phase 2: 방법 1 구현 (권장)
- 16개 개별 설명 생성
- JSON 파싱 추가
- Fallback: 파싱 실패 시 Option B로 복귀

### Phase 3: 방법 3 구현 (최종)
- Hierarchical description
- 더 풍부한 context
- 성능 최적화

---

## 코드 수정 가이드 (Phase 2)

### 1. 프롬프트 수정 (rl_utils.py)

```python
elif 'utkinect' in env_name.lower():
    # ... (기존 코드)
    
    qs = "You are analyzing a sequence of 3 representative RGB frames from a video. "
    qs += f"The coarse-level action labels are: {history_text}. "
    qs += "For EACH coarse label, provide a fine-grained description of body movements. "
    qs += "Your response should be a valid json: \n{\n "
    qs += "\"descriptions\": [\"detailed description for label 1\", \"detailed description for label 2\", ...]\n}"
```

### 2. VLM 출력 파싱 (train_rl.py)

```python
# Step 3에서 fg_embedding 생성 시
for txt in text_outputs:
    try:
        # JSON 파싱 시도
        parsed = json.loads(txt)
        descriptions = parsed.get("descriptions", [])
        
        if len(descriptions) == len(limited_history):
            # 성공: 각 설명을 임베딩으로 변환
            embs = []
            for desc in descriptions:
                tokens = _clip_safe_tokenize(desc, device)
                emb = clip_model.encode_text(tokens)
                embs.append(emb)
            fg_sequence = torch.stack(embs)  # [N, 512]
        else:
            # 개수 불일치: Fallback
            raise ValueError("Description count mismatch")
    except:
        # 파싱 실패: Fallback to Option B
        clean_txt = txt.split("thoughts")[-1].strip()
        tokens = _clip_safe_tokenize(clean_txt, device)
        emb = clip_model.encode_text(tokens)
        fg_sequence = emb.unsqueeze(0).repeat(16, 1)  # [16, 512]
    
    batch_fg_sequences.append(fg_sequence)
```

### 3. FUTR 입력 조정 (joint_model.py)

```python
# train_step에서 context 처리
if fg_embedding is not None:
    # fg_embedding: [B, 16, 512] (이미 시퀀스 형태)
    valid_fg_embed = torch.stack(valid_fg_list).to(self.device)
    # Repeat 불필요! 이미 16개
```

---

## 실험 계획

### Ablation Study

| 실험 | Fine-grained Text | 예상 MoC | 학습 속도 |
|------|-------------------|----------|-----------|
| Baseline | 없음 (Visual only) | 0.30 | 빠름 |
| Option B | 1개 통합 (복사) | 0.45 | 빠름 |
| Option A-1 | 16개 개별 | 0.60 | 중간 |
| Option A-3 | Hierarchical | 0.70 | 중간 |

### 평가 메트릭
- MoC (Mean over Classes)
- First-action accuracy
- Per-class recall
- VLM 생성 품질 (BLEU, ROUGE)

---

## 결론

**현재 구현 (Option B)**: 빠른 프로토타이핑에 적합
**권장 개선 (Option A-1)**: 성능 향상 기대
**최종 목표 (Option A-3)**: 최고 성능

단계적으로 구현하면서 각 단계의 성능을 비교하는 것을 추천합니다!
