import torch
import random
from difflib import SequenceMatcher
from typing import List, Optional
import clip
import wandb
import torch.nn.functional as F
from a2c_ppo_acktr.datasets.utkinect_constants import (
    UTKINECT_ACTIONS,
    UTKINECT_DISPLAY,
)


def _normalize_label(label: Optional[str]) -> str:
    if not isinstance(label, str):
        return ""
    return label.strip().replace(" ", "").lower()


def _needs_two_digit_guard(env_name: str) -> bool:
    return 'points' in env_name.lower()


def _get_action_list(env_name: str) -> List[str]:
    if env_name == 'gym_cards/NumberLine-v0':
        return ["-", "+"]
    if env_name == 'gym_cards/Blackjack-v0':
        return ["stand", "hit"]
    if env_name == 'gym_cards/EZPoints-v0':
        return ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
                       "+", "*", "="]
    if env_name == 'gym_cards/Points24-v0':
        return ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
                       "+", "-", "*", "/", "(", ")", "="]
    if 'utkinect' in env_name.lower():
        return UTKINECT_ACTIONS
    raise NotImplementedError("Action list not implemented for this env!")


def _extract_action_from_text(string: str, action_list: List[str], guard_ten: bool) -> Optional[str]:
    if not isinstance(string, str):
        return None
    content = string.lower()
    action_index = content.find('"action":')
    if action_index >= 0:
        content = content[action_index:]
    matches = []
    text = content
    if guard_ten and '10' in text:
        matches.append('10')
        text = text.replace('10', ' ')
    for action in action_list:
        if action == '10' and guard_ten:
            continue
        if action in text:
            matches.append(action)
    seen = []
    for action in matches:
        if action not in seen:
            seen.append(action)
    if len(seen) == 1 and seen[0] in action_list:
        return seen[0]
    return None


def decode_text_actions(text_actions: List[str], env_name: str) -> List[Optional[str]]:
    action_list = _get_action_list(env_name)
    guard_ten = _needs_two_digit_guard(env_name)
    return [_extract_action_from_text(text, action_list, guard_ten) for text in text_actions]

def _clip_safe_tokenize(text: str, device, max_words: int = 70):
    if text is None:
        text = ""
    text = str(text).strip()
    if not text:
        return clip.tokenize("").to(device)

    words = text.split()
    # 처음엔 max_words 또는 전체 길이 중 작은 쪽에서 시작
    k = min(max_words, len(words))

    while k > 0:
        candidate = " ".join(words[:k])
        try:
            tokens = clip.tokenize(candidate)
            return tokens.to(device)
        except RuntimeError as e:
            if "context length" in str(e):
                # 너무 길다는 에러면 단어 수를 더 줄여서 다시 시도
                k -= 10  # 한 번에 10단어씩 줄여보기
                continue
            else:
                # 다른 에러면 그대로 넘김
                raise

    # 혹시라도 다 실패하면 빈 문자열로 fallback
    return clip.tokenize("").to(device)

def semantic_reward_from_text(text_actions: List[str], infos, env_name: str, clip_model, device):
    if infos is None or 'utkinect' not in env_name.lower():
        return None

    clip_model = clip_model.to(device).float().eval()
    rewards = []

    # 각 프로세스(환경)별로 루프를 돌아야 합니다.
    for i, (full_text, info) in enumerate(zip(text_actions, infos)):
        # 1. 해당 프로세스의 텍스트에서 설명 부분 추출
        try:
            if '"thoughts": ' in full_text:
                desc = full_text.split('"thoughts": ')[1].split('\n')[0].replace('"', '').strip()
            elif '"fine-grained description corresponding to each frame": ' in full_text:
                desc = full_text.split('"fine-grained description corresponding to each frame": ')[1].split(']')[0].replace('"', '').strip()
            else:
                desc = full_text
        except:
            desc = full_text

        # 2. 예측 설명(Description) 임베딩
        with torch.no_grad():
            pred_tokens = _clip_safe_tokenize(desc, device)
            pred_features = clip_model.encode_text(pred_tokens)
            pred_features = F.normalize(pred_features, dim=-1)

            # 3. 정답 레이블(Target) 임베딩 
            # 단순히 레이블만 넣기보다 "A photo of [label]" 혹은 설명을 더해주는 것이 CLIP 성능에 좋습니다.
            target_label = _normalize_label(info.get("target_next_action") if info else "none")
            target_text = f"A person is {target_label}" # CLIP이 더 잘 이해하는 문장 형태로 변환
            target_tokens = clip.tokenize(target_text).to(device)
            target_features = clip_model.encode_text(target_tokens)
            target_features = F.normalize(target_features, dim=-1)

            # 4. 거리 계산 (Cosine Similarity 기반)
            # similarity가 1에 가까울수록(거리가 0에 가까울수록) 좋은 것
            similarity = (pred_features * target_features).sum()
            distance = 1.0 - similarity
            
            # 보상은 거리의 음수값 (-1.0 ~ 0.0 사이)
            rewards.append(-distance.item())

    if not rewards:
        return None
    return torch.tensor(rewards).unsqueeze(1)


def get_prompt(env_name, action_only, infos = None, predicted_history=None):
    """
        This function defines the prompt for the text-to-action task, depending on the environments
        env_name: determines the prompts for each environment
        info: additional information that can be added to the prompt, if none, then use the default prompt
    """
    if env_name == 'gym_cards/NumberLine-v0':
        qs = "You are playing a game called number line. You will see a target number and a current number in the image. "
        qs = qs + "And your goal is to move the current number closer to the target by choosing either adding or subtracting one to the current number. "
        qs = qs + "Your response should be a valid json file in the following format: \n{\n "
        if not action_only:
            qs = qs + "\"current number\": \"x\", \n"
            qs = qs + "\"target number\": \"x\", \n"
            qs = qs + "\"thoughts\": \"{first read out the current and target number, then think carefully about which action to choose}\", \n"
        qs = qs + "\"action\": \"-\" or \"+\" \n}"
    elif env_name == 'gym_cards/Blackjack-v0':
        qs = "You are a blackjack player. You are observing the current game state, you can choose between ['stand', 'hit']. "
        qs = qs + "Your response should be a valid json file in the following format: \n{\n "
        if not action_only:
            qs = qs + "\"thoughts\": \"{first describe your total points and the dealer's total points then think about which action to choose}\", \n"
        qs = qs + "\"action\": \"stand\" or \"hit\" \n}"
    elif env_name == 'gym_cards/EZPoints-v0':
        try:
            text_formula = ''.join(str(element) for element in infos[0]['Formula'])
        except:
            text_formula = ''
        qs = "You are an expert card game player. You are observing two cards in the image. "
        qs = qs + f"You are observing the current formula: {text_formula}. "
        qs = qs + "You can choose between ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '+', '*', '=']. "
        qs = qs + "The number or operator you choose will be appended to the current formula. "
        qs = qs + "Note that 'J', 'Q', and 'K' count as '10'. "
        qs = qs + "Your goal is to output a formula that evaluates to 12, and each number can only be used once. "
        qs = qs + "Your response should be a valid json file in the following format: \{\n"
        if not action_only:
            qs = qs + " \"cards\": [x, y], \n"
            qs = qs + f"\"current formula\": {text_formula}, \n"
            qs = qs + "\"thoughts\": {First check whether the current formula 'z' is complete. "
            qs = qs + "If the current formula 'z' is complete, output '='. "
            qs = qs + "Otherwise consider which number or operator should be appended to the current formula to make it equal 12.} \n"
        qs = qs + "\"action\": \"{number}\" or \"{operator}\" \n \}"
    elif env_name == 'gym_cards/Points24-v0':
        try:
            text_formula = ''.join(str(element) for element in infos[0]['Formula'])
        except:
            text_formula = ''
        qs = "You are an expert 24 points card game player. You are observing thee four cards in the image. "
        qs = qs + f"You are observing the current formula: {text_formula}. "
        qs = qs + "You can choose between ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '+', '-', '*', '/', '(', ')', '=']. "
        qs = qs + "The number or operator you choose will be appended to the current formula. "
        qs = qs + "Note that 'J', 'Q', and 'K' count as '10'. "
        qs = qs + "Your goal is to output a formula that evaluates to 24, and each number can only be used once. "
        qs = qs + "Your response should be a valid json file in the following format: \{\n"
        if not action_only:
            qs = qs + " \"cards\": [x, y, z, w], \n"
            qs = qs + f"\"current formula\": {text_formula}, \n"
            qs = qs + "\"thoughts\": {First check whether the current formula equals 24. "
            qs = qs + "If the current formula equals 24, output '='. "
            qs = qs + "Otherwise consider which number or operator should be appended to the current formula to make it equal 24.} \n"
        qs = qs + "\"action\": \"{number}\" or \"{operator}\" \n \}"
    elif 'utkinect' in env_name.lower():
        display_actions = list(UTKINECT_DISPLAY.values())
        history = []
        
        if predicted_history is not None:
            history = [UTKINECT_DISPLAY.get(_normalize_label(act), act) for act in predicted_history]
        elif infos and len(infos) > 0 and isinstance(infos[0], dict):
            raw_history = infos[0].get("action_history", [])
            history = [UTKINECT_DISPLAY.get(_normalize_label(act), act) for act in raw_history]
        history_text = ', '.join(history) if history else 'None'
        qs = "You are analyzing RGB frames that contain coarse-level label of human action sequences. "
        qs = qs + f"The coarse labels of the frames are: [{history_text}]. "
        qs = qs + "Generate the corresponding fine-grained description for each coarse-level action label."
        qs = qs + " Your response should be a valid json file in the following format: \n{\n "
        if not action_only:
            qs = qs + "\"fine-grained description corresponding to each frame\": [\"label_0\", \"label_1\", ...], \n"
            qs = qs + "\"thoughts\": \"{describe what the person is doing and reason about the corresponding fine-grained description}\", \n"
        #qs = qs + f"\"action\": \"one of {display_actions}\" \n}}"
    
    return qs

# Define the function that processes the list of strings according to the specified rules
def text_projection(text_actions: List[str], env_name):
    output_indices = []
    action_list = _get_action_list(env_name)
    guard_ten = _needs_two_digit_guard(env_name)
    for string in text_actions:
        matched_action = _extract_action_from_text(string, action_list, guard_ten)
        if matched_action is None:
            output_indices.append(random.randint(0, len(action_list) - 1))
        else:
            output_indices.append(action_list.index(matched_action))
    return torch.Tensor([output_indices]).long().reshape(-1, 1)
