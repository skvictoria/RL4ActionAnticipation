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

def semantic_reward_from_text(text_actions: List[str], infos, env_name: str, clip_model, device, 
                              joint_model=None, prev_infos=None):
    """
    Compute reward based on:
    1. How well fine-grained text describes current coarse labels (alignment reward)
    2. How much it improves action anticipation accuracy (task reward)
    """
    if infos is None or 'utkinect' not in env_name.lower():
        return None

    clip_model = clip_model.to(device).float().eval()
    rewards = []

    for i, (full_text, info) in enumerate(zip(text_actions, infos)):
        reward = 0.0
        
        # === Part 1: Alignment Reward (현재 coarse label과의 일치도) ===
        try:
            if '"thoughts": ' in full_text:
                desc = full_text.split('"thoughts": ')[1].split('}')[0].replace('"', '').strip()
            elif '"fine-grained description corresponding to each frame": ' in full_text:
                desc = full_text.split('"fine-grained description corresponding to each frame": ')[1].split(']')[0].replace('"', '').strip()
            else:
                desc = full_text
        except:
            desc = full_text

        with torch.no_grad():
            pred_tokens = _clip_safe_tokenize(desc, device)
            pred_features = clip_model.encode_text(pred_tokens)
            pred_features = F.normalize(pred_features, dim=-1)

            # Current coarse label (not future!)
            current_label = _normalize_label(info.get("current_action", "none"))
            current_text = f"A person is {current_label}"
            current_tokens = clip.tokenize(current_text).to(device)
            current_features = clip_model.encode_text(current_tokens)
            current_features = F.normalize(current_features, dim=-1)

            # Alignment similarity (0~1 range)
            alignment_sim = (pred_features * current_features).sum().item()
            alignment_reward = alignment_sim  # 0~1 range
        
        # === Part 2: Task Reward (Action Anticipation Accuracy) ===
        task_reward = 0.0
        if joint_model is not None and prev_infos is not None:
            # Check if anticipation was correct
            predicted_next = info.get("predicted_next_action", "none")
            target_next = info.get("target_next_action", "none")
            
            if _normalize_label(predicted_next) == _normalize_label(target_next):
                task_reward = 1.0  # Correct anticipation
            else:
                task_reward = 0.0  # Wrong anticipation
        
        # === Combined Reward ===
        # Weight: 0.3 for alignment, 0.7 for task performance
        reward = 0.3 * alignment_reward + 0.7 * task_reward
        rewards.append(reward)

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
        
        qs = "You are analyzing a sequence of 3 representative RGB frames from a video showing human actions. "
        qs += "Frame 1 is from the middle of the elapsed time, Frame 2 is from the 3/4 point, and Frame 3 is the current frame. "
        qs += f"The coarse-level action labels for the observed sequence are: [{history_text}]. "
        qs += "Your task is to generate a detailed, fine-grained description that explains HOW the person is performing these actions. "
        qs += "Focus on body movements, posture, hand positions, and motion patterns that characterize each action. "
        qs += "This description will be used to predict the next action the person will perform. "
        qs = qs + "Your response should be a valid json file in the following format: \n{\n "
        if not action_only:
            qs = qs + "\"thoughts\": \"{Provide a detailed, fine-grained description of the observed actions. "
            qs = qs + "Describe specific body movements, gestures, and motion patterns visible across the 3 frames. "
            qs = qs + "Be concrete and descriptive, e.g., 'The person extends their right arm forward while bending their knees' "
            qs = qs + "rather than just repeating the coarse label.}\"\n}"
        else:
            qs = qs + "\"thoughts\": \"{detailed fine-grained description}\"\n}"
    
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
