# inference.py
from patch import replace_llama_attn_with_xformers_attn
if replace_llama_attn_with_xformers_attn():
    print("using xformers")
else:
    print("using native attention")

import os
import time
import numpy as np
import torch
from collections import deque
from tqdm import tqdm
from functools import partial
import re
import copy

# 기존 모듈 임포트
from a2c_ppo_acktr import algo, utils, rl_utils
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import VLMPolicy, VLMValue
from a2c_ppo_acktr.llava_interface import init_pretrained_model, find_all_linear_names, load_lora_model
from a2c_ppo_acktr.rl_utils import get_prompt, text_projection, _clip_safe_tokenize
from a2c_ppo_acktr.datasets.utkinect_constants import UTKINECT_ACTIONS
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token
from llava.model import LlavaLlamaForCausalLM
from llava.model.language_model.llava_mistral import LlavaMistralForCausalLM
from collections import Counter
import transformers
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer
import clip
import accelerate
from accelerate.state import AcceleratorState

# Joint Model 임포트
from joint_model import JointFUTR

def get_valid_gt_sequence(gt_path):
    """GT 파일을 읽어 유효한 라벨 시퀀스와 해당 프레임 번호 리스트를 반환합니다."""
    valid_labels = []
    valid_frame_indices = []
    if not os.path.exists(gt_path):
        return [], []
    
    with open(gt_path, "r") as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split(",")]
            if len(parts) < 2: continue
            img_path, label = parts[0], parts[1]
            
            # 라벨 정규화 및 유효성 체크
            label = label.strip().replace(" ", "").lower()
            if label == "undefined" or label not in UTKINECT_ACTIONS:
                continue
            
            # 파일명에서 숫자로 된 프레임 인덱스 추출
            basename = os.path.basename(img_path)
            digits = "".join(ch for ch in basename if ch.isdigit())
            frame_idx = int(digits) if digits else -1
            
            valid_labels.append(label)
            valid_frame_indices.append(frame_idx)
            
    return valid_labels, valid_frame_indices

def main():
    print(UTKINECT_ACTIONS)
    args = get_args()
    
    # 1. Accelerator 및 Device 설정
    accelerator = accelerate.Accelerator()
    device = accelerator.device

    VLM_LORA_PATH = "/home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/save_dir/utkinects/long/model/transformer/1/i3d_transcript/runs0/_20_30_50_erank_40p_64_latent_20251226/vlm_lora_epoch_75"
    
    # 2. LLaVA 모델 로딩 (main.py 로직 반영)
    args.env_name = "utkinect/eval"
    args.model_path = 'liuhaotian/llava-v1.5-7b'
    args.utkinect_root = "/home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/datasets/utkinect"
    args.cache_dir= "/home/hice1/skim3513/RL4ActionAnticipation/hf_cache"
    args.use_lora = True
    args.train_vision = "all"
    EVAL_CKPT_PATH = "/home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/save_dir/utkinects/long/model/transformer/1/i3d_transcript/runs0/_20_30_50_erank_40p_64_latent_20251226/futr_joint_epoch_76.ckpt"

    model_path = args.model_path
    cache_dir = args.cache_dir
    print(f"Loading LLaVA from {model_path}...")

    if VLM_LORA_PATH:
        base, tokenizer = load_lora_model(VLM_LORA_PATH, model_base=model_path, cache_dir=cache_dir)
    elif "lora" in model_path:
        base, tokenizer = load_lora_model(model_path, cache_dir=cache_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir, use_fast=False)
        if args.q8:
            base = LlavaLlamaForCausalLM.from_pretrained(model_path, load_in_8bit=True, cache_dir=cache_dir)
        elif args.q4:
            q4_config = transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4'
            )
            base = LlavaLlamaForCausalLM.from_pretrained(model_path, load_in_4bit=True, quantization_config=q4_config, cache_dir=cache_dir)
        else:
            base = LlavaLlamaForCausalLM.from_pretrained(model_path, cache_dir=cache_dir)

    base.config.max_length = 1024
    base, tokenizer = init_pretrained_model(base, tokenizer, pretrain_mm_adapter=args.pretrain_mm_adapter)
    image_processor = base.get_vision_tower().image_processor
    
    if args.use_lora:
        base_lora_config = LoraConfig(
            r=128, lora_alpha=256,
            target_modules=find_all_linear_names(base, args.train_vision),
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        )
        base = get_peft_model(base, base_lora_config)
    
    value_model = VLMValue(base).to(device)
    
    # 3. Joint FUTR 모델 로드 (학습된 체크포인트 경로 설정)
    dataset_root = os.path.abspath(os.path.expanduser(args.utkinect_root))
    # TODO: 실제 평가할 체크포인트 경로로 수정하세요.
        
    print(f"Loading JointFUTR from {EVAL_CKPT_PATH}...")
    joint_model = JointFUTR(device, dataset_root, model_path=EVAL_CKPT_PATH)
    joint_model.model.eval()

    # 4. 환경 설정
    utkinect_config = {
        "dataset_root": dataset_root,
        "split": args.utkinect_split, 
        "history_window": args.utkinect_history,
        "frame_skip": args.utkinect_frame_skip,
    }
    
    # 평가 시에는 단일 프로세스(num_processes=1) 사용
    envs = make_vec_envs(args.env_name, args.seed, 1, 
                         args.gamma, None, device, False, 1,
                         utkinect_config=utkinect_config)

    # 5. CLIP 모델 로드 (Text Encoder 용)
    clip_model, _ = clip.load("ViT-B/32", device=device)
    clip_model.eval()

    # 6. Actor-Critic 초기화
    projection_f = partial(text_projection, env_name=args.env_name)
    actor_critic = VLMPolicy(tokenizer=tokenizer,
                             image_processor=image_processor,
                             value_model=value_model,
                             projection_f=projection_f,
                             INPUT_IDS=None, # 아래 루프에서 동적으로 생성
                             args=args)
    
    actor_critic = accelerator.prepare(actor_critic)
    actor_critic.eval()

    # === Inference Loop ===
    print("\nStarting Inference...")
    total_correct = 0
    total_samples = 0
    
    obs = envs.reset()
    infos = envs.get_current_infos()
    
    # train_rl.py의 멀티 프레임 로직을 위한 관찰 버퍼
    obs_buffer = [obs] 
    
    eval_steps = 200 # 평가할 스텝 수
    fg_buffer = []
    num_sampled_frames = 16
    for step in tqdm(range(eval_steps)):
        # --- [Step 1] FUTR: Coarse History Prediction ---
        pred_hist_list = joint_model.predict_coarse(infos)
        predicted_history = pred_hist_list[0] if pred_hist_list else []
        
        # History 길이 제한 (train_rl.py 로직 반영)
        max_history = 8
        limited_history = predicted_history[-max_history:] if predicted_history else []

        # 프롬프트 생성 (3개의 이미지 토큰 사용 - train_rl.py 방식)
        qs = get_prompt(args.env_name, args.action_only_prompt, infos, predicted_history=limited_history)
        qs = (DEFAULT_IMAGE_TOKEN + "\n") * 3 + qs
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        # --- [Step 2] VLM: Multi-Frame Fine-grained Prediction ---
        with torch.no_grad():
            INPUT_IDS = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
            INPUT_IDS[INPUT_IDS == 0] = 259 # whitespace 처리

            # 3프레임 샘플링 (0.5, 0.75, 1.0 지점)
            curr_idx = len(obs_buffer) - 1
            history_indices = [int(curr_idx * 0.5), int(curr_idx * 0.75), curr_idx]
            
            # [1, 3, C, H, W] 형태로 구성
            multi_obs = torch.stack([obs_buffer[i] for i in history_indices], dim=1).to(device)
            
            # act() 호출 (VLM 추론)
            _, _, action, _, _, text_outputs = actor_critic.act(multi_obs, INPUT_IDS=INPUT_IDS)
        
        # --- [Step 3] CLIP Embedding (Text to Feature) ---
        fg_input = None
        if text_outputs:
            raw_text = text_outputs[0]
            # train_rl.py의 클리닝 로직 적용
            try:
                clean_txt = raw_text.split("thoughts")[-1].replace('"', '').replace(':', '').replace('}', '').strip()
            except:
                clean_txt = raw_text.strip()
            
            tokens = _clip_safe_tokenize(clean_txt, device)
            with torch.no_grad():
                current_fg = clip_model.encode_text(tokens).float() # [1, 512]
            
            # 버퍼에 현재 임베딩 추가 (CPU로 옮겨서 메모리 절약)
            fg_buffer.append(current_fg.squeeze(0).cpu())
            
            # 학습 시와 동일하게 16개 프레임 샘플링 (INSIGHT 방식)
            observed_len = len(fg_buffer)
            indices = np.linspace(0, observed_len - 1, num_sampled_frames, dtype=int)
            sampled_fg = torch.stack([fg_buffer[idx] for idx in indices]).to(device) # [16, 512]
            fg_input = sampled_fg.unsqueeze(0) # [1, 16, 512] (Batch dim 추가)

        # --- [Step 4] FUTR: Future Action Anticipation ---
        # VLM이 준 fine-grained context(fg_input)를 사용하여 미래 예측
        future_preds_batch = joint_model.predict_future(infos, fg_input)
        preds = future_preds_batch[0] if future_preds_batch else []
        
        # --- [Step 5] Ground Truth Comparison (Duration-aware) ---
        info = infos[0]
        seq_id = info.get('sequence_id')
        curr_frame = info.get('frame_index', 0) # 파일명에서 온 절대 프레임 번호 (예: 500)
        
        gt_file = os.path.join(joint_model.gt_path, f"{seq_id}.txt")
        # [수정] 절대 프레임 번호가 아닌, 유효 시퀀스 내의 상대 인덱스를 찾습니다.
        valid_labels, valid_indices = get_valid_gt_sequence(gt_file)
        
        try:
            # 현재 frame_index가 유효 리스트의 몇 번째(index)인지 찾음
            current_seq_idx = valid_indices.index(curr_frame)
            # 미래 GT는 해당 인덱스 이후의 모든 라벨
            future_gt = valid_labels[current_seq_idx + 1:]
        except ValueError:
            # 현재 프레임이 GT 시퀀스에 없는 경우 (건너뜀)
            future_gt = []

        future_total_len = len(future_gt)
        if future_total_len > 0:
            print(f"\n--- Step {step} | Frame {curr_frame} Analysis ---")
            accumulated_ratio = 0.0
            
            for q_idx, p in enumerate(preds):
                if p['action'] == "UNDEFINED" or p['action'] == joint_model.pad_idx: continue
                
                # 예측된 duration(0~1)을 미래 시퀀스 길이에 곱해 구간 계산
                start_offset = int(accumulated_ratio * future_total_len)
                duration_frames = int(p['duration'] * future_total_len)
                end_offset = min(start_offset + duration_frames, future_total_len)
                
                if start_offset >= future_total_len: break
                
                # 해당 구간의 GT 행동들 추출
                gt_segment = future_gt[start_offset:end_offset]
                if not gt_segment: continue
                
                # 구간 내 가장 많이 등장한 행동(Majority Vote)과 비교
                
                most_common_gt = Counter(gt_segment).most_common(1)[0][0]
                
                norm_pred = p['action'].lower().replace(" ", "")
                is_correct = (norm_pred == most_common_gt)
                
                if is_correct:
                    total_correct += 1
                total_samples += 1
                # inference.py의 비교 루프 내부
                print(f"DEBUG: Step {step} Q{q_idx} | Pred: {norm_pred} | GT: {most_common_gt} | Match: {is_correct}")
                print(f" Q{q_idx}: [Pred] {p['action']:<15} | [GT] {most_common_gt:<15} | Match: {is_correct}")
                
                accumulated_ratio += p['duration']
                if accumulated_ratio >= 1.0: break
        
        # if step % 10 == 0:
        #     print(f"\n[Step {step}] VLM Output: {text_outputs[0]}")
        #     print(f"Prediction: {prediction} | GT: {target}")
        
        # 환경 스텝 진행
        obs, _, done, infos = envs.step(action)
        
        # 버퍼 업데이트 및 에피소드 종료 처리
        if done[0]:
            obs_buffer = [obs]
            fg_buffer = []
        else:
            obs_buffer.append(obs)
        
    # 최종 결과 출력
    accuracy = (total_correct / total_samples) * 100 if total_samples > 0 else 0
    print(f"\n====== Inference Results ======")
    print(f"Total Samples: {total_samples}, Correct: {total_correct}, Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()