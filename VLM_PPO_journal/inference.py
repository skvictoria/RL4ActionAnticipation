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

# 기존 모듈 임포트 (main.py와 동일하게 유지)
from a2c_ppo_acktr import algo, utils, rl_utils
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import VLMPolicy, VLMValue
from a2c_ppo_acktr.llava_interface import init_pretrained_model, find_all_linear_names, load_lora_model
from a2c_ppo_acktr.rl_utils import get_prompt, text_projection, _clip_safe_tokenize

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token
from llava.model import LlavaLlamaForCausalLM
from llava.model.language_model.llava_mistral import LlavaMistralForCausalLM

import transformers
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer
import clip
import accelerate
from accelerate.state import AcceleratorState

# Joint Model 임포트
from joint_model import JointFUTR

def main():
    args = get_args()
    
    # 1. Accelerator 및 Device 설정 (main.py 방식)
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    
    # 2. LLaVA 모델 로딩 (main.py의 로직 100% 반영)
    model_path = args.model_path
    cache_dir = args.cache_dir
    print(f"Loading LLaVA from {model_path}...")

    if "lora" in model_path:
        base, tokenizer = load_lora_model(model_path, cache_dir=cache_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir, use_fast=False)
        if args.q8:
            print("8bit quantization")
            base = LlavaLlamaForCausalLM.from_pretrained(model_path, load_in_8bit=True, cache_dir=cache_dir)
        elif args.q4:
            q4_config = transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4'
            )
            print("4bit quantization")
            base = LlavaLlamaForCausalLM.from_pretrained(model_path, load_in_4bit=True, quantization_config=q4_config, cache_dir=cache_dir)
        else:
            base = LlavaLlamaForCausalLM.from_pretrained(model_path, cache_dir=cache_dir)

    base.config.max_length = 1024
    base, tokenizer = init_pretrained_model(base, tokenizer, pretrain_mm_adapter=args.pretrain_mm_adapter)
    image_processor = base.get_vision_tower().image_processor
    
    if args.use_lora:
        base_lora_config = LoraConfig(
            r=128,
            lora_alpha=256,
            target_modules=find_all_linear_names(base, args.train_vision),
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        base = get_peft_model(base, base_lora_config)
    
    value_model = VLMValue(base).to(device)
    
    # 3. Joint FUTR 모델 로드 (체크포인트)
    dataset_root = os.path.abspath(os.path.expanduser(args.utkinect_root))
    
    # 평가용 체크포인트 경로 (필요시 args로 관리 가능)
    EVAL_CKPT_PATH = "/home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/save_dir/utkinects/long/model/transformer/1/i3d_transcript/runs0/_20_30_50_erank_40p_64_latent_20251208/futr_joint_epoch_26.ckpt"
    
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
    
    envs = make_vec_envs(args.env_name, args.seed, 1, # 평가 시 프로세스 1개 고정
                         args.gamma, None, device, False, 1,
                         utkinect_config=utkinect_config)

    # 5. CLIP 모델 로드
    clip_model, _ = clip.load("ViT-B/32", device=device)
    clip_model.eval()

    # 6. Actor-Critic 초기화
    # 더미 프롬프트로 초기 INPUT_IDS 생성
    prompt_dummy = "test"
    INPUT_IDS_dummy = tokenizer_image_token(prompt_dummy, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
    projection_f = partial(text_projection, env_name=args.env_name)
    
    actor_critic = VLMPolicy(tokenizer=tokenizer,
                             image_processor=image_processor,
                             value_model=value_model,
                             projection_f=projection_f,
                             INPUT_IDS=INPUT_IDS_dummy,
                             args=args)
    
    # Accelerator 준비
    actor_critic = accelerator.prepare(actor_critic)
    actor_critic.eval()

    # === Inference Loop ===
    print("\nStarting Inference...")
    total_correct = 0
    total_samples = 0
    
    obs = envs.reset()
    infos = envs.get_current_infos()
    
    eval_steps = 200 
    
    for i in tqdm(range(eval_steps)):
        # --- [Step 1] FUTR: Coarse Prediction ---
        pred_hist_list = joint_model.predict_coarse(infos)
        predicted_history = pred_hist_list[0] if pred_hist_list else []
        
        # --- Prompt Generation (main.py 방식) ---
        qs = get_prompt(args.env_name, args.action_only_prompt, infos, predicted_history=predicted_history)
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        # --- [Step 2] VLM: Fine-grained Prediction ---
        with torch.no_grad():
            INPUT_IDS = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
            INPUT_IDS[INPUT_IDS == 0] = 259
            
            # act() 호출
            _, _, action, _, _, text_outputs = actor_critic.act(obs, INPUT_IDS=INPUT_IDS)
        
        # --- CLIP Embedding ---
        fg_tensor = None
        clean_txt = ""
        print(text_outputs)
        if text_outputs:
            try:
                clean_txt = text_outputs[0].split("thoughts")[-1].replace('"', '').replace(':', '').strip()
            except:
                clean_txt = text_outputs[0]
            print("--------------- this is clean text: -----------------")
            print(clean_txt)
            tokens = _clip_safe_tokenize(clean_txt, device)
            with torch.no_grad():
                fg_tensor = clip_model.encode_text(tokens)
            fg_tensor = fg_tensor.float()

        # --- [Step 3] FUTR: Future Prediction ---
        future_preds = joint_model.predict_future(infos, fg_tensor)
        print(future_preds)
        prediction = future_preds[0]
        
        # --- Ground Truth Comparison ---
        target = infos[0].get('target_next_action', 'UNDEFINED')
        norm_pred = prediction.strip().lower().replace(" ", "")
        norm_target = target.strip().lower().replace(" ", "")
        print(norm_pred, norm_target)
        
        if norm_pred == norm_target:
            total_correct += 1
        total_samples += 1
        
        # 다음 스텝으로 진행
        obs, _, done, infos = envs.step(action)
        
    # 결과 출력
    accuracy = (total_correct / total_samples) * 100 if total_samples > 0 else 0
    print(f"\n====== Inference Results ======")
    print(f"Total Samples: {total_samples}, Correct: {total_correct}, Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()