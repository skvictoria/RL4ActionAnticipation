import os
import time
import numpy as np
import torch
from collections import deque
from tqdm import tqdm
from functools import partial

# 기존 모듈 임포트
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import VLMPolicy, VLMValue
from a2c_ppo_acktr.llava_interface import init_pretrained_model, find_all_linear_names, load_lora_model
from a2c_ppo_acktr.rl_utils import get_prompt, text_projection, _clip_safe_tokenize
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token
from llava.model import LlavaLlamaForCausalLM
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer
import clip

# Joint Model 임포트
from joint_model import JointFUTR

def main():
    args = get_args()
    
    # [설정] 평가 모드로 설정
    args.num_processes = 1  # 평가 시에는 프로세스 1개로 순차 실행 권장
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 1. LLaVA 모델 초기화 (학습과 동일)
    print(f"Loading LLaVA from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, cache_dir=args.cache_dir)
    base = LlavaLlamaForCausalLM.from_pretrained(args.model_path, cache_dir=args.cache_dir)
    base.config.max_length = 1024
    base, tokenizer = init_pretrained_model(base, tokenizer, pretrain_mm_adapter=args.pretrain_mm_adapter)
    
    image_processor = base.get_vision_tower().image_processor
    
    # LoRA 설정 (학습된 가중치가 있다면 로드해야 함, 여기서는 구조만 잡음)
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
    
    # 2. Joint FUTR 모델 로드 (체크포인트)
    dataset_root = os.path.abspath(os.path.expanduser(args.utkinect_root))
    
    # [수정] 평가할 체크포인트 경로 입력
    EVAL_CKPT_PATH = "/home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/save_dir/utkinects/long/model/transformer/1/i3d_transcript/runs0/_20_30_50_erank_40p_64_latent_20251208/futr_joint_epoch_26.ckpt"
    
    print(f"Loading JointFUTR from {EVAL_CKPT_PATH}...")
    if not os.path.exists(EVAL_CKPT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {EVAL_CKPT_PATH}")
        
    joint_model = JointFUTR(device, dataset_root, model_path=EVAL_CKPT_PATH)
    joint_model.model.eval() # 평가 모드 전환

    # 3. 환경 설정 (Test Split 사용 권장)
    utkinect_config = {
        "dataset_root": dataset_root,
        "split": "test", # 평가용 Split (없으면 train/val 사용)
        "history_window": args.utkinect_history,
        "frame_skip": args.utkinect_frame_skip,
    }
    
    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, None, device, False, 1,
                         utkinect_config=utkinect_config)

    # 4. CLIP 모델 로드 (텍스트 임베딩용)
    clip_model, _ = clip.load("ViT-B/32", device=device)
    clip_model.eval()

    # 5. Actor-Critic 초기화
    prompt_dummy = "test"
    INPUT_IDS_dummy = tokenizer_image_token(prompt_dummy, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
    projection_f = partial(text_projection, env_name=args.env_name)
    
    actor_critic = VLMPolicy(tokenizer=tokenizer,
                             image_processor=image_processor,
                             value_model=value_model,
                             projection_f=projection_f,
                             INPUT_IDS=INPUT_IDS_dummy,
                             args=args)

    # === Inference Loop ===
    print("\nStarting Inference...")
    
    total_correct = 0
    total_samples = 0
    
    obs = envs.reset()
    infos = envs.get_current_infos()
    
    # 평가할 스텝 수 (전체 데이터셋 크기에 맞춰 조정)
    eval_steps = 200 
    
    for i in tqdm(range(eval_steps)):
        # --- [Step 1] FUTR: Coarse Prediction ---
        pred_hist_list = joint_model.predict_coarse(infos)
        predicted_history = pred_hist_list[0] if pred_hist_list else []
        
        # --- Prompt Generation ---
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
            
            # act() 호출 (temperature 낮춰서 결정론적으로)
            _, _, action, _, _, text_outputs = actor_critic.act(obs, INPUT_IDS=INPUT_IDS)
        
        # --- CLIP Embedding ---
        fg_tensor = None
        if text_outputs:
            try:
                # 텍스트 전처리 (Thoughts 등 제거)
                clean_txt = text_outputs[0].split("thoughts")[-1].replace('"', '').replace(':', '').strip()
            except:
                clean_txt = text_outputs[0]
                
            tokens = _clip_safe_tokenize(clean_txt, device)
            with torch.no_grad():
                fg_tensor = clip_model.encode_text(tokens) # [1, 512]

        # --- [Step 3] FUTR: Future Prediction ---
        # predict_future 함수는 joint_model.py에 구현되어 있어야 함
        future_preds = joint_model.predict_future(infos, fg_tensor)
        prediction = future_preds[0] # Batch size 1 가정
        
        # --- Ground Truth Comparison ---
        target = infos[0].get('target_next_action', 'UNDEFINED')
        
        # 정규화 후 비교
        norm_pred = prediction.strip().lower().replace(" ", "")
        norm_target = target.strip().lower().replace(" ", "")
        
        if norm_pred == norm_target:
            total_correct += 1
        total_samples += 1
        
        print(f"Step {i} | GT: {target} | Pred: {prediction} | VLM Thought: {clean_txt[:50]}...")
        
        # 다음 스텝으로 진행
        obs, _, done, infos = envs.step(action)
        
    # 결과 출력
    accuracy = (total_correct / total_samples) * 100 if total_samples > 0 else 0
    print(f"\n====== Inference Results ======")
    print(f"Total Samples: {total_samples}")
    print(f"Correct: {total_correct}")
    print(f"Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()