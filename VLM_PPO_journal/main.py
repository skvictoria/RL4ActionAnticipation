from patch import replace_llama_attn_with_xformers_attn
if replace_llama_attn_with_xformers_attn():
    print("using xformers")
else:
    print("using native attention")

import copy
import glob
import os
import time
from collections import deque
import sys
import re
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import algo, utils, rl_utils
from a2c_ppo_acktr.rl_utils import get_prompt, text_projection
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import VLMPolicy, VLMValue
from a2c_ppo_acktr.storage import RolloutStorage
from a2c_ppo_acktr.llava_interface import llava_evaluate, llava_generate
from a2c_ppo_acktr.llava_interface import init_pretrained_model, find_all_linear_names, load_lora_model

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token
from llava.model import LlavaLlamaForCausalLM
from llava.model.language_model.llava_mistral import LlavaMistralForCausalLM

import math
import random
from functools import partial
from typing import List, Optional

from train_rl import train, warmup_futr # [NEW] Import warmup function
from joint_model import JointFUTR

import clip
clip_model, _ = clip.load("ViT-B/32", device='cuda:0')
for param in clip_model.parameters():
    param.requires_grad = False

try:
    from accelerate.utils.memory import clear_device_cache
except (ImportError, AttributeError):
    import accelerate.utils.memory as accelerate_memory
    if not hasattr(accelerate_memory, "clear_device_cache"):
        def clear_device_cache():
            return None
        accelerate_memory.clear_device_cache = clear_device_cache

from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoImageProcessor
import transformers
from tqdm import tqdm
import accelerate
from accelerate.state import AcceleratorState
import warnings
warnings.filterwarnings("ignore")



def main():
    FUTR_MODEL_PATH = "/home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/save_dir/utkinects/long/model/transformer/1/i3d_transcript/runs0/_20_30_50_erank_40p_64_latent_20251208/futr_joint_epoch_26.ckpt"

    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    torch.set_num_threads(1)

    accelerator = accelerate.Accelerator(gradient_accumulation_steps=args.grad_accum_steps)
    device = accelerator.device
    model_device = device

    # initialization of llava
    model_path = args.model_path
    cache_dir = args.cache_dir
    print(model_path)

    if "lora" in model_path:
        base, tokenizer = load_lora_model(model_path, cache_dir=cache_dir)
        if args.q8 or args.q4:
            raise ValueError("Lora model does not support 8bit or 4bit quantization")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
        if args.q8:
            print("8bit quantization")
            if 'mistral' in model_path.lower():
                base =  LlavaMistralForCausalLM.from_pretrained(model_path, load_in_8bit=True, cache_dir=cache_dir)
            else:
                base = LlavaLlamaForCausalLM.from_pretrained(model_path, load_in_8bit=True, cache_dir=cache_dir)
        elif args.q4:
            q4_config = transformers.BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type='nf4'
                    )
            print("4bit quantization")
            if 'mistral' in model_path.lower():
                base =  LlavaMistralForCausalLM.from_pretrained(model_path, load_in_4bit=True, quantization_config=q4_config, cache_dir=cache_dir)
            else:
                base = LlavaLlamaForCausalLM.from_pretrained(model_path, load_in_4bit=True, quantization_config=q4_config, cache_dir=cache_dir)
        else:
            if 'mistral' in model_path.lower():
                base =  LlavaMistralForCausalLM.from_pretrained(model_path, cache_dir=cache_dir)
            else:
                base = LlavaLlamaForCausalLM.from_pretrained(model_path, cache_dir=cache_dir)

    use_grad_ckpt = True
    if use_grad_ckpt:
        if hasattr(base, "enable_input_require_grads"):
            base.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            base.get_input_embeddings().register_forward_hook(make_inputs_require_grad)


    base.config.max_length = 1024
    print("Model max context length:{}".format(base.config.max_length))
    base, tokenizer = init_pretrained_model(base, tokenizer, pretrain_mm_adapter = args.pretrain_mm_adapter)
    image_processor = base.get_vision_tower().image_processor

    base_lora_config = LoraConfig(
            r=128,
            lora_alpha=256,
            target_modules=find_all_linear_names(base,args.train_vision),
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
    if args.use_lora:
        base = get_peft_model(base, base_lora_config)
    value_model = VLMValue(base)
    value_model = value_model.to(model_device)

    utkinect_enabled = 'utkinect' in args.env_name.lower()
    utkinect_config = None
    if utkinect_enabled:
        dataset_root = os.path.abspath(os.path.expanduser(args.utkinect_root))
        if not os.path.isdir(dataset_root):
            raise FileNotFoundError(f"UTKinect root not found: {dataset_root}")
        utkinect_config = {
            "dataset_root": dataset_root,
            "split": args.utkinect_split,
            "history_window": args.utkinect_history,
            "frame_skip": args.utkinect_frame_skip,
        }

    # Initialize Joint FUTR Model
    joint_model = None
    start_epoch = 0
    if utkinect_enabled:
        # 경로가 존재하는지 확인 후 로드
        if 1:#not os.path.exists(FUTR_MODEL_PATH):
            print(f"Warning: FUTR_MODEL_PATH {FUTR_MODEL_PATH} does not exist. Using random initialization.")
            FUTR_MODEL_PATH = "/home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/save_dir/utkinects/long/model/transformer/1/i3d_transcript/runs0/_20_30_50_erank_40p_64_latent_20251226"

        else:
            # [NEW] 파일명에서 Epoch 번호 추출 (예: ...epoch_5.ckpt -> 5)
            match = re.search(r'epoch_(\d+)', os.path.basename(FUTR_MODEL_PATH))
            if match:
                loaded_epoch = int(match.group(1))
                start_epoch = loaded_epoch + 1
                print(f"[Main] Resuming training from epoch {start_epoch} (Loaded: {loaded_epoch})")
            else:
                print("[Main] Could not parse epoch from filename. Starting from 0.")
        # Initialize
        joint_model = JointFUTR(device, dataset_root, model_path=FUTR_MODEL_PATH, lr=1e-5)

    if "gym_cards" in args.env_name.lower():
        import gym_cards  # noqa: F401
        envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                             args.gamma, None, device, False, 1)
    elif utkinect_enabled:
        envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                             args.gamma, None, device, False, 1,
                             utkinect_config=utkinect_config)
    else:
        print("Environment not supported")
        exit(1)

    # [NEW] Warmup FUTR if enabled
    # This prevents collapse to NONE label by training on GT data first
    if joint_model is not None and start_epoch == 0:
        warmup_futr(args, envs, joint_model, num_steps=500)

    # Reset envs for main training loop
    obs = envs.reset()
    infos = envs.get_current_infos()
    if infos is None:
        infos = [{} for _ in range(args.num_processes)]
    
    # Initial Prediction for Prompt
    predicted_history = None
    if joint_model:
        # [FIX] Use predict_coarse instead of predict_batch
        pred_hist_list = joint_model.predict_coarse(infos)
        predicted_history = pred_hist_list[0] if pred_hist_list else []

    ## Inputing Prompt here
    qs = get_prompt(args.env_name, args.action_only_prompt, infos, predicted_history=predicted_history)
    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    print(prompt)

    INPUT_IDS = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
    INPUT_IDS[INPUT_IDS == 0] = 259 # 869: . (period), 29871: SPIECE, 259: whitespace

    projection_f = partial(text_projection, env_name=args.env_name)

    actor_critic = VLMPolicy(tokenizer=tokenizer,
                             image_processor=image_processor,
                             value_model=value_model,
                             projection_f=projection_f,
                             INPUT_IDS=INPUT_IDS,
                             args=args)
    optimizer = optim.Adam(actor_critic.value_model.parameters(), lr=args.init_lr, eps=args.eps, weight_decay=args.weight_decay)

    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.lr_max_steps, eta_min=args.end_lr)

    AcceleratorState().deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = 1

    actor_critic, optimizer, lr_scheduler = accelerator.prepare(actor_critic, optimizer, lr_scheduler)

    agent = algo.PPO(
            actor_critic,
            optimizer,
            accelerator,
            args.clip_param,
            args.ppo_epoch,
            args.mini_batch_size,
            args.value_loss_coef,
            args.entropy_coef,
            max_grad_norm=args.max_grad_norm)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space, args.max_new_tokens)

    _, output_ids, action, action_log_prob, action_tokens_log_prob, _ = actor_critic.act(obs, INPUT_IDS = INPUT_IDS)
    print("action:{}".format(action))
    print("action_log_prob:{}".format(action_log_prob))
    print("action_tokens_log_prob:{}".format(action_tokens_log_prob))

    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=args.eval_num_per_episode)
    episode_success_rate = deque(maxlen=args.eval_num_per_episode)
    episode_action_tokens_log_prob = deque(maxlen=args.eval_num_per_episode)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    if args.use_wandb:
        import wandb
        run_name = args.wandb_run + "-" + args.env_name
        wandb.init(project=args.wandb_project, name=run_name, group=run_name, config=args)

    print(qs)
    running_episode_rewards = torch.zeros(args.num_processes).flatten()

    num_explore = int(args.explore_portion*num_updates)
    prev_infos = copy.deepcopy(infos)

    if not os.path.exists(FUTR_MODEL_PATH):
        os.makedirs(FUTR_MODEL_PATH, exist_ok=True)
    
    for j in tqdm(range(start_epoch, num_updates)):
        train(args, actor_critic, prompt, tokenizer, rollouts, infos, envs, episode_rewards, 
              running_episode_rewards, episode_success_rate, episode_action_tokens_log_prob, 
              agent, lr_scheduler, start, j, num_updates, clip_model, joint_model=joint_model)
        if joint_model is not None and (j % 1 == 0):
            save_path = os.path.join(FUTR_MODEL_PATH.replace('futr_joint_epoch_26.ckpt', ''), f"futr_joint_epoch_{j}.ckpt")
            joint_model.save_model(save_path)

if __name__ == "__main__":
    main()