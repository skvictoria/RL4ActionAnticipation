from patch import replace_llama_attn_with_xformers_attn
if replace_llama_attn_with_xformers_attn():
    print("using xformers")
else:
    print("using native attention")
import copy
import time
import numpy as np
import torch
from a2c_ppo_acktr import utils, rl_utils
from a2c_ppo_acktr.rl_utils import get_prompt, _clip_safe_tokenize
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token
import accelerate
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# [NEW] FUTR Warmup Function
def warmup_futr(args, envs, joint_model, num_steps=500):
    print(f"\n****** [Warmup] Starting FUTR Warmup for {num_steps} steps... ******")
    
    # Ensure envs are ready (get initial infos)
    infos = envs.get_current_infos()
    if infos is None:
        envs.reset()
        infos = envs.get_current_infos()

    total_loss = 0
    device = joint_model.device
    
    for step in range(num_steps):
        # Dummy action (UTKinect environment ignores action input anyway)
        action = torch.zeros((args.num_processes, 1)).long().to(device)
        
        # Step environment to get new frames and GT
        obs, reward, done, infos = envs.step(action)
        
        # Train FUTR using ONLY Visual features (Visual -> Coarse/Future)
        # fg_embedding is None, so it learns the base task first.
        loss = joint_model.train_step(infos, fg_embedding=None)
        total_loss += loss
        
        if (step + 1) % 50 == 0:
            print(f"[Warmup] Step {step+1}/{num_steps} | Avg Loss: {total_loss/50:.4f}")
            total_loss = 0
            
    print("****** [Warmup] FUTR Warmup Complete. Model weights initialized. ******\n")


def train(args, actor_critic, prompt, tokenizer, rollouts, infos, envs, episode_rewards, running_episode_rewards, episode_success_rate, episode_action_tokens_log_prob, agent, lr_scheduler, start, j, num_updates, clip_model, joint_model=None):

    for step in range(args.num_steps):
        
        # --- [Step 1] FUTR Predicts Coarse Labels (for Prompt) ---
        predicted_history = None
        if joint_model is not None and 'utkinect' in args.env_name.lower():
            # Use 'predict_coarse' (Correct method name)
            pred_hist_list = joint_model.predict_coarse(infos)
            predicted_history = pred_hist_list[0] if pred_hist_list else []

        # --- Generate Prompt with FUTR's Coarse Prediction ---
        qs = get_prompt(args.env_name, args.action_only_prompt, infos, predicted_history=predicted_history)
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        curr_prompt = conv.get_prompt()

        # --- [Step 2] VLM Predicts Fine-grained Labels ---
        with torch.no_grad():
            INPUT_IDS = tokenizer_image_token(curr_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
            INPUT_IDS[INPUT_IDS == 0] = 259 
            value, output_id, action, action_log_prob, action_tokens_log_prob, text_outputs = actor_critic.act(
                    rollouts.obs[step], INPUT_IDS = INPUT_IDS)
        
        text_action = text_outputs[0] if text_outputs else ""
        prev_infos = copy.deepcopy(infos)
        
        # Environment Step
        obs, reward, done, infos = envs.step(action)
        
        # Calculate Reward
        semantic_reward = rl_utils.semantic_reward_from_text(
            text_outputs, prev_infos, args.env_name, clip_model, reward.device)
        reward = semantic_reward.to(reward.device)
        
        if step % 50 == 0 or step == args.num_steps - 1:
            print(f"[collect] update {j+1}/{num_updates}, step {step+1}/{args.num_steps}")

        # --- [Step 3] Joint Training: FUTR Predicts Future ---
        futr_loss = 0.0
        if joint_model is not None:
            # 1. Encode VLM output
            fg_embeds = []
            clip_model = clip_model.to(reward.device).eval()
            for txt in text_outputs:
                # Basic cleaning
                try:
                    clean_txt = txt.split("thoughts")[-1].replace('"', '').replace(':', '').strip()
                except:
                    clean_txt = txt
                # print("-------------")
                # print("clean text: ", clean_txt)
                # print("-------------")
                tokens = _clip_safe_tokenize(clean_txt, reward.device)
                with torch.no_grad():
                    emb = clip_model.encode_text(tokens) 
                    fg_embeds.append(emb.squeeze(0))
            
            if fg_embeds:
                fg_tensor = torch.stack(fg_embeds) 
                # 2. Train FUTR
                futr_loss = joint_model.train_step(prev_infos, fg_tensor)

        # Store in Rollouts
        masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
        bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])
        
        running_episode_rewards += reward.flatten()
        for i, d, r in zip(range(args.num_processes), done, reward):
            if d:
                episode_rewards.append(running_episode_rewards[i].item())
                if running_episode_rewards[i] > 0:
                    episode_success_rate.append(1)
                else:
                    episode_success_rate.append(0)
                episode_action_tokens_log_prob.append(action_tokens_log_prob[i].item())
                running_episode_rewards[i] = 0
        
        rollouts.insert(obs, output_id, action,
                        action_log_prob, value, reward, masks, bad_masks)

    print(f"****** iteration number: {j} | FUTR Loss: {futr_loss:.4f} ******")
    
    with torch.no_grad():
        pred_hist_next = []
        if joint_model:
            pred_hist_next = joint_model.predict_coarse(infos)[0] 
            
        qs = get_prompt(args.env_name, args.action_only_prompt, infos, predicted_history=pred_hist_next)
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        next_prompt = conv.get_prompt()
        
        NEXT_INPUT_IDS = tokenizer_image_token(next_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
        NEXT_INPUT_IDS[NEXT_INPUT_IDS == 0] = 259
        
        next_value = actor_critic.get_value(rollouts.obs[-1], INPUT_IDS=NEXT_INPUT_IDS).detach()

    rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                args.gae_lambda, args.use_proper_time_limits)
    value_loss, action_loss, dist_entropy = agent.update(rollouts)
    lr_scheduler.step()
    
    rollouts.after_update()