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

    num_sampled_frames = 16 # INSIGHT 논문 방식: 고정된 시퀀스 길이

    for step in range(args.num_steps):
        
        # --- [Step 1] FUTR Predicts Coarse Labels (for Prompt) ---
        predicted_history = None
        if joint_model is not None and 'utkinect' in args.env_name.lower():
            pred_hist_list = joint_model.predict_coarse(infos)
            predicted_history = pred_hist_list[0] if pred_hist_list else []

        max_history = 8 
        limited_history = predicted_history[-max_history:] if predicted_history else []

        qs = get_prompt(args.env_name, args.action_only_prompt, infos, 
                        predicted_history=limited_history)
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
        
        prev_infos = copy.deepcopy(infos)
        
        # Environment Step
        obs, reward, done, infos = envs.step(action)
        
        # Calculate Reward
        semantic_reward = rl_utils.semantic_reward_from_text(
            text_outputs, prev_infos, args.env_name, clip_model, reward.device)
        reward = semantic_reward.to(reward.device)

        # --- [Step 3] Update Embedding Buffer & Joint Training ---
        futr_loss = 0.0
        if joint_model is not None:
            # 1. VLM 출력 텍스트를 CLIP 임베딩으로 일괄 변환 (Batch Processing)
            clip_model = clip_model.to(reward.device).eval()
            all_tokens = []
            for txt in text_outputs:
                try:
                    clean_txt = txt.split("thoughts")[-1].replace('"', '').replace(':', '').strip()
                except:
                    clean_txt = txt
                all_tokens.append(_clip_safe_tokenize(clean_txt, reward.device))
            
            all_tokens_tensor = torch.cat(all_tokens)
            with torch.no_grad():
                all_embs = clip_model.encode_text(all_tokens_tensor).detach().cpu() # [B, 512]

            batch_fg_sequences = [] 
            for i in range(args.num_processes):
                # 2. 이전 버퍼 가져오기 (없으면 초기화)
                if 'fg_buffer' not in prev_infos[i]:
                    prev_infos[i]['fg_buffer'] = []
                
                # 3. 현재 프레임 임베딩 추가
                current_emb = all_embs[i]
                prev_infos[i]['fg_buffer'].append(current_emb)

                # 4. Joint 학습을 위한 시퀀스 샘플링 (INSIGHT 방식)
                observed_len = len(prev_infos[i]['fg_buffer'])
                # np.linspace로 0부터 현재까지 num_sampled_frames만큼 균등 추출
                sample_indices = np.linspace(0, observed_len - 1, num_sampled_frames, dtype=int)
                sampled_fg = torch.stack([prev_infos[i]['fg_buffer'][idx] for idx in sample_indices])
                batch_fg_sequences.append(sampled_fg)

                # 5. 에피소드 종료 여부에 따른 버퍼 전송/초기화
                if done[i]:
                    infos[i]['fg_buffer'] = []
                else:
                    infos[i]['fg_buffer'] = prev_infos[i]['fg_buffer']

            # 6. 배치 단위로 FUTR 학습 진행
            fg_tensor_seq = torch.stack(batch_fg_sequences).to(reward.device) # [Batch, 16, 512]
            futr_loss = joint_model.train_step(prev_infos, fg_tensor_seq)

        # --- [Step 4] Rollout Storage & Logging ---
        if step % 10 == 0 or step == args.num_steps - 1:
            print(f"[collect] update {j+1}/{num_updates}, step {step+1}/{args.num_steps}")

        masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
        bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])
        
        running_episode_rewards += reward.flatten()
        for i, d, r in zip(range(args.num_processes), done, reward):
            if d:
                episode_rewards.append(running_episode_rewards[i].item())
                episode_success_rate.append(1 if running_episode_rewards[i] > 0 else 0)
                episode_action_tokens_log_prob.append(action_tokens_log_prob[i].item())
                running_episode_rewards[i] = 0
        
        rollouts.insert(obs, output_id, action,
                        action_log_prob, value, reward, masks, bad_masks)

    print(f"****** iteration number: {j} | FUTR Loss: {futr_loss:.4f} ******")
    
    with torch.no_grad():
        pred_hist_next = []
        if joint_model:
            pred_hist_next_list = joint_model.predict_coarse(infos)
            pred_hist_next = pred_hist_next_list[0] if pred_hist_next_list else []
            
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