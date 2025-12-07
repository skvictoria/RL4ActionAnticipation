from patch import replace_llama_attn_with_xformers_attn
if replace_llama_attn_with_xformers_attn():
    print("using xformers")
else:
    print("using native attention")
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from a2c_ppo_acktr import utils, rl_utils
from a2c_ppo_acktr.rl_utils import get_prompt
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token
try:
    from accelerate.utils.memory import clear_device_cache  # noqa: F401
except (ImportError, AttributeError):
    import accelerate.utils.memory as accelerate_memory

    if not hasattr(accelerate_memory, "clear_device_cache"):
        def clear_device_cache():
            return None
        accelerate_memory.clear_device_cache = clear_device_cache
import transformers
from tqdm import tqdm
import accelerate
import warnings
warnings.filterwarnings("ignore")

def train(args, actor_critic, prompt, tokenizer, rollouts, infos, envs, episode_rewards, running_episode_rewards, episode_success_rate, episode_action_tokens_log_prob, agent, lr_scheduler, start, j, num_updates, clip_model):

    for step in range(args.num_steps):
        # Sample actions
        with torch.no_grad():
            INPUT_IDS = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
            INPUT_IDS[INPUT_IDS == 0] = 259 # 869: . (period), 29871: SPIECE, 259: whitespace
            value, output_id, action, action_log_prob, action_tokens_log_prob, text_outputs = actor_critic.act(
                    rollouts.obs[step], INPUT_IDS = INPUT_IDS)
        text_action = text_outputs[0] if text_outputs else ""
        prev_infos = copy.deepcopy(infos)
        obs, reward, done, infos = envs.step(action)
        print("semantic reward info: ", text_outputs)
        print("prev info: ", prev_infos)
        ## TODO: Add semantic reward with coarse action prediction
        semantic_reward = rl_utils.semantic_reward_from_text(
            text_outputs, prev_infos, args.env_name, clip_model, reward.device)
        reward = semantic_reward.to(reward.device)
        if step % 50 == 0 or step == args.num_steps - 1:
            print(f"[collect] update {j+1}/{num_updates}, step {step+1}/{args.num_steps}, action {text_action}")

        qs = get_prompt(args.env_name, args.action_only_prompt, infos)
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        masks = torch.FloatTensor(
            [[0.0] if done_ else [1.0] for done_ in done])

        running_episode_rewards += reward.flatten()
        for i, d, r in zip(range(args.num_processes), done, reward):
            print("num process: ", i, "done: ", d, "reward: ", r)
            if d:
                print("d----")
                episode_rewards.append(running_episode_rewards[i].item())
                if running_episode_rewards[i] > 0:
                    print("running episode resards more than 0")
                    episode_success_rate.append(1)
                else:
                    episode_success_rate.append(0)
                episode_action_tokens_log_prob.append(action_tokens_log_prob[i].item())
                running_episode_rewards[i] = 0
        # bad_mask is a legacy implementation of the storage.py file
        bad_masks = torch.FloatTensor(
            [[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])
        rollouts.insert(obs, output_id, action,
                        action_log_prob, value, reward, masks, bad_masks)

    print("****** iteration number:{} ******".format(j))
    print("prompt:{}".format(prompt))
    print("text_action:{}".format(text_action))
    print("current observation:{}".format(prev_infos))
    print("ground truth:{}".format(infos))
    print("action log prob:{}".format(action_log_prob))
    print("action tokens log prob:{}".format(action_tokens_log_prob))
    with torch.no_grad():
        next_value = actor_critic.get_value(
            rollouts.obs[-1]).detach()

    rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                args.gae_lambda, args.use_proper_time_limits)
    value_loss, action_loss, dist_entropy = agent.update(rollouts)
    lr_scheduler.step()
    print(f"[update] iteration {j+1}/{num_updates} complete. value_loss={value_loss:.4f}, action_loss={action_loss:.4f}")

    rollouts.after_update()
    if len(episode_rewards) > 1:
        print("episode rewards length more than 1")
        total_num_steps = (j + 1) * args.num_processes * args.num_steps
        end = time.time()

        print(
            "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.2f}/{:.2f}, min/max reward {:.2f}/{:.2f}, success_rate {:.2f}\n"
            .format(j, total_num_steps,
                    int(total_num_steps / (end - start)),
                    len(episode_rewards), np.mean(episode_rewards),
                    np.median(episode_rewards), np.min(episode_rewards),
                    np.max(episode_rewards), np.mean(episode_success_rate),
                    dist_entropy, value_loss, action_loss))
        if args.use_wandb:
            print("using wandb")
            wandb.log({"iteration": j,
                    "num_timesteps": total_num_steps,
                    "FPS": int(total_num_steps / (end - start)),
                    "episode_reward.mean": np.mean(episode_rewards),
                    "episode_reward.median": np.median(episode_rewards),
                    "episode_reward.min": np.min(episode_rewards),
                    "episode_reward.max": np.max(episode_rewards),
                    "episode_success_rate.mean": np.mean(episode_success_rate),
                    "episode_action_tokens_log_prob.mean": np.mean(episode_action_tokens_log_prob),
                    "distribution_entropy": dist_entropy,
                    "value.loss": value_loss,
                    "action.loss": action_loss,
                    "reward.max": rollouts.rewards.max().item(),
                    "reward.min": rollouts.rewards.min().item(),
                    "reward.mean": rollouts.rewards.mean().item(),
                    "reward.std": rollouts.rewards.std().item(),
                    "reward.median": rollouts.rewards.median().item(),
                    "return.max": rollouts.returns.max().item(),
                    "return.min": rollouts.returns.min().item(),
                    "return.mean": rollouts.returns.mean().item(),
                    "return.std": rollouts.returns.std().item(),
                    "value.max": rollouts.value_preds.max().item(),
                    "value.min": rollouts.value_preds.min().item(),
                    "value.mean": rollouts.value_preds.mean().item(),
                    "value.std": rollouts.value_preds.std().item(),})