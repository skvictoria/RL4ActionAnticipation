import os
import time
from collections import deque
import numpy as np
import torch
import transformers
import torch.optim as optim
from accelerate import Accelerator

# RL and Environment Imports
from a2c_ppo_acktr import algo
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import VLMPolicy, VLMValue
from a2c_ppo_acktr.storage import RolloutStorage

# Reward Calculation Import
from sentence_transformers import SentenceTransformer, util

# --- LLaVA Imports for Direct Loading ---
from transformers import AutoTokenizer, BitsAndBytesConfig
from llava.model import LlavaLlamaForCausalLM
from llava.model.language_model.llava_mistral import LlavaMistralForCausalLM
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria

# Import from your custom environment file
from utkinects_env import ACTION_LIST

import warnings
warnings.filterwarnings("ignore")

def main():
    args = get_args()

    # --- Basic Setup ---
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    log_dir = os.path.expanduser(args.save_dir)
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    # --- LLaVA Model Loading (Your Suggested Method) ---
    print("Loading LLaVA model using direct transformers import...")
    model_path = args.llava_model_path
    cache_dir = args.cache_dir

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)

    # 2. Load Model (with quantization options)
    # Note: Using device_map="auto" is crucial for multi-GPU or large models.
    if args.q8:
        print("Loading model with 8-bit quantization...")
        model_kwargs = {"load_in_8bit": True, "torch_dtype": torch.float16, "device_map": "auto"}
    elif args.q4:
        print("Loading model with 4-bit quantization...")
        q4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
        model_kwargs = {"quantization_config": q4_config, "device_map": "auto"}
    else:
        print("Loading model in default precision...")
        model_kwargs = {"torch_dtype": torch.float16, "device_map": "auto"}

    if 'mistral' in model_path.lower():
        llava_model = LlavaMistralForCausalLM.from_pretrained(model_path, cache_dir=cache_dir, **model_kwargs)
    else:
        llava_model = LlavaLlamaForCausalLM.from_pretrained(model_path, cache_dir=cache_dir, **model_kwargs)

    # 3. Load Vision Tower and get the Image Processor from it (Correct Way)
    vision_tower = llava_model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device=device, dtype=torch.float16)
    image_processor = vision_tower.image_processor

    print("LLaVA model and components loaded successfully.")
    # ---

    # --- Initialize Environment and RL Agent ---
    envs = make_vec_envs(
        args.env_name, args.seed, args.num_processes, args.gamma,
        log_dir, device, False,
        dataset_path=args.dataset_path
    )
    similarity_model = SentenceTransformer('all-MiniLM-L6-v2').to(device)
    print("Loading LLaVA model using direct transformers import...")
    model_path = args.llava_model_path
    cache_dir = args.cache_dir
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
    value_model = VLMValue(base=llava_model)
    prompt_text = (
        "You are an AI assistant for action recognition. "
        f"Your job is to classify the image into one of these categories of action: {', '.join(ACTION_LIST)}\n"
        "First, Observe the image. "
        "Second, provide a chain of thought for what detailed, fine-grained action is happening. Try to catch the object's hidden intention of the action as possible. "
        f"Then, classify the action into one of these categories: {', '.join(ACTION_LIST)}\n"
        "Format your response EXACTLY as follows:\n"
        "THOUGHT: [Your reasoning here].\n"
        "ACTION: [One of the action categories here]."
    )
    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + '\n' + prompt_text)
    conv.append_message(conv.roles[1], None)
    prompt_for_model = conv.get_prompt()
    INPUT_IDS = tokenizer_image_token(prompt_for_model, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').to(device)

    def text_to_action_index(text_action):
        try:
            # 여러 줄의 응답 중 마지막 줄에 액션이 있다고 가정
            action_str = text_action[0].strip().split('\n')[-1]
            action_str = action_str.split(' ')[-1].replace(' ', '')
            action_str = action_str.replace('.', '')
            return ACTION_LIST.index(action_str)
        except:
            return ACTION_LIST.index("UNDEFINED") # 파싱 실패 시 'UNDEFINED'로 처리
    
    projection_f = text_to_action_index
    actor_critic = VLMPolicy(
        tokenizer=tokenizer,
        image_processor=image_processor,
        value_model=value_model,
        args=args,
        INPUT_IDS=INPUT_IDS,
        projection_f=projection_f
    )
    actor_critic.to(device)
    optimizer = optim.Adam(actor_critic.parameters(), lr=args.init_lr, eps=args.eps)
    accelerator = Accelerator()
    device = accelerator.device
    agent = algo.PPO(
        actor_critic=actor_critic, optimizer=optimizer, accelerator=accelerator, clip_param=args.clip_param, ppo_epoch=args.ppo_epoch, mini_batch_size=args.mini_batch_size,
        value_loss_coef=args.value_loss_coef, entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm
    )
    rollouts = RolloutStorage(
        num_steps=args.num_steps,
        num_processes=args.num_processes,
        obs_shape=envs.observation_space.shape,
        action_space=envs.action_space,
        max_new_tokens=args.max_new_tokens
    )

    # --- Start Training ---
    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)
    episode_rewards = deque(maxlen=10)
    # 프로세스별로 현재 에피소드의 누적 보상 저장
    #running_episode_rewards = np.zeros(args.num_processes, dtype=np.float32)
    start = time.time()
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes

    for j in range(num_updates):
        for step in range(args.num_steps):
            with torch.no_grad():
                value, output_ids, action, action_log_prob, _ = accelerator.unwrap_model(actor_critic).act(rollouts.obs[step])

            # LLaVA 응답을 파싱하여 CoT 보상을 계산합니다.
            llava_response = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
            #print("llava response: ", llava_response)
            try:
                thought = llava_response.split("THOUGHT:")[1].split("ACTION:")[0].strip()
            except Exception:
                thought = ""
            
            if thought:
                action_text = ACTION_LIST[action] 
                print("action/ thought: ", action_text, thought)
                thought_embedding = similarity_model.encode(thought, convert_to_tensor=True)
                action_embedding = similarity_model.encode(action_text, convert_to_tensor=True)
                cot_reward = util.pytorch_cos_sim(thought_embedding, action_embedding).item()
            else:
                cot_reward = -1.0
            print(cot_reward)
            print(action)
            # 환경을 step하고, 보상을 CoT 보상으로 덮어씁니다.
            # action을 텐서 형태로 전달해야 합니다.
            obs, _, done, infos = envs.step(torch.tensor([action]).to(device))
            reward = torch.tensor([[cot_reward]], device=device)

            # # 결과 저장
            # for info in infos:
            #     print(info)
            #     if 'episode' in info.keys():
            #         print("episode in info")
            #         episode_rewards.append(info['episode']['r'])
            if isinstance(done, torch.Tensor):
                done_np = done.cpu().numpy()
            else:
                done_np = np.array(done, dtype=bool)
            done_np = np.atleast_1d(done_np)
            # reward는 여기선 모든 프로세스에 같은 cot_reward 하나만 있는 구조일 수도 있음
            # num_processes=1 이라면 단순히:
            #running_episode_rewards += cot_reward
            episode_rewards.append(cot_reward)
            # # 각 프로세스별 done 체크
            # for i, d in enumerate(done_np):
            #     print(i,d)
            #     if d:
            #         episode_rewards.append(running_episode_rewards[i])
            #         running_episode_rewards[i] = 0.0  # 다음 에피소드 시작

            masks = torch.FloatTensor([[0.0] if d else [1.0] for d in done_np])
            bad_masks = torch.FloatTensor([[1.0]])

            # insert(self, obs, output_ids, actions, action_log_probs, value_preds, rewards, masks, bad_masks)
            rollouts.insert(
                obs=obs,
                output_ids=output_ids,
                actions=torch.tensor([action]),
                action_log_probs=action_log_prob,
                value_preds=value,
                rewards=reward,
                masks=masks,
                bad_masks=bad_masks
            )
            # ---

        with torch.no_grad():
            next_value = accelerator.unwrap_model(actor_critic).get_value(rollouts.obs[-1])

        print("has_nan next_value:        ", torch.isnan(next_value).any().item())
        print("has_nan value_preds:       ", torch.isnan(rollouts.value_preds).any().item())
        print("has_nan rewards:           ", torch.isnan(rollouts.rewards).any().item())
        print("has_nan masks:             ", torch.isnan(rollouts.masks).any().item())
        print("has_nan bad_masks:         ", torch.isnan(rollouts.bad_masks).any().item())
        print("has_nan returns(before):   ", torch.isnan(rollouts.returns).any().item())

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.gae_lambda, args.use_proper_time_limits)

        print("has_nan returns(after):    ", torch.isnan(rollouts.returns).any().item())
    
        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        rollouts.after_update()

        if 1:#j % args.log_interval == 0 and len(episode_rewards) > 0:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                f"Updates {j}, Timesteps {total_num_steps}, FPS {int(total_num_steps / (end - start))}\n"
                f"Last {len(episode_rewards)} episodes: mean reward {np.mean(episode_rewards):.2f}\n"
                f"Losses: value {value_loss:.5f}, action {action_loss:.5f}, entropy {dist_entropy:.5f}\n"
            )

if __name__ == "__main__":
    main()