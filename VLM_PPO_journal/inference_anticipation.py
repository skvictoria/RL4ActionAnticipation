"""
Action Anticipation Inference Script

학습된 VLM + FUTR 모델을 불러와서 action anticipation 수행
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm

# Add project paths
sys.path.insert(0, os.path.dirname(__file__))

from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import VLMPolicy, VLMValue
from a2c_ppo_acktr.llava_interface import load_lora_model
from a2c_ppo_acktr import rl_utils
from joint_model import JointFUTR

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.model import LlavaLlamaForCausalLM
from llava.model.language_model.llava_mistral import LlavaMistralForCausalLM

from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer
import transformers
import clip
import re


def load_trained_models(args, device):
    """학습된 VLM과 FUTR 모델 로드"""
    
    print("=" * 80)
    print("Loading trained models...")
    print("=" * 80)
    
    # 1. Load LLaVA base model
    model_path = args.model_path
    cache_dir = args.cache_dir
    
    if "lora" in model_path:
        base, tokenizer = load_lora_model(model_path, cache_dir=cache_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
        if 'mistral' in model_path.lower():
            base = LlavaMistralForCausalLM.from_pretrained(model_path, cache_dir=cache_dir)
        else:
            base = LlavaLlamaForCausalLM.from_pretrained(model_path, cache_dir=cache_dir)
    
    base.config.max_length = 1024
    image_processor = base.get_vision_tower().image_processor
    
    # 2. Load trained LoRA weights for VLM
    if args.vlm_checkpoint:
        print(f"\nLoading VLM checkpoint from: {args.vlm_checkpoint}")
        checkpoint = torch.load(args.vlm_checkpoint, map_location=device)
        
        # LoRA 설정
        base_lora_config = LoraConfig(
            r=128,
            lora_alpha=256,
            target_modules=["q_proj", "v_proj"],  # 기본 설정
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        base = get_peft_model(base, base_lora_config)
        
        # Load state dict
        if 'actor_critic' in checkpoint:
            base.load_state_dict(checkpoint['actor_critic'], strict=False)
        elif 'model_state_dict' in checkpoint:
            base.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            base.load_state_dict(checkpoint, strict=False)
        
        print("✓ VLM checkpoint loaded successfully")
    else:
        print("⚠ No VLM checkpoint provided, using pretrained weights only")
    
    base = base.to(device).eval()
    
    # 3. Create value model (not needed for inference, but kept for compatibility)
    value_model = VLMValue(base)
    value_model = value_model.to(device).eval()
    
    # 4. Load CLIP model
    print("\nLoading CLIP model...")
    clip_model, _ = clip.load("ViT-B/32", device=device)
    clip_model = clip_model.float().eval()
    for param in clip_model.parameters():
        param.requires_grad = False
    print("✓ CLIP model loaded")
    
    # 5. Load FUTR model
    print("\nLoading FUTR model...")
    dataset_root = os.path.abspath(os.path.expanduser(args.utkinect_root))
    joint_model = JointFUTR(device, dataset_root, model_path=args.futr_checkpoint, lr=1e-6)
    joint_model.model.eval()
    print("✓ FUTR model loaded")
    
    return base, tokenizer, image_processor, value_model, clip_model, joint_model


def generate_fine_grained_descriptions(vlm_model, tokenizer, image_processor, obs, coarse_labels, device, args):
    """VLM으로 fine-grained descriptions 생성"""
    
    # Prepare prompt
    segment_labels = coarse_labels[-4:] if len(coarse_labels) >= 4 else coarse_labels
    
    qs = f"The observed sequence is divided into temporal segments: {segment_labels}. "
    qs += "Generate fine-grained descriptions for each segment. "
    qs += '''Your response should be a valid json:
{
  "segment_descriptions": ["desc1", "desc2", "desc3", "desc4"]
}
'''
    
    qs = (DEFAULT_IMAGE_TOKEN + "\n") * 3 + qs
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    INPUT_IDS = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
    INPUT_IDS[INPUT_IDS == 0] = 259
    INPUT_IDS = INPUT_IDS.to(device)
    
    # Generate
    with torch.no_grad():
        # obs shape: [1, 3, C, H, W] (3 frames)
        output_ids = vlm_model.generate(
            INPUT_IDS,
            images=obs.to(device),
            do_sample=True,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
        )
    
    text_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return text_output


def parse_and_embed_descriptions(text_output, clip_model, device):
    """4개 segment descriptions 파싱 및 CLIP embedding 생성"""
    
    try:
        # JSON 파싱
        json_start = text_output.find('{')
        json_end = text_output.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            json_str = text_output[json_start:json_end]
            json_str = re.sub(r'\\(?!["\\/bfnrtu])', r'', json_str)
            parsed = json.loads(json_str)
            segment_descs = parsed.get("segment_descriptions", [])
            
            if len(segment_descs) == 4:
                # 4개 설명을 각각 embedding으로 변환
                segment_embs = []
                for desc in segment_descs:
                    tokens = rl_utils._clip_safe_tokenize(str(desc), device)
                    with torch.no_grad():
                        emb = clip_model.encode_text(tokens).detach().cpu()
                        if emb.dim() > 1:
                            emb = emb.squeeze(0)
                    segment_embs.append(emb)
                
                # 각 embedding을 4번씩 복사 → 16개 시퀀스
                fg_sequence = []
                for emb in segment_embs:
                    fg_sequence.extend([emb] * 4)
                fg_sequence = torch.stack(fg_sequence)  # [16, 512]
                
                return fg_sequence, segment_descs
            else:
                raise ValueError(f"Expected 4 descriptions, got {len(segment_descs)}")
        else:
            raise ValueError("No JSON found")
    
    except Exception as e:
        print(f"⚠ JSON parsing failed: {e}")
        # Fallback: 1개 통합 설명
        clean_txt = text_output.split("thoughts")[-1].replace('"', '').replace(':', '').strip()
        tokens = rl_utils._clip_safe_tokenize(clean_txt, device)
        with torch.no_grad():
            emb = clip_model.encode_text(tokens).detach().cpu()
            if emb.dim() > 1:
                emb = emb.squeeze(0)
        fg_sequence = emb.unsqueeze(0).repeat(16, 1)
        
        return fg_sequence, [clean_txt]


def predict_future_actions(joint_model, infos, fg_embeddings):
    """FUTR로 미래 행동 예측"""
    
    # fg_embeddings: [B, 16, 512]
    predicted_sequences = joint_model.predict_future(infos, fg_embeddings)
    
    return predicted_sequences


def evaluate_on_dataset(args, models, device):
    """전체 데이터셋에 대해 평가"""
    
    base, tokenizer, image_processor, value_model, clip_model, joint_model = models
    
    # Create environment
    utkinect_config = {
        "dataset_root": os.path.abspath(os.path.expanduser(args.utkinect_root)),
        "split": args.utkinect_split,
        "history_window": args.utkinect_history,
        "frame_skip": args.utkinect_frame_skip,
    }
    
    envs = make_vec_envs(
        args.env_name, args.seed, 1,  # num_processes=1 for inference
        args.gamma, None, device, False, 1,
        utkinect_config=utkinect_config
    )
    
    # Reset environment
    obs = envs.reset()
    infos = envs.get_current_infos()
    
    results = []
    total_moc = 0.0
    total_first_acc = 0.0
    num_samples = 0
    
    print("\n" + "=" * 80)
    print("Starting inference...")
    print("=" * 80)
    
    for step in tqdm(range(args.num_inference_steps)):
        # 1. Get coarse labels from FUTR
        pred_hist_list = joint_model.predict_coarse(infos)
        predicted_history = pred_hist_list[0] if pred_hist_list else []
        
        # 2. Sample 3 frames for VLM
        if step < 3:
            history_indices = [0, max(0, step-1), step]
        else:
            history_indices = [int(step * 0.5), int(step * 0.75), step]
        
        # Note: In real inference, you'd need to maintain a frame buffer
        # For simplicity, we use current obs repeated
        multi_obs = obs.unsqueeze(0).repeat(3, 1, 1, 1, 1)  # [3, 1, C, H, W]
        multi_obs = multi_obs.transpose(0, 1)  # [1, 3, C, H, W]
        
        # 3. Generate fine-grained descriptions
        text_output = generate_fine_grained_descriptions(
            base, tokenizer, image_processor, multi_obs, 
            predicted_history, device, args
        )
        
        # 4. Parse and embed
        fg_sequence, segment_descs = parse_and_embed_descriptions(
            text_output, clip_model, device
        )
        
        # 5. Predict future actions
        fg_batch = fg_sequence.unsqueeze(0).to(device)  # [1, 16, 512]
        predicted_future = predict_future_actions(joint_model, infos, fg_batch)
        
        # 6. Get ground truth
        target_future = infos[0].get("target_future_sequence", ["none"] * 16)
        
        # 7. Compute metrics
        moc_score = rl_utils._compute_moc(predicted_future[0], target_future)
        first_acc = 1.0 if rl_utils._normalize_label(predicted_future[0][0]) == rl_utils._normalize_label(target_future[0]) else 0.0
        
        total_moc += moc_score
        total_first_acc += first_acc
        num_samples += 1
        
        # 8. Store results
        result = {
            "step": step,
            "sequence_id": infos[0].get("sequence_id"),
            "frame_index": infos[0].get("frame_index"),
            "coarse_labels": predicted_history[-5:],
            "fine_grained_descriptions": segment_descs,
            "predicted_future": predicted_future[0],
            "target_future": target_future,
            "moc_score": moc_score,
            "first_action_accuracy": first_acc,
        }
        results.append(result)
        
        # 9. Step environment
        action = torch.zeros((1, 1)).long().to(device)
        obs, reward, done, infos = envs.step(action)
        
        if done[0]:
            obs = envs.reset()
            infos = envs.get_current_infos()
        
        # Print progress
        if (step + 1) % 10 == 0:
            avg_moc = total_moc / num_samples
            avg_first_acc = total_first_acc / num_samples
            print(f"\nStep {step+1}/{args.num_inference_steps}")
            print(f"  Avg MoC: {avg_moc:.4f}")
            print(f"  Avg First-Action Acc: {avg_first_acc:.4f}")
    
    # Final statistics
    avg_moc = total_moc / num_samples
    avg_first_acc = total_first_acc / num_samples
    
    print("\n" + "=" * 80)
    print("Inference Complete!")
    print("=" * 80)
    print(f"Total samples: {num_samples}")
    print(f"Average MoC: {avg_moc:.4f}")
    print(f"Average First-Action Accuracy: {avg_first_acc:.4f}")
    
    return results, {"avg_moc": avg_moc, "avg_first_acc": avg_first_acc}


def save_results(results, stats, output_dir):
    """결과 저장"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results
    results_file = os.path.join(output_dir, "inference_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Detailed results saved to: {results_file}")
    
    # Save statistics
    stats_file = os.path.join(output_dir, "inference_stats.json")
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"✓ Statistics saved to: {stats_file}")
    
    # Save summary
    summary_file = os.path.join(output_dir, "inference_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Action Anticipation Inference Summary\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total samples: {len(results)}\n")
        f.write(f"Average MoC: {stats['avg_moc']:.4f}\n")
        f.write(f"Average First-Action Accuracy: {stats['avg_first_acc']:.4f}\n")
    print(f"✓ Summary saved to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description='Action Anticipation Inference')
    
    # Model paths
    parser.add_argument('--model-path', type=str, default='liuhaotian/llava-v1.5-7b',
                        help='Path to base LLaVA model')
    parser.add_argument('--vlm-checkpoint', type=str, required=True,
                        help='Path to trained VLM checkpoint')
    parser.add_argument('--futr-checkpoint', type=str, required=True,
                        help='Path to trained FUTR checkpoint')
    parser.add_argument('--cache-dir', type=str, default=None,
                        help='Cache directory for models')
    
    # Dataset
    parser.add_argument('--env-name', type=str, default='utkinect/test',
                        help='Environment name')
    parser.add_argument('--utkinect-root', type=str, required=True,
                        help='Path to UTKinect dataset')
    parser.add_argument('--utkinect-split', type=str, default='test',
                        help='Dataset split (train/test)')
    parser.add_argument('--utkinect-history', type=int, default=6,
                        help='History window size')
    parser.add_argument('--utkinect-frame-skip', type=int, default=1,
                        help='Frame skip')
    
    # Inference settings
    parser.add_argument('--num-inference-steps', type=int, default=100,
                        help='Number of inference steps')
    parser.add_argument('--temperature', type=float, default=0.2,
                        help='Sampling temperature')
    parser.add_argument('--max-new-tokens', type=int, default=256,
                        help='Max new tokens for generation')
    parser.add_argument('--conv-mode', type=str, default='vicuna_v1',
                        help='Conversation mode')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='./inference_results',
                        help='Output directory for results')
    
    # Other
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Load models
    models = load_trained_models(args, device)
    
    # Run inference
    results, stats = evaluate_on_dataset(args, models, device)
    
    # Save results
    save_results(results, stats, args.output_dir)
    
    print("\n✓ Inference completed successfully!")


if __name__ == "__main__":
    main()
