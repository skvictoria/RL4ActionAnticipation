"""
Simple Action Anticipation Inference Script (Segfault-safe version)

단계별로 모델을 로드하고 테스트하는 안전한 버전
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import gc

# Add project paths
sys.path.insert(0, os.path.dirname(__file__))

print("Step 1: Importing modules...")

from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import VLMPolicy, VLMValue
from a2c_ppo_acktr import rl_utils
from joint_model import JointFUTR

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token
from llava.model import LlavaLlamaForCausalLM

from peft import PeftModel
from transformers import AutoTokenizer
import clip
import re

print("✓ All modules imported successfully")


def load_models_safely(args, device):
    """안전하게 모델 로드 (단계별 확인)"""
    
    print("\n" + "=" * 80)
    print("Loading models step by step...")
    print("=" * 80)
    
    # Disable tokenizers parallelism to avoid deadlocks
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # 1. Load tokenizer first
    print("\n[1/5] Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, 
            cache_dir=args.cache_dir,
            use_fast=False,
            trust_remote_code=True
        )
        print("✓ Tokenizer loaded")
    except Exception as e:
        print(f"✗ Failed to load tokenizer: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # 2. Load base model with safe settings
    print("\n[2/5] Loading base LLaVA model...")
    print(f"  Model: {args.model_path}")
    print(f"  Device: {device}")
    
    try:
        # Load with minimal memory footprint
        base = LlavaLlamaForCausalLM.from_pretrained(
            args.model_path,
            cache_dir=args.cache_dir,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        print("✓ Base model loaded")
    except Exception as e:
        print(f"✗ Failed to load base model: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Move to device
    print(f"  Moving to {device}...")
    try:
        base = base.to(device)
        base.eval()
        print("✓ Model moved to device")
    except Exception as e:
        print(f"✗ Failed to move model to device: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # 3. Load LoRA weights if provided
    if args.vlm_checkpoint:
        print(f"\n[3/5] Loading VLM checkpoint...")
        print(f"  Checkpoint: {args.vlm_checkpoint}")
        
        try:
            if os.path.isdir(args.vlm_checkpoint):
                base = PeftModel.from_pretrained(base, args.vlm_checkpoint)
                print("✓ LoRA weights loaded from directory")
            elif args.vlm_checkpoint.endswith('.pt'):
                checkpoint = torch.load(args.vlm_checkpoint, map_location=device)
                base.load_state_dict(checkpoint.get('model_state_dict', checkpoint), strict=False)
                print("✓ Checkpoint loaded from .pt file")
        except Exception as e:
            print(f"⚠ Warning: Failed to load checkpoint: {e}")
            print("  Continuing with base model only...")
    else:
        print("\n[3/5] No VLM checkpoint provided, using base model")
    
    # 4. Load CLIP
    print("\n[4/5] Loading CLIP model...")
    try:
        clip_model, _ = clip.load("ViT-B/32", device=device)
        clip_model = clip_model.float()  # Convert to float32
        clip_model.eval()
        for param in clip_model.parameters():
            param.requires_grad = False
        print("✓ CLIP model loaded")
    except Exception as e:
        print(f"✗ Failed to load CLIP: {e}")
        raise
    
    # 5. Load FUTR
    print("\n[5/5] Loading FUTR model...")
    print(f"  Checkpoint: {args.futr_checkpoint}")
    print(f"  Dataset root: {args.utkinect_root}")
    
    try:
        dataset_root = os.path.abspath(os.path.expanduser(args.utkinect_root))
        joint_model = JointFUTR(device, dataset_root, model_path=args.futr_checkpoint, lr=1e-6)
        joint_model.model.eval()
        print("✓ FUTR model loaded")
    except Exception as e:
        print(f"✗ Failed to load FUTR: {e}")
        raise
    
    # Get image processor
    image_processor = base.get_vision_tower().image_processor
    
    print("\n" + "=" * 80)
    print("✓ All models loaded successfully!")
    print("=" * 80)
    
    # Clear cache
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return base, tokenizer, image_processor, clip_model, joint_model


def run_single_inference(base, tokenizer, image_processor, clip_model, joint_model, 
                        obs, infos, device, args):
    """단일 inference 실행"""
    
    # 1. Predict coarse labels
    pred_hist_list = joint_model.predict_coarse(infos)
    predicted_history = pred_hist_list[0] if pred_hist_list else []
    
    # 2. Generate fine-grained descriptions
    segment_labels = predicted_history[-4:] if len(predicted_history) >= 4 else predicted_history
    
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
    
    # Prepare images (use current obs repeated 3 times)
    multi_obs = obs.unsqueeze(0).repeat(3, 1, 1, 1, 1).transpose(0, 1)  # [1, 3, C, H, W]
    
    # Generate
    with torch.no_grad():
        output_ids = base.generate(
            INPUT_IDS,
            images=multi_obs.to(device),
            do_sample=True,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
        )
    
    text_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # 3. Parse and embed
    try:
        json_start = text_output.find('{')
        json_end = text_output.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            json_str = text_output[json_start:json_end]
            json_str = re.sub(r'\\(?!["\\/bfnrtu])', r'', json_str)
            parsed = json.loads(json_str)
            segment_descs = parsed.get("segment_descriptions", [])
            
            if len(segment_descs) == 4:
                segment_embs = []
                for desc in segment_descs:
                    tokens = rl_utils._clip_safe_tokenize(str(desc), device)
                    with torch.no_grad():
                        emb = clip_model.encode_text(tokens).detach().cpu()
                        if emb.dim() > 1:
                            emb = emb.squeeze(0)
                    segment_embs.append(emb)
                
                fg_sequence = []
                for emb in segment_embs:
                    fg_sequence.extend([emb] * 4)
                fg_sequence = torch.stack(fg_sequence)
            else:
                raise ValueError(f"Expected 4 descriptions, got {len(segment_descs)}")
        else:
            raise ValueError("No JSON found")
    except Exception as e:
        print(f"⚠ JSON parsing failed: {e}, using fallback")
        clean_txt = text_output.split("thoughts")[-1].replace('"', '').replace(':', '').strip()
        tokens = rl_utils._clip_safe_tokenize(clean_txt, device)
        with torch.no_grad():
            emb = clip_model.encode_text(tokens).detach().cpu()
            if emb.dim() > 1:
                emb = emb.squeeze(0)
        fg_sequence = emb.unsqueeze(0).repeat(16, 1)
        segment_descs = [clean_txt]
    
    # 4. Predict future actions
    fg_batch = fg_sequence.unsqueeze(0).to(device)
    predicted_future = joint_model.predict_future(infos, fg_batch)
    
    # 5. Get ground truth
    target_future = infos[0].get("target_future_sequence", ["none"] * 16)
    
    # 6. Compute metrics
    moc_score = rl_utils._compute_moc(predicted_future[0], target_future)
    first_acc = 1.0 if rl_utils._normalize_label(predicted_future[0][0]) == rl_utils._normalize_label(target_future[0]) else 0.0
    
    return {
        "coarse_labels": predicted_history[-5:],
        "fine_grained_descriptions": segment_descs,
        "predicted_future": predicted_future[0],
        "target_future": target_future,
        "moc_score": moc_score,
        "first_action_accuracy": first_acc,
    }


def main():
    parser = argparse.ArgumentParser(description='Simple Action Anticipation Inference')
    
    # Model paths
    parser.add_argument('--model-path', type=str, default='liuhaotian/llava-v1.5-7b')
    parser.add_argument('--vlm-checkpoint', type=str, default=None)
    parser.add_argument('--futr-checkpoint', type=str, required=True)
    parser.add_argument('--cache-dir', type=str, default=None)
    
    # Dataset
    parser.add_argument('--env-name', type=str, default='utkinect/test')
    parser.add_argument('--utkinect-root', type=str, required=True)
    parser.add_argument('--utkinect-split', type=str, default='test')
    parser.add_argument('--utkinect-history', type=int, default=6)
    parser.add_argument('--utkinect-frame-skip', type=int, default=1)
    
    # Inference settings
    parser.add_argument('--num-inference-steps', type=int, default=10)
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--conv-mode', type=str, default='vicuna_v1')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='./inference_results')
    
    # Other
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.99)
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Load models
    try:
        base, tokenizer, image_processor, clip_model, joint_model = load_models_safely(args, device)
    except Exception as e:
        print(f"\n✗ Failed to load models: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create environment
    print("\n" + "=" * 80)
    print("Creating environment...")
    print("=" * 80)
    
    utkinect_config = {
        "dataset_root": os.path.abspath(os.path.expanduser(args.utkinect_root)),
        "split": args.utkinect_split,
        "history_window": args.utkinect_history,
        "frame_skip": args.utkinect_frame_skip,
    }
    
    try:
        envs = make_vec_envs(
            args.env_name, args.seed, 1,
            args.gamma, None, device, False, 1,
            utkinect_config=utkinect_config
        )
        print("✓ Environment created")
    except Exception as e:
        print(f"✗ Failed to create environment: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Reset environment
    obs = envs.reset()
    infos = envs.get_current_infos()
    
    # Run inference
    print("\n" + "=" * 80)
    print(f"Running inference ({args.num_inference_steps} steps)...")
    print("=" * 80 + "\n")
    
    results = []
    total_moc = 0.0
    total_first_acc = 0.0
    
    for step in tqdm(range(args.num_inference_steps)):
        try:
            result = run_single_inference(
                base, tokenizer, image_processor, clip_model, joint_model,
                obs, infos, device, args
            )
            
            result["step"] = step
            result["sequence_id"] = infos[0].get("sequence_id")
            result["frame_index"] = infos[0].get("frame_index")
            
            results.append(result)
            total_moc += result["moc_score"]
            total_first_acc += result["first_action_accuracy"]
            
            # Step environment
            action = torch.zeros((1, 1)).long().to(device)
            obs, reward, done, infos = envs.step(action)
            
            if done[0]:
                obs = envs.reset()
                infos = envs.get_current_infos()
            
            # Print progress
            if (step + 1) % 5 == 0:
                avg_moc = total_moc / (step + 1)
                avg_first_acc = total_first_acc / (step + 1)
                print(f"\nStep {step+1}: MoC={avg_moc:.4f}, First-Acc={avg_first_acc:.4f}")
        
        except Exception as e:
            print(f"\n✗ Error at step {step}: {e}")
            import traceback
            traceback.print_exc()
            break
    
    # Final statistics
    if len(results) > 0:
        avg_moc = total_moc / len(results)
        avg_first_acc = total_first_acc / len(results)
        
        print("\n" + "=" * 80)
        print("Inference Complete!")
        print("=" * 80)
        print(f"Total samples: {len(results)}")
        print(f"Average MoC: {avg_moc:.4f}")
        print(f"Average First-Action Accuracy: {avg_first_acc:.4f}")
        
        # Save results
        os.makedirs(args.output_dir, exist_ok=True)
        
        results_file = os.path.join(args.output_dir, "inference_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to: {results_file}")
        
        stats = {"avg_moc": avg_moc, "avg_first_acc": avg_first_acc, "num_samples": len(results)}
        stats_file = os.path.join(args.output_dir, "inference_stats.json")
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"✓ Statistics saved to: {stats_file}")
    else:
        print("\n✗ No results generated")


if __name__ == "__main__":
    main()
