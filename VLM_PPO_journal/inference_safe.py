"""
Safe Inference Script with Segmentation Fault Prevention

Segmentation fault 방지를 위한 안전한 inference 스크립트
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Segmentation fault 방지를 위한 환경 변수 설정
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'

# Add project paths
sys.path.insert(0, os.path.dirname(__file__))

from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr import rl_utils
from joint_model import JointFUTR

from transformers import AutoTokenizer, AutoConfig
import clip
import re


def safe_load_tokenizer(model_path, cache_dir=None):
    """안전하게 tokenizer 로드"""
    print(f"\n[Tokenizer] Loading from: {model_path}")
    
    try:
        # Option 1: 직접 경로에서 로드 시도
        if os.path.exists(model_path):
            print("  Attempting to load from local path...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                use_fast=False,  # Fast tokenizer 비활성화 (segfault 방지)
                trust_remote_code=True,
                cache_dir=cache_dir
            )
            print("  ✓ Tokenizer loaded from local path")
            return tokenizer
    except Exception as e:
        print(f"  ✗ Failed to load from local path: {e}")
    
    try:
        # Option 2: HuggingFace Hub에서 로드
        print("  Attempting to load from HuggingFace Hub...")
        tokenizer = AutoTokenizer.from_pretrained(
            "liuhaotian/llava-v1.5-7b",
            use_fast=False,
            trust_remote_code=True,
            cache_dir=cache_dir
        )
        print("  ✓ Tokenizer loaded from HuggingFace Hub")
        return tokenizer
    except Exception as e:
        print(f"  ✗ Failed to load from HuggingFace Hub: {e}")
    
    # Option 3: Llama tokenizer fallback
    print("  Attempting to use Llama tokenizer as fallback...")
    try:
        from transformers import LlamaTokenizer
        tokenizer = LlamaTokenizer.from_pretrained(
            "huggyllama/llama-7b",
            use_fast=False,
            cache_dir=cache_dir
        )
        print("  ✓ Llama tokenizer loaded as fallback")
        return tokenizer
    except Exception as e:
        print(f"  ✗ All tokenizer loading methods failed: {e}")
        raise RuntimeError("Cannot load tokenizer")


def load_futr_only(args, device):
    """FUTR 모델만 로드 (VLM 없이)"""
    
    print("=" * 80)
    print("Loading FUTR model only (VLM-free inference)")
    print("=" * 80)
    
    # Load CLIP model
    print("\n1. Loading CLIP model...")
    clip_model, _ = clip.load("ViT-B/32", device=device)
    clip_model = clip_model.float().eval()
    for param in clip_model.parameters():
        param.requires_grad = False
    print("✓ CLIP model loaded")
    
    # Load FUTR model
    print(f"\n2. Loading FUTR model: {args.futr_checkpoint}")
    dataset_root = os.path.abspath(os.path.expanduser(args.utkinect_root))
    joint_model = JointFUTR(device, dataset_root, model_path=args.futr_checkpoint, lr=1e-6)
    joint_model.model.eval()
    print("✓ FUTR model loaded")
    
    # Load tokenizer safely
    print("\n3. Loading tokenizer...")
    tokenizer = safe_load_tokenizer(args.model_path, args.cache_dir)
    print("✓ Tokenizer loaded")
    
    print("\n" + "=" * 80)
    print("Models loaded successfully!")
    print("=" * 80 + "\n")
    
    return tokenizer, clip_model, joint_model


def evaluate_futr_only(args, models, device):
    """FUTR만 사용하여 평가 (VLM 없이)"""
    
    tokenizer, clip_model, joint_model = models
    
    # Create environment
    utkinect_config = {
        "dataset_root": os.path.abspath(os.path.expanduser(args.utkinect_root)),
        "split": args.utkinect_split,
        "history_window": args.utkinect_history,
        "frame_skip": args.utkinect_frame_skip,
    }
    
    envs = make_vec_envs(
        args.env_name, args.seed, 1,
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
    print("Starting FUTR-only inference (no VLM fine-grained descriptions)")
    print("=" * 80)
    
    for step in tqdm(range(args.num_inference_steps)):
        # 1. Get coarse labels from FUTR (visual features only)
        pred_hist_list = joint_model.predict_coarse(infos)
        predicted_history = pred_hist_list[0] if pred_hist_list else []
        
        # 2. Use FUTR for future prediction WITHOUT fine-grained context
        # (context=None means visual features only)
        predicted_future = joint_model.predict_future(infos, fg_embedding=None)
        
        # 3. Get ground truth
        target_future = infos[0].get("target_future_sequence", ["none"] * 16)
        
        # 4. Compute metrics
        moc_score = rl_utils._compute_moc(predicted_future[0], target_future)
        first_acc = 1.0 if rl_utils._normalize_label(predicted_future[0][0]) == rl_utils._normalize_label(target_future[0]) else 0.0
        
        total_moc += moc_score
        total_first_acc += first_acc
        num_samples += 1
        
        # 5. Store results
        result = {
            "step": step,
            "sequence_id": infos[0].get("sequence_id"),
            "frame_index": infos[0].get("frame_index"),
            "coarse_labels": predicted_history[-5:],
            "predicted_future": predicted_future[0],
            "target_future": target_future,
            "moc_score": moc_score,
            "first_action_accuracy": first_acc,
        }
        results.append(result)
        
        # 6. Step environment
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
    print("\nNote: This is FUTR-only baseline (no VLM fine-grained descriptions)")
    
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
        f.write("Action Anticipation Inference Summary (FUTR-only)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total samples: {len(results)}\n")
        f.write(f"Average MoC: {stats['avg_moc']:.4f}\n")
        f.write(f"Average First-Action Accuracy: {stats['avg_first_acc']:.4f}\n")
        f.write("\nNote: This is FUTR-only baseline without VLM fine-grained descriptions\n")
    print(f"✓ Summary saved to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description='Safe Action Anticipation Inference')
    
    # Model paths
    parser.add_argument('--model-path', type=str, default='liuhaotian/llava-v1.5-7b',
                        help='Path to base model (for tokenizer only)')
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
    
    # Output
    parser.add_argument('--output-dir', type=str, default='./inference_results_safe',
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
    
    try:
        # Load models (FUTR only, no VLM)
        models = load_futr_only(args, device)
        
        # Run inference
        results, stats = evaluate_futr_only(args, models, device)
        
        # Save results
        save_results(results, stats, args.output_dir)
        
        print("\n✓ Inference completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
