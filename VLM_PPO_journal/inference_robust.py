"""
Robust Inference Script with Multiple Fallback Strategies

Segmentation fault 방지를 위한 강력한 inference 스크립트
여러 fallback 전략을 통해 안정적인 추론 수행
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
os.environ['MKL_NUM_THREADS'] = '1'

# Add project paths
sys.path.insert(0, os.path.dirname(__file__))

from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr import rl_utils
from joint_model import JointFUTR

from transformers import AutoTokenizer, AutoConfig, LlamaTokenizer
import clip
import re


def safe_load_tokenizer_v2(model_path, cache_dir=None):
    """
    안전하게 tokenizer 로드 (여러 fallback 전략 사용)
    
    Strategy:
    1. Local checkpoint directory (config.json 있는 경우)
    2. Local checkpoint directory with base model fallback
    3. HuggingFace Hub (liuhaotian/llava-v1.5-7b)
    4. Llama tokenizer fallback
    5. Llama-2 tokenizer fallback
    """
    print(f"\n{'='*80}")
    print(f"[Tokenizer] Loading tokenizer...")
    print(f"{'='*80}")
    print(f"Target path: {model_path}")
    
    # Strategy 1: Load from local checkpoint if config.json exists
    if os.path.exists(model_path):
        config_path = os.path.join(model_path, "config.json") if os.path.isdir(model_path) else None
        
        if config_path and os.path.exists(config_path):
            print(f"\n[Strategy 1] Found config.json at: {config_path}")
            try:
                print("  Attempting to load tokenizer from local checkpoint...")
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    use_fast=False,
                    trust_remote_code=True,
                    cache_dir=cache_dir,
                    local_files_only=True  # Force local loading
                )
                print("  ✓ Tokenizer loaded from local checkpoint")
                return tokenizer
            except Exception as e:
                print(f"  ✗ Failed: {e}")
        
        # Strategy 2: Load from local with base model fallback
        if os.path.isdir(model_path):
            print(f"\n[Strategy 2] Trying local path with base model fallback...")
            try:
                # Check if adapter_config.json exists (LoRA checkpoint)
                adapter_config_path = os.path.join(model_path, "adapter_config.json")
                if os.path.exists(adapter_config_path):
                    print("  Found LoRA checkpoint, loading base model tokenizer...")
                    with open(adapter_config_path, 'r') as f:
                        adapter_config = json.load(f)
                        base_model_name = adapter_config.get("base_model_name_or_path", "liuhaotian/llava-v1.5-7b")
                    
                    print(f"  Base model: {base_model_name}")
                    tokenizer = AutoTokenizer.from_pretrained(
                        base_model_name,
                        use_fast=False,
                        trust_remote_code=True,
                        cache_dir=cache_dir
                    )
                    print("  ✓ Tokenizer loaded from base model")
                    return tokenizer
            except Exception as e:
                print(f"  ✗ Failed: {e}")
    
    # Strategy 3: Load from HuggingFace Hub
    print(f"\n[Strategy 3] Loading from HuggingFace Hub...")
    try:
        print("  Attempting to load liuhaotian/llava-v1.5-7b...")
        tokenizer = AutoTokenizer.from_pretrained(
            "liuhaotian/llava-v1.5-7b",
            use_fast=False,
            trust_remote_code=True,
            cache_dir=cache_dir
        )
        print("  ✓ Tokenizer loaded from HuggingFace Hub")
        return tokenizer
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    
    # Strategy 4: Llama tokenizer fallback
    print(f"\n[Strategy 4] Trying Llama tokenizer fallback...")
    try:
        tokenizer = LlamaTokenizer.from_pretrained(
            "huggyllama/llama-7b",
            use_fast=False,
            cache_dir=cache_dir
        )
        print("  ✓ Llama tokenizer loaded")
        return tokenizer
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    
    # Strategy 5: Llama-2 tokenizer fallback
    print(f"\n[Strategy 5] Trying Llama-2 tokenizer fallback...")
    try:
        tokenizer = LlamaTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            use_fast=False,
            cache_dir=cache_dir
        )
        print("  ✓ Llama-2 tokenizer loaded")
        return tokenizer
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    
    print(f"\n{'='*80}")
    print("✗ All tokenizer loading strategies failed!")
    print(f"{'='*80}\n")
    raise RuntimeError("Cannot load tokenizer with any strategy")


def load_futr_only(args, device):
    """FUTR 모델만 로드 (VLM 없이)"""
    
    print("\n" + "="*80)
    print("Loading FUTR model only (VLM-free inference)")
    print("="*80)
    
    # Load CLIP model
    print("\n[1/3] Loading CLIP model...")
    try:
        clip_model, _ = clip.load("ViT-B/32", device=device)
        clip_model = clip_model.float().eval()
        for param in clip_model.parameters():
            param.requires_grad = False
        print("✓ CLIP model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load CLIP: {e}")
        raise
    
    # Load FUTR model
    print(f"\n[2/3] Loading FUTR model...")
    print(f"  Checkpoint: {args.futr_checkpoint}")
    try:
        dataset_root = os.path.abspath(os.path.expanduser(args.utkinect_root))
        if not os.path.exists(dataset_root):
            raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
        
        joint_model = JointFUTR(device, dataset_root, model_path=args.futr_checkpoint, lr=1e-6)
        joint_model.model.eval()
        print("✓ FUTR model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load FUTR: {e}")
        raise
    
    # Load tokenizer safely
    print(f"\n[3/3] Loading tokenizer...")
    try:
        tokenizer = safe_load_tokenizer_v2(args.model_path, args.cache_dir)
        print("✓ Tokenizer loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load tokenizer: {e}")
        raise
    
    print("\n" + "="*80)
    print("✓ All models loaded successfully!")
    print("="*80 + "\n")
    
    return tokenizer, clip_model, joint_model


def evaluate_futr_only(args, models, device):
    """FUTR만 사용하여 평가 (VLM 없이)"""
    
    tokenizer, clip_model, joint_model = models
    
    # Create environment
    print("\n" + "="*80)
    print("Creating evaluation environment...")
    print("="*80)
    
    utkinect_config = {
        "dataset_root": os.path.abspath(os.path.expanduser(args.utkinect_root)),
        "split": args.utkinect_split,
        "history_window": args.utkinect_history,
        "frame_skip": args.utkinect_frame_skip,
    }
    
    print(f"  Dataset: {utkinect_config['dataset_root']}")
    print(f"  Split: {utkinect_config['split']}")
    print(f"  History window: {utkinect_config['history_window']}")
    print(f"  Frame skip: {utkinect_config['frame_skip']}")
    
    try:
        envs = make_vec_envs(
            args.env_name, args.seed, 1,
            args.gamma, None, device, False, 1,
            utkinect_config=utkinect_config
        )
        print("✓ Environment created successfully\n")
    except Exception as e:
        print(f"✗ Failed to create environment: {e}")
        raise
    
    # Reset environment
    obs = envs.reset()
    infos = envs.get_current_infos()
    
    results = []
    total_moc = 0.0
    total_first_acc = 0.0
    num_samples = 0
    
    # Per-class statistics for MoC
    class_correct = {}
    class_total = {}
    
    print("="*80)
    print("Starting FUTR-only inference")
    print("="*80)
    print(f"Number of steps: {args.num_inference_steps}")
    print(f"Note: Using visual features only (no VLM fine-grained descriptions)")
    print("="*80 + "\n")
    
    for step in tqdm(range(args.num_inference_steps), desc="Inference"):
        try:
            # 1. Get coarse labels from FUTR (visual features only)
            pred_hist_list = joint_model.predict_coarse(infos)
            predicted_history = pred_hist_list[0] if pred_hist_list else []
            
            # 2. Use FUTR for future prediction WITHOUT fine-grained context
            # (fg_embedding=None means visual features only)
            predicted_future = joint_model.predict_future(infos, fg_embedding=None)
            
            # 3. Get ground truth
            target_future = infos[0].get("target_future_sequence", ["none"] * 16)
            
            # 4. Compute metrics
            moc_score = rl_utils._compute_moc(predicted_future[0], target_future)
            first_pred = rl_utils._normalize_label(predicted_future[0][0])
            first_target = rl_utils._normalize_label(target_future[0])
            first_acc = 1.0 if first_pred == first_target else 0.0
            
            # Update per-class statistics
            if first_target not in class_total:
                class_total[first_target] = 0
                class_correct[first_target] = 0
            class_total[first_target] += 1
            class_correct[first_target] += first_acc
            
            total_moc += moc_score
            total_first_acc += first_acc
            num_samples += 1
            
            # 5. Store results
            result = {
                "step": step,
                "sequence_id": infos[0].get("sequence_id"),
                "frame_index": infos[0].get("frame_index"),
                "coarse_labels": predicted_history[-5:] if len(predicted_history) >= 5 else predicted_history,
                "predicted_future": predicted_future[0],
                "target_future": target_future,
                "moc_score": float(moc_score),
                "first_action_accuracy": float(first_acc),
            }
            results.append(result)
            
            # 6. Step environment
            action = torch.zeros((1, 1)).long().to(device)
            obs, reward, done, infos = envs.step(action)
            
            if done[0]:
                obs = envs.reset()
                infos = envs.get_current_infos()
            
            # Print progress
            if (step + 1) % 50 == 0:
                avg_moc = total_moc / num_samples
                avg_first_acc = total_first_acc / num_samples
                print(f"\n[Progress] Step {step+1}/{args.num_inference_steps}")
                print(f"  Avg MoC: {avg_moc:.4f}")
                print(f"  Avg First-Action Acc: {avg_first_acc:.4f}")
                print(f"  Samples processed: {num_samples}")
        
        except Exception as e:
            print(f"\n✗ Error at step {step}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Final statistics
    avg_moc = total_moc / num_samples if num_samples > 0 else 0.0
    avg_first_acc = total_first_acc / num_samples if num_samples > 0 else 0.0
    
    # Per-class accuracy
    per_class_acc = {}
    for cls in class_total:
        per_class_acc[cls] = class_correct[cls] / class_total[cls] if class_total[cls] > 0 else 0.0
    
    print("\n" + "="*80)
    print("Inference Complete!")
    print("="*80)
    print(f"Total samples: {num_samples}")
    print(f"Average MoC: {avg_moc:.4f}")
    print(f"Average First-Action Accuracy: {avg_first_acc:.4f}")
    print(f"\nPer-class First-Action Accuracy:")
    for cls in sorted(per_class_acc.keys()):
        print(f"  {cls}: {per_class_acc[cls]:.4f} ({class_correct[cls]}/{class_total[cls]})")
    print("\nNote: This is FUTR-only baseline (no VLM fine-grained descriptions)")
    print("="*80 + "\n")
    
    stats = {
        "avg_moc": float(avg_moc),
        "avg_first_acc": float(avg_first_acc),
        "total_samples": num_samples,
        "per_class_accuracy": {k: float(v) for k, v in per_class_acc.items()},
        "per_class_counts": {k: int(v) for k, v in class_total.items()},
    }
    
    return results, stats


def save_results(results, stats, output_dir):
    """결과 저장"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results
    results_file = os.path.join(output_dir, "inference_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Detailed results saved to: {results_file}")
    
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
        f.write(f"Total samples: {stats['total_samples']}\n")
        f.write(f"Average MoC: {stats['avg_moc']:.4f}\n")
        f.write(f"Average First-Action Accuracy: {stats['avg_first_acc']:.4f}\n\n")
        f.write("Per-class First-Action Accuracy:\n")
        for cls in sorted(stats['per_class_accuracy'].keys()):
            acc = stats['per_class_accuracy'][cls]
            count = stats['per_class_counts'][cls]
            f.write(f"  {cls}: {acc:.4f} ({count} samples)\n")
        f.write("\nNote: This is FUTR-only baseline without VLM fine-grained descriptions\n")
    print(f"✓ Summary saved to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description='Robust Action Anticipation Inference')
    
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
    parser.add_argument('--output-dir', type=str, default='./inference_results_robust',
                        help='Output directory for results')
    
    # Other
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*80}")
    print(f"Robust Action Anticipation Inference")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"{'='*80}\n")
    
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
        
        print("\n" + "="*80)
        print("✓ Inference completed successfully!")
        print("="*80)
        print(f"Results saved to: {args.output_dir}")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"✗ Error during inference")
        print(f"{'='*80}")
        print(f"{e}")
        print(f"{'='*80}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
