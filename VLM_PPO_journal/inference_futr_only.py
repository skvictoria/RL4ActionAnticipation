"""
FUTR-only Inference Script
diagnose_segfault.py를 베이스로 만든 안전한 inference 스크립트
VLM 없이 FUTR만 사용하여 segmentation fault 회피
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm

# diagnose_segfault.py와 동일한 환경 변수 설정
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

# Add project paths
sys.path.insert(0, os.path.dirname(__file__))

from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr import rl_utils
from joint_model import JointFUTR


def print_section(title):
    """diagnose_segfault.py 스타일 출력"""
    print("\n" + "="*80)
    print(f"{title}")
    print("="*80)


def load_futr_model(args, device):
    """FUTR 모델만 로드 (diagnose_segfault.py test_futr와 동일)"""
    
    print_section("Loading FUTR Model")
    
    if not os.path.exists(args.futr_checkpoint):
        raise FileNotFoundError(f"FUTR checkpoint not found: {args.futr_checkpoint}")
    
    if not os.path.exists(args.dataset_root):
        raise FileNotFoundError(f"Dataset root not found: {args.dataset_root}")
    
    print(f"FUTR checkpoint: {args.futr_checkpoint}")
    print(f"Dataset root: {args.dataset_root}")
    print(f"Device: {device}")
    
    try:
        joint_model = JointFUTR(device, args.dataset_root, model_path=args.futr_checkpoint, lr=1e-6)
        joint_model.model.eval()
        print("✓ FUTR model loaded successfully")
        return joint_model
    except Exception as e:
        print(f"✗ FUTR loading failed: {e}")
        raise


def create_environment(args, device):
    """환경 생성"""
    
    print_section("Creating Environment")
    
    utkinect_config = {
        "dataset_root": args.dataset_root,
        "split": args.split,
        "history_window": args.history_window,
        "frame_skip": args.frame_skip,
    }
    
    print(f"Environment: {args.env_name}")
    print(f"Dataset: {utkinect_config['dataset_root']}")
    print(f"Split: {utkinect_config['split']}")
    print(f"History window: {utkinect_config['history_window']}")
    print(f"Frame skip: {utkinect_config['frame_skip']}")
    
    envs = make_vec_envs(
        args.env_name, args.seed, 1,
        args.gamma, None, device, False, 1,
        utkinect_config=utkinect_config
    )
    print("✓ Environment created successfully")
    
    return envs


def run_inference(args, joint_model, envs, device):
    """FUTR-only inference 실행"""
    
    print_section("Starting FUTR-only Inference")
    
    print(f"Number of steps: {args.num_steps}")
    print(f"Note: Using visual features only (no VLM)")
    print("")
    
    # Reset environment
    obs = envs.reset()
    infos = envs.get_current_infos()
    
    results = []
    total_moc = 0.0
    total_first_acc = 0.0
    num_samples = 0
    
    # Per-class statistics
    class_correct = {}
    class_total = {}
    
    for step in tqdm(range(args.num_steps), desc="Inference"):
        try:
            # 1. Get coarse labels from FUTR
            pred_hist_list = joint_model.predict_coarse(infos)
            predicted_history = pred_hist_list[0] if pred_hist_list else []
            
            # 2. Predict future actions (FUTR-only, no VLM)
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
            if (step + 1) % 20 == 0:
                avg_moc = total_moc / num_samples
                avg_first_acc = total_first_acc / num_samples
                print(f"\n[Progress] Step {step+1}/{args.num_steps}")
                print(f"  Avg MoC: {avg_moc:.4f}")
                print(f"  Avg First-Action Acc: {avg_first_acc:.4f}")
        
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
    
    print_section("Inference Complete!")
    
    print(f"Total samples: {num_samples}")
    print(f"Average MoC: {avg_moc:.4f}")
    print(f"Average First-Action Accuracy: {avg_first_acc:.4f}")
    print(f"\nPer-class First-Action Accuracy:")
    for cls in sorted(per_class_acc.keys()):
        print(f"  {cls}: {per_class_acc[cls]:.4f} ({class_correct[cls]}/{class_total[cls]})")
    print("\nNote: This is FUTR-only baseline (no VLM)")
    
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
    
    print_section("Saving Results")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results
    results_file = os.path.join(output_dir, "inference_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Detailed results: {results_file}")
    
    # Save statistics
    stats_file = os.path.join(output_dir, "inference_stats.json")
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"✓ Statistics: {stats_file}")
    
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
        f.write("\nNote: This is FUTR-only baseline without VLM\n")
    print(f"✓ Summary: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description='FUTR-only Inference (from diagnose_segfault.py)')
    
    # Paths (diagnose_segfault.py와 동일한 기본값)
    parser.add_argument('--futr-checkpoint', type=str,
                        default='/home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/save_dir/utkinects/long/model/transformer/1/i3d_transcript/runs0/_20_30_50_erank_40p_64_latent_20251226/futr_joint_epoch_99.ckpt',
                        help='Path to FUTR checkpoint')
    parser.add_argument('--dataset-root', type=str,
                        default='/home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/datasets/utkinect',
                        help='Path to UTKinect dataset')
    
    # Dataset settings
    parser.add_argument('--env-name', type=str, default='utkinect/test',
                        help='Environment name')
    parser.add_argument('--split', type=str, default='test',
                        help='Dataset split')
    parser.add_argument('--history-window', type=int, default=6,
                        help='History window size')
    parser.add_argument('--frame-skip', type=int, default=1,
                        help='Frame skip')
    
    # Inference settings
    parser.add_argument('--num-steps', type=int, default=100,
                        help='Number of inference steps')
    parser.add_argument('--output-dir', type=str, default='./inference_results_futr_only',
                        help='Output directory')
    
    # Other
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print_section("FUTR-only Inference Script")
    print("Based on diagnose_segfault.py (which works without errors)")
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"\nEnvironment variables:")
    print(f"  PYTHONNOUSERSITE: {os.environ.get('PYTHONNOUSERSITE', 'Not set')}")
    print(f"  TOKENIZERS_PARALLELISM: {os.environ.get('TOKENIZERS_PARALLELISM')}")
    
    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    try:
        # Load FUTR model (diagnose_segfault.py와 동일한 방식)
        joint_model = load_futr_model(args, device)
        
        # Create environment
        envs = create_environment(args, device)
        
        # Run inference
        results, stats = run_inference(args, joint_model, envs, device)
        
        # Save results
        save_results(results, stats, args.output_dir)
        
        print_section("✓ Inference Completed Successfully!")
        print(f"Results saved to: {args.output_dir}")
        
    except Exception as e:
        print_section("✗ Error During Inference")
        print(f"{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
