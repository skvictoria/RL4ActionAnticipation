"""
Minimal Inference Script - Uses existing training infrastructure

학습 코드와 동일한 방식으로 모델을 로드하여 segfault 방지
"""

import os
import sys
import argparse
import torch
import numpy as np
import json
from tqdm import tqdm
import copy

# Add project paths
sys.path.insert(0, os.path.dirname(__file__))

# Disable warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("Importing modules...")

from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.llava_interface import load_lora_model
from a2c_ppo_acktr import rl_utils
from joint_model import JointFUTR

import clip
import re

print("✓ Modules imported")


def main():
    parser = argparse.ArgumentParser()
    
    # Model paths
    parser.add_argument('--model-path', type=str, default='liuhaotian/llava-v1.5-7b')
    parser.add_argument('--vlm-checkpoint', type=str, default=None,
                        help='Path to VLM checkpoint (optional)')
    parser.add_argument('--futr-checkpoint', type=str, required=True)
    
    # Dataset
    parser.add_argument('--utkinect-root', type=str, required=True)
    parser.add_argument('--utkinect-split', type=str, default='test')
    parser.add_argument('--utkinect-history', type=int, default=6)
    parser.add_argument('--utkinect-frame-skip', type=int, default=1)
    
    # Inference
    parser.add_argument('--num-steps', type=int, default=10)
    parser.add_argument('--output-dir', type=str, default='./inference_results')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    print("\n" + "=" * 80)
    print("Loading models...")
    print("=" * 80)
    
    # 1. Load VLM using training code's method
    print("\n[1/4] Loading VLM...")
    try:
        # First load base model
        print(f"  Loading base model: {args.model_path}")
        base, tokenizer = load_lora_model(args.model_path)
        base = base.to(device)
        
        # Then load checkpoint if provided
        if args.vlm_checkpoint and os.path.exists(args.vlm_checkpoint):
            print(f"  Loading checkpoint: {args.vlm_checkpoint}")
            
            if os.path.isdir(args.vlm_checkpoint):
                # Try to load LoRA weights from directory
                try:
                    from peft import PeftModel
                    base = PeftModel.from_pretrained(base, args.vlm_checkpoint)
                    print("  ✓ LoRA weights loaded from directory")
                except Exception as e:
                    print(f"  ⚠ Could not load LoRA from directory: {e}")
                    print("  Continuing with base model...")
            
            elif args.vlm_checkpoint.endswith('.pt'):
                # Load from .pt file
                try:
                    checkpoint = torch.load(args.vlm_checkpoint, map_location=device)
                    base.load_state_dict(checkpoint.get('model_state_dict', checkpoint), strict=False)
                    print("  ✓ Checkpoint loaded from .pt file")
                except Exception as e:
                    print(f"  ⚠ Could not load .pt checkpoint: {e}")
                    print("  Continuing with base model...")
        else:
            print("  ⚠ No valid checkpoint provided, using base model only")
        
        base.eval()
        print("✓ VLM loaded")
    except Exception as e:
        print(f"✗ Failed to load VLM: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 2. Load CLIP
    print("\n[2/4] Loading CLIP...")
    try:
        clip_model, _ = clip.load("ViT-B/32", device=device)
        clip_model = clip_model.float().eval()
        for param in clip_model.parameters():
            param.requires_grad = False
        print("✓ CLIP loaded")
    except Exception as e:
        print(f"✗ Failed to load CLIP: {e}")
        return
    
    # 3. Load FUTR
    print("\n[3/4] Loading FUTR...")
    try:
        dataset_root = os.path.abspath(os.path.expanduser(args.utkinect_root))
        joint_model = JointFUTR(device, dataset_root, model_path=args.futr_checkpoint, lr=1e-6)
        joint_model.model.eval()
        print("✓ FUTR loaded")
    except Exception as e:
        print(f"✗ Failed to load FUTR: {e}")
        return
    
    # 4. Create environment
    print("\n[4/4] Creating environment...")
    try:
        utkinect_config = {
            "dataset_root": dataset_root,
            "split": args.utkinect_split,
            "history_window": args.utkinect_history,
            "frame_skip": args.utkinect_frame_skip,
        }
        
        envs = make_vec_envs(
            f'utkinect/{args.utkinect_split}', args.seed, 1,
            0.99, None, device, False, 1,
            utkinect_config=utkinect_config
        )
        print("✓ Environment created")
    except Exception as e:
        print(f"✗ Failed to create environment: {e}")
        return
    
    print("\n" + "=" * 80)
    print("Starting inference...")
    print("=" * 80)
    
    # Reset environment
    obs = envs.reset()
    infos = envs.get_current_infos()
    
    results = []
    total_moc = 0.0
    total_first_acc = 0.0
    
    for step in tqdm(range(args.num_steps)):
        try:
            # 1. Predict coarse labels
            pred_hist_list = joint_model.predict_coarse(infos)
            predicted_history = pred_hist_list[0] if pred_hist_list else []
            
            # 2. Generate fine-grained text (simplified - use coarse labels directly)
            # In full version, you'd use VLM here
            segment_labels = predicted_history[-4:] if len(predicted_history) >= 4 else predicted_history
            
            # Create embeddings from coarse labels
            segment_embs = []
            for label in segment_labels[:4]:
                tokens = rl_utils._clip_safe_tokenize(str(label), device)
                with torch.no_grad():
                    emb = clip_model.encode_text(tokens).detach().cpu()
                    if emb.dim() > 1:
                        emb = emb.squeeze(0)
                segment_embs.append(emb)
            
            # Pad to 4 segments if needed
            while len(segment_embs) < 4:
                segment_embs.append(segment_embs[-1] if segment_embs else torch.zeros(512))
            
            # Repeat each 4 times
            fg_sequence = []
            for emb in segment_embs:
                fg_sequence.extend([emb] * 4)
            fg_sequence = torch.stack(fg_sequence)  # [16, 512]
            
            # 3. Predict future actions
            fg_batch = fg_sequence.unsqueeze(0).to(device)
            predicted_future = joint_model.predict_future(infos, fg_batch)
            
            # 4. Get ground truth
            target_future = infos[0].get("target_future_sequence", ["none"] * 16)
            
            # 5. Compute metrics
            moc_score = rl_utils._compute_moc(predicted_future[0], target_future)
            first_acc = 1.0 if rl_utils._normalize_label(predicted_future[0][0]) == rl_utils._normalize_label(target_future[0]) else 0.0
            
            total_moc += moc_score
            total_first_acc += first_acc
            
            # 6. Store result
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
            
            # 7. Step environment
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
        
        stats = {
            "avg_moc": avg_moc,
            "avg_first_acc": avg_first_acc,
            "num_samples": len(results)
        }
        stats_file = os.path.join(args.output_dir, "inference_stats.json")
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"✓ Statistics saved to: {stats_file}")
    else:
        print("\n✗ No results generated")


if __name__ == "__main__":
    main()
