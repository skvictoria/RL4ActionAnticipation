"""
Complete Inference with VLM + FUTR
diagnose_segfault.py를 베이스로 VLM까지 추가한 완전한 inference
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import re

# diagnose_segfault.py와 동일한 환경 변수 설정
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['PYTHONNOUSERSITE'] = '1'

# Add project paths
sys.path.insert(0, os.path.dirname(__file__))

from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr import rl_utils
from joint_model import JointFUTR

from transformers import AutoTokenizer
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.model import LlavaLlamaForCausalLM
from llava.mm_utils import tokenizer_image_token
from peft import PeftModel

import clip


def load_models(args, device):
    """VLM + FUTR 모델 로드 (diagnose_segfault.py 방식 + VLM 추가)"""
    
    print("\n" + "="*80)
    print("Loading Models (VLM + FUTR)")
    print("="*80)
    
    # 1. Load Tokenizer (diagnose_segfault.py와 동일)
    print("\n[1/5] Loading Tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "liuhaotian/llava-v1.5-7b",
            use_fast=False,
            trust_remote_code=True
        )
        print(f"✓ Tokenizer loaded: vocab size = {len(tokenizer)}")
    except Exception as e:
        print(f"✗ Tokenizer loading failed: {e}")
        raise
    
    # 2. Load Base LLaVA Model (새로 추가)
    print("\n[2/5] Loading Base LLaVA Model...")
    try:
        base_model = LlavaLlamaForCausalLM.from_pretrained(
            "liuhaotian/llava-v1.5-7b",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        base_model.config.max_length = 1024
        print("✓ Base LLaVA model loaded")
    except Exception as e:
        print(f"✗ Base model loading failed: {e}")
        raise
    
    # 3. Load LoRA Weights (새로 추가)
    if args.vlm_checkpoint and os.path.exists(args.vlm_checkpoint):
        print(f"\n[3/5] Loading VLM LoRA Weights...")
        print(f"  Checkpoint: {args.vlm_checkpoint}")
        try:
            vlm_model = PeftModel.from_pretrained(base_model, args.vlm_checkpoint)
            vlm_model.eval()
            print("✓ VLM LoRA weights loaded")
        except Exception as e:
            print(f"✗ LoRA loading failed: {e}")
            print("  Using base model without LoRA")
            vlm_model = base_model
    else:
        print(f"\n[3/5] No VLM checkpoint provided, using base model")
        vlm_model = base_model
    
    # Get image processor
    image_processor = vlm_model.get_vision_tower().image_processor
    
    # 4. Load CLIP (diagnose_segfault.py와 동일)
    print("\n[4/5] Loading CLIP...")
    try:
        clip_model, _ = clip.load("ViT-B/32", device=device)
        clip_model = clip_model.float().eval()
        for param in clip_model.parameters():
            param.requires_grad = False
        print("✓ CLIP model loaded")
    except Exception as e:
        print(f"✗ CLIP loading failed: {e}")
        raise
    
    # 5. Load FUTR (diagnose_segfault.py와 동일)
    print(f"\n[5/5] Loading FUTR...")
    print(f"  Checkpoint: {args.futr_checkpoint}")
    print(f"  Dataset: {args.dataset_root}")
    try:
        joint_model = JointFUTR(device, args.dataset_root, model_path=args.futr_checkpoint, lr=1e-6)
        joint_model.model.eval()
        print("✓ FUTR model loaded")
    except Exception as e:
        print(f"✗ FUTR loading failed: {e}")
        raise
    
    print("\n" + "="*80)
    print("✓ All models loaded successfully!")
    print("="*80 + "\n")
    
    return tokenizer, image_processor, vlm_model, clip_model, joint_model


def generate_fine_grained_descriptions(vlm_model, tokenizer, image_processor, obs, coarse_labels, device, args):
    """VLM으로 fine-grained descriptions 생성"""
    
    # Prepare prompt (4-segment approach)
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
    
    # Get the actual device of the model
    model_device = next(vlm_model.parameters()).device
    INPUT_IDS = INPUT_IDS.to(model_device)
    
    # Generate
    with torch.no_grad():
        output_ids = vlm_model.generate(
            INPUT_IDS,
            images=obs.to(model_device),
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


def run_inference(args, models, device):
    """VLM + FUTR inference 실행"""
    
    tokenizer, image_processor, vlm_model, clip_model, joint_model = models
    
    # Create environment
    print("="*80)
    print("Creating Environment...")
    print("="*80)
    
    utkinect_config = {
        "dataset_root": args.dataset_root,
        "split": args.split,
        "history_window": args.history_window,
        "frame_skip": args.frame_skip,
    }
    
    print(f"  Dataset: {utkinect_config['dataset_root']}")
    print(f"  Split: {utkinect_config['split']}")
    
    envs = make_vec_envs(
        args.env_name, args.seed, 1,
        args.gamma, None, device, False, 1,
        utkinect_config=utkinect_config
    )
    print("✓ Environment created\n")
    
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
    
    print("="*80)
    print("Starting VLM + FUTR Inference")
    print("="*80)
    print(f"Number of steps: {args.num_steps}")
    print(f"Using VLM fine-grained descriptions: {args.vlm_checkpoint is not None}")
    print("="*80 + "\n")
    
    for step in tqdm(range(args.num_steps), desc="Inference"):
        try:
            # 1. Get coarse labels from FUTR
            pred_hist_list = joint_model.predict_coarse(infos)
            predicted_history = pred_hist_list[0] if pred_hist_list else []
            
            # 2. Generate fine-grained descriptions with VLM (if available)
            if args.vlm_checkpoint and step < 3:
                # For first few steps, use simple approach
                fg_embedding = None
            elif args.vlm_checkpoint:
                # Sample 3 frames for VLM
                history_indices = [int(step * 0.5), int(step * 0.75), step]
                multi_obs = obs.unsqueeze(0).repeat(3, 1, 1, 1, 1)  # [3, 1, C, H, W]
                multi_obs = multi_obs.transpose(0, 1)  # [1, 3, C, H, W]
                
                # Generate descriptions
                text_output = generate_fine_grained_descriptions(
                    vlm_model, tokenizer, image_processor, multi_obs,
                    predicted_history, device, args
                )
                
                # Parse and embed
                fg_sequence, segment_descs = parse_and_embed_descriptions(
                    text_output, clip_model, device
                )
                fg_embedding = fg_sequence.unsqueeze(0).to(device)  # [1, 16, 512]
            else:
                fg_embedding = None
                segment_descs = []
            
            # 3. Predict future actions with FUTR
            predicted_future = joint_model.predict_future(infos, fg_embedding)
            
            # 4. Get ground truth
            target_future = infos[0].get("target_future_sequence", ["none"] * 16)
            
            # 5. Compute metrics
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
            
            # 6. Store results
            result = {
                "step": step,
                "sequence_id": infos[0].get("sequence_id"),
                "frame_index": infos[0].get("frame_index"),
                "coarse_labels": predicted_history[-5:] if len(predicted_history) >= 5 else predicted_history,
                "fine_grained_descriptions": segment_descs if args.vlm_checkpoint else [],
                "predicted_future": predicted_future[0],
                "target_future": target_future,
                "moc_score": float(moc_score),
                "first_action_accuracy": float(first_acc),
            }
            results.append(result)
            
            # 7. Step environment
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
    
    print("\n" + "="*80)
    print("Inference Complete!")
    print("="*80)
    print(f"Total samples: {num_samples}")
    print(f"Average MoC: {avg_moc:.4f}")
    print(f"Average First-Action Accuracy: {avg_first_acc:.4f}")
    print(f"\nPer-class First-Action Accuracy:")
    for cls in sorted(per_class_acc.keys()):
        print(f"  {cls}: {per_class_acc[cls]:.4f} ({class_correct[cls]}/{class_total[cls]})")
    
    if args.vlm_checkpoint:
        print("\nUsing VLM fine-grained descriptions")
    else:
        print("\nFUTR-only (no VLM)")
    print("="*80 + "\n")
    
    stats = {
        "avg_moc": float(avg_moc),
        "avg_first_acc": float(avg_first_acc),
        "total_samples": num_samples,
        "per_class_accuracy": {k: float(v) for k, v in per_class_acc.items()},
        "per_class_counts": {k: int(v) for k, v in class_total.items()},
        "used_vlm": args.vlm_checkpoint is not None,
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
        f.write("Action Anticipation Inference Summary\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total samples: {stats['total_samples']}\n")
        f.write(f"Average MoC: {stats['avg_moc']:.4f}\n")
        f.write(f"Average First-Action Accuracy: {stats['avg_first_acc']:.4f}\n\n")
        f.write("Per-class First-Action Accuracy:\n")
        for cls in sorted(stats['per_class_accuracy'].keys()):
            acc = stats['per_class_accuracy'][cls]
            count = stats['per_class_counts'][cls]
            f.write(f"  {cls}: {acc:.4f} ({count} samples)\n")
        f.write(f"\nUsed VLM: {stats['used_vlm']}\n")
    print(f"✓ Summary saved to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description='Complete Inference (VLM + FUTR)')
    
    # Paths
    parser.add_argument('--vlm-checkpoint', type=str,
                        default='/home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/save_dir/utkinects/long/model/transformer/1/i3d_transcript/runs0/_20_30_50_erank_40p_64_latent_20251226/vlm_checkpoints/epoch_99',
                        help='Path to VLM checkpoint (optional)')
    parser.add_argument('--futr-checkpoint', type=str,
                        default='/home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/save_dir/utkinects/long/model/transformer/1/i3d_transcript/runs0/_20_30_50_erank_40p_64_latent_20251226/futr_joint_epoch_99.ckpt',
                        help='Path to FUTR checkpoint')
    parser.add_argument('--dataset-root', type=str,
                        default='/home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/datasets/utkinect',
                        help='Path to UTKinect dataset')
    
    # Dataset settings
    parser.add_argument('--env-name', type=str, default='utkinect/test')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--history-window', type=int, default=6)
    parser.add_argument('--frame-skip', type=int, default=1)
    
    # VLM settings
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--conv-mode', type=str, default='vicuna_v1')
    
    # Inference settings
    parser.add_argument('--num-steps', type=int, default=100)
    parser.add_argument('--output-dir', type=str, default='./inference_results_complete')
    
    # Other
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.99)
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*80)
    print("Complete Inference (VLM + FUTR)")
    print("="*80)
    print("Based on diagnose_segfault.py + VLM loading")
    print("="*80)
    print(f"Device: {device}")
    print(f"Environment variables:")
    print(f"  PYTHONNOUSERSITE: {os.environ.get('PYTHONNOUSERSITE')}")
    print(f"  TOKENIZERS_PARALLELISM: {os.environ.get('TOKENIZERS_PARALLELISM')}")
    print("="*80 + "\n")
    
    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    try:
        # Load models
        models = load_models(args, device)
        
        # Run inference
        results, stats = run_inference(args, models, device)
        
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
