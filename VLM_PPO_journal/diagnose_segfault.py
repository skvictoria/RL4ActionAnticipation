"""
Segmentation Fault 진단 스크립트

이 스크립트는 segmentation fault의 원인을 찾기 위해
각 컴포넌트를 개별적으로 테스트합니다.
"""

import os
import sys

# Segmentation fault 방지 환경 변수
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

def print_section(title):
    print("\n" + "="*80)
    print(f"{title}")
    print("="*80)

def test_imports():
    """기본 라이브러리 import 테스트"""
    print_section("1. Testing Basic Imports")
    
    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  Device: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import transformers
        print(f"✓ Transformers: {transformers.__version__}")
    except Exception as e:
        print(f"✗ Transformers import failed: {e}")
        return False
    
    try:
        import tokenizers
        print(f"✓ Tokenizers: {tokenizers.__version__}")
    except Exception as e:
        print(f"✗ Tokenizers import failed: {e}")
        return False
    
    try:
        import clip
        print(f"✓ CLIP imported successfully")
    except Exception as e:
        print(f"✗ CLIP import failed: {e}")
        return False
    
    return True

def test_tokenizer_hub():
    """HuggingFace Hub에서 tokenizer 로드 테스트"""
    print_section("2. Testing Tokenizer from HuggingFace Hub")
    
    try:
        from transformers import AutoTokenizer
        print("Attempting to load: liuhaotian/llava-v1.5-7b")
        print("This may take a while on first run...")
        
        tokenizer = AutoTokenizer.from_pretrained(
            "liuhaotian/llava-v1.5-7b",
            use_fast=False,
            trust_remote_code=True
        )
        print("✓ Tokenizer loaded from HuggingFace Hub")
        print(f"  Vocab size: {len(tokenizer)}")
        
        # Test tokenization
        test_text = "Hello, world!"
        tokens = tokenizer(test_text)
        print(f"✓ Tokenization test passed")
        print(f"  Input: {test_text}")
        print(f"  Tokens: {tokens['input_ids'][:10]}...")
        
        return True
    except Exception as e:
        print(f"✗ HuggingFace Hub tokenizer failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tokenizer_local(checkpoint_path):
    """로컬 체크포인트에서 tokenizer 로드 테스트"""
    print_section("3. Testing Tokenizer from Local Checkpoint")
    
    if not os.path.exists(checkpoint_path):
        print(f"⚠ Checkpoint path does not exist: {checkpoint_path}")
        print("  Skipping local tokenizer test")
        return None
    
    print(f"Checkpoint path: {checkpoint_path}")
    
    # Check for required files
    required_files = ['config.json', 'tokenizer_config.json']
    optional_files = ['tokenizer.model', 'vocab.json', 'merges.txt']
    
    print("\nChecking for required files:")
    for fname in required_files:
        fpath = os.path.join(checkpoint_path, fname)
        exists = os.path.exists(fpath)
        status = "✓" if exists else "✗"
        print(f"  {status} {fname}")
        if not exists:
            print(f"    ⚠ Missing required file!")
    
    print("\nChecking for optional files:")
    for fname in optional_files:
        fpath = os.path.join(checkpoint_path, fname)
        exists = os.path.exists(fpath)
        status = "✓" if exists else "-"
        print(f"  {status} {fname}")
    
    try:
        from transformers import AutoTokenizer
        print("\nAttempting to load tokenizer from local checkpoint...")
        
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_path,
            use_fast=False,
            trust_remote_code=True,
            local_files_only=True
        )
        print("✓ Tokenizer loaded from local checkpoint")
        print(f"  Vocab size: {len(tokenizer)}")
        
        # Test tokenization
        test_text = "Hello, world!"
        tokens = tokenizer(test_text)
        print(f"✓ Tokenization test passed")
        
        return True
    except Exception as e:
        print(f"✗ Local checkpoint tokenizer failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_clip():
    """CLIP 모델 로드 테스트"""
    print_section("4. Testing CLIP Model")
    
    try:
        import torch
        import clip
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading CLIP on device: {device}")
        
        clip_model, preprocess = clip.load("ViT-B/32", device=device)
        print("✓ CLIP model loaded")
        
        # Convert to float32
        clip_model = clip_model.float()
        print("✓ CLIP converted to float32")
        
        # Test text encoding
        text = clip.tokenize(["a photo of a cat"]).to(device)
        with torch.no_grad():
            text_features = clip_model.encode_text(text)
        print(f"✓ CLIP text encoding test passed")
        print(f"  Feature shape: {text_features.shape}")
        
        return True
    except Exception as e:
        print(f"✗ CLIP test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_futr(checkpoint_path, dataset_root):
    """FUTR 모델 로드 테스트"""
    print_section("5. Testing FUTR Model")
    
    if not os.path.exists(checkpoint_path):
        print(f"⚠ FUTR checkpoint does not exist: {checkpoint_path}")
        print("  Skipping FUTR test")
        return None
    
    if not os.path.exists(dataset_root):
        print(f"⚠ Dataset root does not exist: {dataset_root}")
        print("  Skipping FUTR test")
        return None
    
    print(f"FUTR checkpoint: {checkpoint_path}")
    print(f"Dataset root: {dataset_root}")
    
    try:
        import torch
        sys.path.insert(0, os.path.dirname(__file__))
        from joint_model import JointFUTR
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading FUTR on device: {device}")
        
        joint_model = JointFUTR(device, dataset_root, model_path=checkpoint_path, lr=1e-6)
        print("✓ FUTR model loaded")
        
        joint_model.model.eval()
        print("✓ FUTR model set to eval mode")
        
        return True
    except Exception as e:
        print(f"✗ FUTR test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_environment():
    """환경 변수 테스트"""
    print_section("6. Testing Environment Variables")
    
    env_vars = [
        'TOKENIZERS_PARALLELISM',
        'OMP_NUM_THREADS',
        'MKL_NUM_THREADS',
        'CUDA_LAUNCH_BLOCKING',
        'CUDA_VISIBLE_DEVICES',
    ]
    
    for var in env_vars:
        value = os.environ.get(var, 'Not set')
        print(f"  {var}: {value}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Diagnose segmentation fault issues')
    parser.add_argument('--vlm-checkpoint', type=str, default='./vlm_checkpoints/epoch_4',
                        help='Path to VLM checkpoint')
    parser.add_argument('--futr-checkpoint', type=str, 
                        default='/home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/save_dir/utkinects/long/model/transformer/1/i3d_transcript/runs0/_20_30_50_erank_40p_64_latent_20251226/futr_joint_epoch_66.ckpt',
                        help='Path to FUTR checkpoint')
    parser.add_argument('--dataset-root', type=str,
                        default='/home/hice1/skim3513/scratch/darai-anticipation/FUTR_proposed/datasets/utkinect',
                        help='Path to dataset root')
    
    args = parser.parse_args()
    
    print("="*80)
    print("Segmentation Fault Diagnostic Tool")
    print("="*80)
    print("\nThis script will test each component individually to identify")
    print("the source of segmentation faults.\n")
    
    results = {}
    
    # Test 1: Imports
    results['imports'] = test_imports()
    
    # Test 2: Tokenizer from Hub
    if results['imports']:
        results['tokenizer_hub'] = test_tokenizer_hub()
    
    # Test 3: Tokenizer from local checkpoint
    if results['imports']:
        results['tokenizer_local'] = test_tokenizer_local(args.vlm_checkpoint)
    
    # Test 4: CLIP
    if results['imports']:
        results['clip'] = test_clip()
    
    # Test 5: FUTR
    if results['imports']:
        results['futr'] = test_futr(args.futr_checkpoint, args.dataset_root)
    
    # Test 6: Environment
    test_environment()
    
    # Summary
    print_section("Diagnostic Summary")
    
    for test_name, result in results.items():
        if result is True:
            status = "✓ PASS"
        elif result is False:
            status = "✗ FAIL"
        else:
            status = "⚠ SKIP"
        print(f"  {status}: {test_name}")
    
    print("\n" + "="*80)
    print("Recommendations:")
    print("="*80)
    
    if results.get('tokenizer_hub') is False:
        print("\n⚠ Tokenizer from HuggingFace Hub failed!")
        print("  Recommendation:")
        print("  1. Check internet connection")
        print("  2. Try: pip install --upgrade transformers tokenizers")
        print("  3. Use local tokenizer if available")
    
    if results.get('tokenizer_local') is False:
        print("\n⚠ Tokenizer from local checkpoint failed!")
        print("  Recommendation:")
        print("  1. Ensure checkpoint was saved with tokenizer.save_pretrained()")
        print("  2. Check that config.json and tokenizer_config.json exist")
        print("  3. Use HuggingFace Hub tokenizer as fallback")
    
    if results.get('clip') is False:
        print("\n⚠ CLIP model failed!")
        print("  Recommendation:")
        print("  1. Try: pip install --upgrade clip")
        print("  2. Check CUDA compatibility")
    
    if results.get('futr') is False:
        print("\n⚠ FUTR model failed!")
        print("  Recommendation:")
        print("  1. Check checkpoint path exists")
        print("  2. Check dataset root exists")
        print("  3. Verify checkpoint format")
    
    if all(v is True for v in results.values() if v is not None):
        print("\n✓ All tests passed!")
        print("  If you still experience segmentation faults during inference,")
        print("  the issue may be in the inference loop or data loading.")
        print("\n  Try running with minimal steps:")
        print("  python3 inference_robust.py --num-inference-steps 5 ...")
    
    print("\n" + "="*80)
    print("For more help, see: SEGFAULT_FIX_GUIDE.md")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
