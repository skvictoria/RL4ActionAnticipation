"""
Segmentation Fault 시나리오 테스트

각 단계별로 테스트하여 정확히 어디서 segfault가 발생하는지 확인
"""

import os
import sys

# 환경 변수 설정
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'

def print_test(name):
    print("\n" + "="*80)
    print(f"TEST: {name}")
    print("="*80)

def test_1_basic_imports():
    """기본 import 테스트"""
    print_test("1. Basic Imports")
    
    try:
        import torch
        print(f"✓ torch: {torch.__version__}")
    except Exception as e:
        print(f"✗ torch failed: {e}")
        return False
    
    try:
        import transformers
        print(f"✓ transformers: {transformers.__version__}")
    except Exception as e:
        print(f"✗ transformers failed: {e}")
        return False
    
    try:
        import clip
        print(f"✓ clip imported")
    except Exception as e:
        print(f"✗ clip failed: {e}")
        return False
    
    return True

def test_2_tokenizer_only():
    """Tokenizer만 로드"""
    print_test("2. Tokenizer Only")
    
    try:
        from transformers import AutoTokenizer
        print("Attempting to load tokenizer...")
        
        tokenizer = AutoTokenizer.from_pretrained(
            "liuhaotian/llava-v1.5-7b",
            use_fast=False
        )
        print(f"✓ Tokenizer loaded: vocab size = {len(tokenizer)}")
        
        # 간단한 tokenization 테스트
        text = "Hello, world!"
        tokens = tokenizer(text)
        print(f"✓ Tokenization works: {tokens['input_ids'][:5]}...")
        
        return True
    except Exception as e:
        print(f"✗ Tokenizer failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_3_llava_imports():
    """LLaVA 관련 import"""
    print_test("3. LLaVA Imports")
    
    sys.path.insert(0, os.path.dirname(__file__))
    
    try:
        print("Importing LLaVA constants...")
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        print("✓ LLaVA constants imported")
    except Exception as e:
        print(f"✗ LLaVA constants failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    try:
        print("Importing LLaVA conversation...")
        from llava.conversation import conv_templates
        print("✓ LLaVA conversation imported")
    except Exception as e:
        print(f"✗ LLaVA conversation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    try:
        print("Importing LLaVA model...")
        from llava.model import LlavaLlamaForCausalLM
        print("✓ LLaVA model imported")
    except Exception as e:
        print(f"✗ LLaVA model failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_4_load_base_model():
    """Base model 로드"""
    print_test("4. Load Base Model")
    
    try:
        import torch
        from transformers import AutoTokenizer
        from llava.model import LlavaLlamaForCausalLM
        
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            "liuhaotian/llava-v1.5-7b",
            use_fast=False
        )
        print("✓ Tokenizer loaded")
        
        print("Loading base model (this may take a while)...")
        model = LlavaLlamaForCausalLM.from_pretrained(
            "liuhaotian/llava-v1.5-7b",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print("✓ Base model loaded")
        
        return True
    except Exception as e:
        print(f"✗ Base model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_5_load_lora():
    """LoRA checkpoint 로드"""
    print_test("5. Load LoRA Checkpoint")
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--vlm-checkpoint', type=str, default='./vlm_checkpoints/epoch_4')
    args = parser.parse_args()
    
    if not os.path.exists(args.vlm_checkpoint):
        print(f"⚠ VLM checkpoint not found: {args.vlm_checkpoint}")
        print("  Skipping LoRA test")
        return None
    
    try:
        import torch
        from transformers import AutoTokenizer
        from llava.model import LlavaLlamaForCausalLM
        from peft import PeftModel
        
        print("Loading base model...")
        tokenizer = AutoTokenizer.from_pretrained(
            "liuhaotian/llava-v1.5-7b",
            use_fast=False
        )
        
        base = LlavaLlamaForCausalLM.from_pretrained(
            "liuhaotian/llava-v1.5-7b",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print("✓ Base model loaded")
        
        print(f"Loading LoRA from: {args.vlm_checkpoint}")
        model = PeftModel.from_pretrained(base, args.vlm_checkpoint)
        print("✓ LoRA loaded")
        
        return True
    except Exception as e:
        print(f"✗ LoRA loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_6_model_forward():
    """Model forward pass"""
    print_test("6. Model Forward Pass")
    
    try:
        import torch
        from transformers import AutoTokenizer
        from llava.model import LlavaLlamaForCausalLM
        
        print("Loading model...")
        tokenizer = AutoTokenizer.from_pretrained(
            "liuhaotian/llava-v1.5-7b",
            use_fast=False
        )
        
        model = LlavaLlamaForCausalLM.from_pretrained(
            "liuhaotian/llava-v1.5-7b",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        model.eval()
        print("✓ Model loaded")
        
        print("Testing forward pass...")
        input_ids = tokenizer("Hello", return_tensors="pt").input_ids
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)
        
        with torch.no_grad():
            outputs = model(input_ids)
        
        print(f"✓ Forward pass successful: output shape = {outputs.logits.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n" + "="*80)
    print("Segmentation Fault Scenario Testing")
    print("="*80)
    print("This script tests each step to identify where segfault occurs")
    print("="*80 + "\n")
    
    # 환경 정보
    print("Environment:")
    print(f"  PYTHONNOUSERSITE: {os.environ.get('PYTHONNOUSERSITE', 'Not set')}")
    print(f"  TOKENIZERS_PARALLELISM: {os.environ.get('TOKENIZERS_PARALLELISM', 'Not set')}")
    
    results = {}
    
    # Test 1: Basic imports
    results['basic_imports'] = test_1_basic_imports()
    if not results['basic_imports']:
        print("\n⚠ Basic imports failed. Cannot continue.")
        return
    
    # Test 2: Tokenizer only
    results['tokenizer'] = test_2_tokenizer_only()
    
    # Test 3: LLaVA imports
    results['llava_imports'] = test_3_llava_imports()
    
    # Test 4: Load base model
    print("\n⚠ WARNING: Next test will load the full model (~13GB)")
    response = input("Continue? (y/n): ")
    if response.lower() == 'y':
        results['base_model'] = test_4_load_base_model()
    else:
        print("Skipping base model test")
        results['base_model'] = None
    
    # Test 5: Load LoRA
    if results.get('base_model') is True:
        response = input("\nTest LoRA loading? (y/n): ")
        if response.lower() == 'y':
            results['lora'] = test_5_load_lora()
        else:
            results['lora'] = None
    
    # Test 6: Forward pass
    if results.get('base_model') is True:
        response = input("\nTest forward pass? (y/n): ")
        if response.lower() == 'y':
            results['forward'] = test_6_model_forward()
        else:
            results['forward'] = None
    
    # Summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    
    for test_name, result in results.items():
        if result is True:
            status = "✓ PASS"
        elif result is False:
            status = "✗ FAIL"
        else:
            status = "⚠ SKIP"
        print(f"  {status}: {test_name}")
    
    print("\n" + "="*80)
    print("Analysis")
    print("="*80)
    
    if results.get('tokenizer') is False:
        print("\n⚠ Segfault occurs during tokenizer loading!")
        print("  This suggests:")
        print("  1. Tokenizers library issue")
        print("  2. Rust/C++ extension problem")
        print("  3. Not related to flash_attn")
    
    elif results.get('llava_imports') is False:
        print("\n⚠ Segfault occurs during LLaVA imports!")
        print("  This suggests:")
        print("  1. Flash attention import issue")
        print("  2. Transformers model import issue")
        print("  3. Check the error traceback above")
    
    elif results.get('base_model') is False:
        print("\n⚠ Segfault occurs during model loading!")
        print("  This suggests:")
        print("  1. Model weights loading issue")
        print("  2. CUDA/PyTorch compatibility")
        print("  3. Memory issue")
    
    elif results.get('lora') is False:
        print("\n⚠ Segfault occurs during LoRA loading!")
        print("  This suggests:")
        print("  1. LoRA checkpoint corruption")
        print("  2. PEFT library issue")
        print("  3. Checkpoint format mismatch")
    
    elif results.get('forward') is False:
        print("\n⚠ Segfault occurs during forward pass!")
        print("  This suggests:")
        print("  1. Model execution issue")
        print("  2. CUDA kernel problem")
        print("  3. Attention mechanism issue")
    
    else:
        print("\n✓ All tests passed!")
        print("  If you still get segfault during inference,")
        print("  the issue is likely in:")
        print("  1. Data loading")
        print("  2. Environment setup")
        print("  3. Specific inference code")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
