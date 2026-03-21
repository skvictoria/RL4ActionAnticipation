"""
모델 로딩 테스트 스크립트
각 단계를 개별적으로 테스트하여 segfault 원인 파악
"""

import os
import sys
import torch

# Disable tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

print("=" * 80)
print("Model Loading Test")
print("=" * 80)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Test 1: Import transformers
print("\n[Test 1] Importing transformers...")
try:
    from transformers import AutoTokenizer
    print("✓ transformers imported")
except Exception as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

# Test 2: Load tokenizer
print("\n[Test 2] Loading tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained(
        "liuhaotian/llava-v1.5-7b",
        use_fast=False,
        trust_remote_code=True
    )
    print(f"✓ Tokenizer loaded (vocab size: {len(tokenizer)})")
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Import LLaVA
print("\n[Test 3] Importing LLaVA...")
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'LLaVA'))
    from llava.model import LlavaLlamaForCausalLM
    print("✓ LLaVA imported")
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Load model (this is where segfault likely occurs)
print("\n[Test 4] Loading LLaVA model...")
print("  This may take a few minutes...")
try:
    model = LlavaLlamaForCausalLM.from_pretrained(
        "liuhaotian/llava-v1.5-7b",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    print("✓ Model loaded")
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Move to device
print("\n[Test 5] Moving model to device...")
try:
    model = model.to(device)
    print("✓ Model moved to device")
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Set to eval mode
print("\n[Test 6] Setting model to eval mode...")
try:
    model.eval()
    print("✓ Model in eval mode")
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("✓ All tests passed!")
print("=" * 80)
