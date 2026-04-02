"""
Safe Inference Script without Flash Attention

Flash Attention 없이 안전하게 실행하는 inference 스크립트
"""

import os
import sys

# Segmentation fault 방지 환경 변수
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

# Flash Attention 비활성화 (중요!)
os.environ['DISABLE_FLASH_ATTN'] = '1'

# PYTHONNOUSERSITE 제거
if 'PYTHONNOUSERSITE' in os.environ:
    print("⚠ WARNING: PYTHONNOUSERSITE detected! Removing...")
    del os.environ['PYTHONNOUSERSITE']

print("\n" + "="*80)
print("Safe Inference (Flash Attention Disabled)")
print("="*80)
print("This script runs inference without flash_attn to avoid segmentation faults")
print("="*80 + "\n")

# Add project paths
sys.path.insert(0, os.path.dirname(__file__))

# Import inference_robust without flash_attn issues
from inference_robust import *

if __name__ == "__main__":
    main()
