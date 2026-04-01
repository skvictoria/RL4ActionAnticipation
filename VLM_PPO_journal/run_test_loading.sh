#!/bin/bash
export OMP_NUM_THREADS=1
export PYTHONNOUSERSITE=1
# Test model loading to identify segfault source

echo "Testing model loading step by step..."
echo "This will help identify where the segmentation fault occurs."
echo ""

CUDA_VISIBLE_DEVICES=0 python test_model_loading.py

echo ""
echo "Test completed!"
