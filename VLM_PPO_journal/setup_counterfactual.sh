#!/bin/bash
# Setup script for counterfactual module on MacBook
# This script verifies the installation and runs basic tests

set -e  # Exit on error

echo "============================================================"
echo "  Counterfactual Action Anticipation - MacBook Setup"
echo "============================================================"
echo ""

# Check Python version
echo "[1/5] Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "  ✓ Python version: $python_version"

# Check if we're in the right directory
if [ ! -f "test_counterfactual_mac.py" ]; then
    echo "  ✗ Error: Please run this script from VLM_PPO_journal directory"
    exit 1
fi
echo "  ✓ In correct directory"

# Check PyTorch installation
echo ""
echo "[2/5] Checking PyTorch installation..."
python3 -c "import torch; print('  ✓ PyTorch version:', torch.__version__)" || {
    echo "  ✗ PyTorch not found. Installing..."
    pip3 install torch torchvision torchaudio
}

# Check if CUDA is available (optional on MacBook)
echo ""
echo "[3/5] Checking CUDA availability..."
cuda_available=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
if [ "$cuda_available" = "True" ]; then
    echo "  ✓ CUDA available - will use GPU"
else
    echo "  ✓ CUDA not available - will use CPU (MacBook mode)"
fi

# Check required dependencies
echo ""
echo "[4/5] Checking dependencies..."
python3 -c "import numpy" 2>/dev/null && echo "  ✓ numpy" || echo "  ✗ numpy missing (pip install numpy)"
python3 -c "import torch" 2>/dev/null && echo "  ✓ torch" || echo "  ✗ torch missing (pip install torch)"

# Check optional dependencies
python3 -c "import psutil" 2>/dev/null && echo "  ✓ psutil (optional)" || echo "  ⚠ psutil missing (optional, for memory tests: pip install psutil)"

# Run tests
echo ""
echo "[5/5] Running compatibility tests..."
echo "  This may take 1-2 minutes..."
echo ""

python3 test_counterfactual_mac.py

# Check test result
if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "  ✓ SETUP COMPLETE - All tests passed!"
    echo "============================================================"
    echo ""
    echo "Next steps:"
    echo "  1. Review COUNTERFACTUAL_README.md for documentation"
    echo "  2. Run examples: python3 example_counterfactual_usage.py"
    echo "  3. Integrate into training with --use-counterfactual flag"
    echo ""
    echo "For full training, transfer to GPU server and run:"
    echo "  python main.py --use-counterfactual --num-counterfactuals 3 [other args]"
    echo ""
else
    echo ""
    echo "============================================================"
    echo "  ✗ SETUP FAILED - Check errors above"
    echo "============================================================"
    echo ""
    echo "Common issues:"
    echo "  - Missing dependencies: pip install -r requirements.txt"
    echo "  - Python version: Requires Python 3.8+"
    echo "  - Memory: Close other applications if running out of RAM"
    echo ""
    exit 1
fi
