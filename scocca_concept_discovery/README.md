# SCoCCA Concept Discovery Demo

Minimal implementation of SCoCCA (Sparse Concept Decomposition via CCA) for discovering and manipulating interpretable concepts in pretrained CLIP models.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run demo
python demo.py
```

## What it does

1. Loads CLIP model for multi-modal embeddings
2. Extracts embeddings from sample images and texts
3. Applies CCA to align vision and text embeddings
4. Uses Lasso to extract sparse concept representations
5. Discovers interpretable concepts
6. Demonstrates concept swapping
7. Shows before/after text descriptions

## Output

The demo will print:
- Discovered concepts with sparsity metrics
- Top activating images for each concept
- Concept swapping results (before/after text descriptions)

## Next Steps

If the demo works well, you can:
- Add Stable Diffusion for image generation
- Refactor into proper modules
- Add more sophisticated concept analysis
- Test on larger datasets
