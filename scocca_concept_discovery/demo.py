#!/usr/bin/env python3
"""
SCoCCA Concept Discovery Demo

Minimal implementation demonstrating:
1. CLIP embedding extraction
2. CCA-based multi-modal alignment
3. Sparse concept extraction via Lasso
4. Concept swapping and visualization
"""

import os
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from sklearn.cross_decomposition import CCA
from sklearn.linear_model import Lasso
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Configuration
# ============================================================================

CONFIG = {
    'clip_model': 'openai/clip-vit-base-patch32',
    'n_cca_components': 64,
    'lasso_alpha': 0.1,
    'n_concepts': 5,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}


# ============================================================================
# CLIP Encoder
# ============================================================================

class CLIPEncoder:
    """Extract embeddings using pretrained CLIP model"""
    
    def __init__(self, model_name: str, device: str):
        print(f"Loading CLIP model: {model_name}...")
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        print(f"✓ CLIP model loaded on {device}")
    
    def encode_images(self, images: List[Image.Image]) -> np.ndarray:
        """Encode batch of images to embeddings"""
        with torch.no_grad():
            inputs = self.processor(images=images, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            image_features = self.model.get_image_features(**inputs)
            # Normalize embeddings
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy()
    
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encode batch of texts to embeddings"""
        with torch.no_grad():
            inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            text_features = self.model.get_text_features(**inputs)
            # Normalize embeddings
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy()
    
    def find_nearest_texts(self, embedding: np.ndarray, candidate_texts: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """Find nearest text descriptions to an embedding"""
        text_embeddings = self.encode_texts(candidate_texts)
        
        # Compute cosine similarity
        embedding = embedding / np.linalg.norm(embedding)
        similarities = text_embeddings @ embedding
        
        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = [(candidate_texts[i], float(similarities[i])) for i in top_indices]
        return results


# ============================================================================
# Concept Discovery
# ============================================================================

def apply_cca(vision_embeddings: np.ndarray, text_embeddings: np.ndarray, n_components: int) -> Tuple[CCA, np.ndarray, np.ndarray]:
    """Apply Canonical Correlation Analysis"""
    print(f"\nApplying CCA with {n_components} components...")
    cca = CCA(n_components=n_components)
    vision_cca = cca.fit_transform(vision_embeddings, text_embeddings)[0]
    text_cca = cca.transform(text_embeddings)
    print(f"✓ CCA completed. Vision shape: {vision_cca.shape}, Text shape: {text_cca.shape}")
    return cca, vision_cca, text_cca


def extract_sparse_concepts(embeddings: np.ndarray, alpha: float, n_concepts: int) -> Tuple[np.ndarray, List[float]]:
    """Extract sparse concept representations using Lasso"""
    print(f"\nExtracting sparse concepts (alpha={alpha})...")
    
    concepts = []
    sparsities = []
    
    for i in range(n_concepts):
        # Use Lasso to find sparse representation
        # Target: reconstruct embeddings with sparse coefficients
        lasso = Lasso(alpha=alpha, max_iter=1000)
        
        # Create a simple target (e.g., principal directions)
        if i < embeddings.shape[1]:
            target = embeddings[:, i]
            lasso.fit(embeddings, target)
            
            concept_vector = lasso.coef_
            sparsity = np.sum(np.abs(concept_vector) < 1e-5) / len(concept_vector)
            
            concepts.append(concept_vector)
            sparsities.append(sparsity)
    
    concepts = np.array(concepts)
    print(f"✓ Extracted {n_concepts} concepts")
    
    return concepts, sparsities


def visualize_concepts(concepts: np.ndarray, sparsities: List[float], 
                       vision_embeddings: np.ndarray, image_names: List[str]):
    """Visualize discovered concepts"""
    print("\n" + "="*70)
    print("DISCOVERED CONCEPTS")
    print("="*70)
    
    for i, (concept, sparsity) in enumerate(zip(concepts, sparsities)):
        print(f"\nConcept {i}:")
        print(f"  Sparsity: {sparsity:.2%} (ratio of zero coefficients)")
        print(f"  Non-zero dims: {np.sum(np.abs(concept) > 1e-5)}/{len(concept)}")
        
        # Find images that activate this concept most
        activations = vision_embeddings @ concept
        top_indices = np.argsort(np.abs(activations))[::-1][:3]
        print(f"  Top activating images: {[image_names[idx] for idx in top_indices]}")
        print(f"  Activation values: {[f'{activations[idx]:.3f}' for idx in top_indices]}")


# ============================================================================
# Concept Swapping
# ============================================================================

def swap_concept(source_embedding: np.ndarray, concepts: np.ndarray, 
                 source_concept_idx: int, target_concept_idx: int, 
                 strength: float = 1.0) -> np.ndarray:
    """Swap one concept with another in an embedding"""
    
    # Decompose source embedding into concept coefficients
    # Simple projection onto concept space
    coefficients = source_embedding @ concepts.T
    
    # Swap the specified concept
    original_coef = coefficients[source_concept_idx]
    target_coef = coefficients[target_concept_idx]
    
    coefficients[source_concept_idx] = original_coef * (1 - strength) + target_coef * strength
    
    # Reconstruct embedding
    swapped_embedding = coefficients @ concepts
    
    # Normalize
    swapped_embedding = swapped_embedding / np.linalg.norm(swapped_embedding)
    
    return swapped_embedding


def demonstrate_concept_swapping(encoder: CLIPEncoder, concepts: np.ndarray, 
                                test_embedding: np.ndarray, test_name: str,
                                candidate_texts: List[str]):
    """Demonstrate concept swapping with before/after text descriptions"""
    print("\n" + "="*70)
    print("CONCEPT SWAPPING DEMONSTRATION")
    print("="*70)
    
    print(f"\nTest image: {test_name}")
    
    # Original embedding
    print("\n[BEFORE SWAP]")
    print("Finding nearest text descriptions to original embedding...")
    nearest_before = encoder.find_nearest_texts(test_embedding, candidate_texts, top_k=5)
    for text, sim in nearest_before:
        print(f"  {sim:.3f}: {text}")
    
    # Swap concept 0 with concept 2
    source_idx, target_idx = 0, 2
    print(f"\nSwapping Concept {source_idx} → Concept {target_idx}...")
    swapped_embedding = swap_concept(test_embedding, concepts, source_idx, target_idx, strength=0.8)
    
    # Swapped embedding
    print("\n[AFTER SWAP]")
    print("Finding nearest text descriptions to swapped embedding...")
    nearest_after = encoder.find_nearest_texts(swapped_embedding, candidate_texts, top_k=5)
    for text, sim in nearest_after:
        print(f"  {sim:.3f}: {text}")
    
    # Compare
    print("\n[COMPARISON]")
    print(f"Top match changed: '{nearest_before[0][0]}' → '{nearest_after[0][0]}'")
    
    # Compute embedding difference
    diff_norm = np.linalg.norm(swapped_embedding - test_embedding)
    print(f"Embedding L2 distance: {diff_norm:.4f}")
    print("✓ Concept swap completed!")


# ============================================================================
# Main Demo
# ============================================================================

def create_sample_data() -> Tuple[List[Image.Image], List[str], List[str]]:
    """Create sample images and texts for demo"""
    print("\nCreating sample data...")
    
    # Create simple colored images as samples
    images = []
    image_names = []
    colors = [
        ('red', (255, 0, 0)),
        ('green', (0, 255, 0)),
        ('blue', (0, 0, 255)),
        ('yellow', (255, 255, 0)),
        ('purple', (128, 0, 128)),
        ('orange', (255, 165, 0)),
        ('pink', (255, 192, 203)),
        ('cyan', (0, 255, 255)),
        ('brown', (165, 42, 42)),
        ('gray', (128, 128, 128)),
    ]
    
    for name, color in colors:
        img = Image.new('RGB', (224, 224), color)
        images.append(img)
        image_names.append(f"{name}_square")
    
    # Corresponding captions
    captions = [
        f"a {name} colored square"
        for name, _ in colors
    ]
    
    # Candidate texts for nearest neighbor search
    candidate_texts = [
        "a red object", "a blue object", "a green object",
        "a yellow object", "a purple object", "an orange object",
        "a warm color", "a cool color", "a bright color", "a dark color",
        "a primary color", "a secondary color",
        "red square", "blue square", "green square",
        "colorful image", "monochrome image",
        "vibrant", "muted", "saturated", "desaturated",
    ]
    
    print(f"✓ Created {len(images)} sample images")
    return images, captions, image_names, candidate_texts


def main():
    print("="*70)
    print("SCoCCA CONCEPT DISCOVERY DEMO")
    print("="*70)
    print(f"Device: {CONFIG['device']}")
    print(f"CCA components: {CONFIG['n_cca_components']}")
    print(f"Lasso alpha: {CONFIG['lasso_alpha']}")
    print(f"Number of concepts: {CONFIG['n_concepts']}")
    
    # Initialize CLIP encoder
    encoder = CLIPEncoder(CONFIG['clip_model'], CONFIG['device'])
    
    # Create sample data
    images, captions, image_names, candidate_texts = create_sample_data()
    
    # Extract embeddings
    print("\nExtracting CLIP embeddings...")
    vision_embeddings = encoder.encode_images(images)
    text_embeddings = encoder.encode_texts(captions)
    print(f"✓ Vision embeddings: {vision_embeddings.shape}")
    print(f"✓ Text embeddings: {text_embeddings.shape}")
    
    # Apply CCA
    cca, vision_cca, text_cca = apply_cca(vision_embeddings, text_embeddings, CONFIG['n_cca_components'])
    
    # Extract sparse concepts
    concepts, sparsities = extract_sparse_concepts(vision_cca, CONFIG['lasso_alpha'], CONFIG['n_concepts'])
    
    # Visualize concepts
    visualize_concepts(concepts, sparsities, vision_cca, image_names)
    
    # Demonstrate concept swapping
    test_embedding = vision_cca[0]  # Use first image as test
    test_name = image_names[0]
    demonstrate_concept_swapping(encoder, concepts, test_embedding, test_name, candidate_texts)
    
    # Save concepts
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "concepts.npy"), concepts)
    np.save(os.path.join(output_dir, "sparsities.npy"), sparsities)
    print(f"\n✓ Concepts saved to {output_dir}/")
    
    print("\n" + "="*70)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("="*70)


if __name__ == "__main__":
    main()
