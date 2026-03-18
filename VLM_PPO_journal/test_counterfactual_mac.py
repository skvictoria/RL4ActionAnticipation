#!/usr/bin/env python3
"""
Test script for counterfactual module on MacBook.
Tests CPU compatibility and basic functionality.

Usage:
    python test_counterfactual_mac.py
"""

import torch
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from a2c_ppo_acktr.counterfactual import (
    CounterfactualPredictor,
    CounterfactualActionSelector,
    CounterfactualLoss,
    create_counterfactual_module
)

def test_device_compatibility():
    """Test CPU/CUDA device handling"""
    print("\n" + "="*60)
    print("TEST 1: Device Compatibility")
    print("="*60)
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        print("✓ GPU detected - will use CUDA")
        device = 'cuda'
    else:
        print("✓ No GPU detected - will use CPU (MacBook mode)")
        device = 'cpu'
    
    # Test tensor creation on device
    test_tensor = torch.randn(10, 10).to(device)
    print(f"✓ Successfully created tensor on {device}")
    print(f"  Tensor device: {test_tensor.device}")
    
    return device


def test_counterfactual_predictor(device):
    """Test CounterfactualPredictor module"""
    print("\n" + "="*60)
    print("TEST 2: Counterfactual Predictor")
    print("="*60)
    
    # Mock FUTR model
    class MockFUTR:
        def __init__(self, device):
            self.device = device
            self.pad_idx = 10
            
        class MockModel:
            def __init__(self, device):
                self.device = device
                
            def __call__(self, visual_features, query=None, context=None, mode='test'):
                B = visual_features.size(0)
                n_query = 16
                n_class = 10
                
                # Return mock outputs
                return {
                    'action': torch.randn(B, n_query, n_class).to(self.device),
                    'seg': torch.randn(B, visual_features.size(1), n_class).to(self.device),
                    'duration': torch.rand(B, n_query).to(self.device)
                }
        
        def __init__(self, device):
            self.device = device
            self.pad_idx = 10
            self.model = self.MockModel(device)
    
    mock_futr = MockFUTR(device)
    
    # Create predictor
    n_class = 10
    predictor = CounterfactualPredictor(
        futr_model=mock_futr,
        n_class=n_class,
        hidden_dim=256,
        device=device
    )
    print(f"✓ Created CounterfactualPredictor on {device}")
    
    # Test forward pass
    B, T, D = 2, 8, 512  # batch=2, time=8, features=512
    K = 3  # 3 counterfactual actions
    
    visual_features = torch.randn(B, T, D).to(device)
    action_indices = torch.randint(0, n_class, (B, K)).to(device)
    
    print(f"  Input shapes:")
    print(f"    visual_features: {visual_features.shape}")
    print(f"    action_indices: {action_indices.shape}")
    
    try:
        cf_futures, outcome_scores, uncertainties = predictor(
            visual_features, 
            action_indices
        )
        
        print(f"✓ Forward pass successful!")
        print(f"  Output shapes:")
        print(f"    cf_futures: {cf_futures.shape}")
        print(f"    outcome_scores: {outcome_scores.shape}")
        print(f"    uncertainties: {uncertainties.shape}")
        
        # Verify output ranges
        assert outcome_scores.min() >= 0 and outcome_scores.max() <= 1, "Outcome scores should be in [0, 1]"
        assert uncertainties.min() >= 0, "Uncertainties should be non-negative"
        print(f"✓ Output value ranges correct")
        
        return True
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_action_selector(device):
    """Test CounterfactualActionSelector"""
    print("\n" + "="*60)
    print("TEST 3: Action Selector")
    print("="*60)
    
    n_actions = 10
    selector = CounterfactualActionSelector(
        n_actions=n_actions,
        safety_threshold=0.3,
        uncertainty_penalty=0.1,
        device=device
    )
    print(f"✓ Created CounterfactualActionSelector")
    
    # Test counterfactual action sampling
    B = 4
    K = 3
    policy_action = torch.randint(0, n_actions, (B,)).to(device)
    policy_logits = torch.randn(B, n_actions).to(device)
    
    cf_actions = selector.sample_counterfactual_actions(
        policy_action, 
        policy_logits, 
        k=K
    )
    
    print(f"✓ Sampled counterfactual actions: {cf_actions.shape}")
    assert cf_actions.shape == (B, K), f"Expected shape ({B}, {K}), got {cf_actions.shape}"
    assert (cf_actions[:, 0] == policy_action).all(), "First counterfactual should be policy action"
    print(f"✓ First counterfactual matches policy action")
    
    # Test action selection
    outcome_scores = torch.rand(B, K).to(device)
    uncertainties = torch.rand(B, K).to(device) * 0.5
    
    selected_actions, info = selector.select_action(
        cf_actions,
        outcome_scores,
        uncertainties,
        exploration_rate=0.0
    )
    
    print(f"✓ Selected actions: {selected_actions.shape}")
    print(f"  Selection info: {info}")
    assert selected_actions.shape == (B,), f"Expected shape ({B},), got {selected_actions.shape}"
    
    return True


def test_counterfactual_loss(device):
    """Test CounterfactualLoss"""
    print("\n" + "="*60)
    print("TEST 4: Counterfactual Loss")
    print("="*60)
    
    n_class = 10
    pad_idx = 10
    loss_fn = CounterfactualLoss(
        n_class=n_class,
        pad_idx=pad_idx,
        contrastive_weight=0.5,
        outcome_weight=0.3
    )
    print(f"✓ Created CounterfactualLoss")
    
    # Create mock data with requires_grad=True for backward pass
    B, K, n_query = 2, 3, 16
    cf_futures = torch.randn(B, K, n_query, n_class, requires_grad=True).to(device)
    actual_future = torch.randint(0, n_class, (B, n_query)).to(device)
    outcome_scores = torch.rand(B, K, requires_grad=True).to(device)
    actual_reward = torch.randn(B, 1).to(device)
    
    try:
        loss, info = loss_fn(
            cf_futures,
            actual_future,
            outcome_scores,
            actual_reward,
            action_taken_idx=0
        )
        
        print(f"✓ Loss computation successful!")
        print(f"  Total loss: {loss.item():.4f}")
        print(f"  Loss components:")
        for key, value in info.items():
            print(f"    {key}: {value:.4f}")
        
        # Test backward pass
        loss.backward()
        print(f"✓ Backward pass successful!")
        
        return True
        
    except Exception as e:
        print(f"✗ Loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_usage(device):
    """Test memory efficiency for MacBook"""
    print("\n" + "="*60)
    print("TEST 5: Memory Usage")
    print("="*60)
    
    try:
        import psutil
    except ImportError:
        print("⚠ psutil not installed - skipping memory test")
        print("  Install with: pip3 install psutil")
        return True
    
    import gc
    
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    
    # Create components
    class MockFUTR:
        def __init__(self, device):
            self.device = device
            self.pad_idx = 10
            
        class MockModel:
            def __init__(self, device):
                self.device = device
                
            def __call__(self, visual_features, query=None, context=None, mode='test'):
                B = visual_features.size(0)
                return {
                    'action': torch.randn(B, 16, 10).to(self.device),
                    'seg': torch.randn(B, visual_features.size(1), 10).to(self.device),
                    'duration': torch.rand(B, 16).to(self.device)
                }
        
        def __init__(self, device):
            self.device = device
            self.pad_idx = 10
            self.model = self.MockModel(device)
    
    mock_futr = MockFUTR(device)
    predictor, selector, loss_fn = create_counterfactual_module(
        joint_model=mock_futr,
        n_actions=10,
        device=device
    )
    
    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    mem_used = mem_after - mem_before
    
    print(f"✓ Memory usage:")
    print(f"  Before: {mem_before:.2f} MB")
    print(f"  After: {mem_after:.2f} MB")
    print(f"  Used: {mem_used:.2f} MB")
    
    if mem_used < 500:  # Less than 500MB
        print(f"✓ Memory usage acceptable for MacBook")
    else:
        print(f"⚠ Warning: High memory usage ({mem_used:.2f} MB)")
    
    # Cleanup
    del predictor, selector, loss_fn, mock_futr
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    return True


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("COUNTERFACTUAL MODULE - MACBOOK COMPATIBILITY TEST")
    print("="*60)
    
    results = {}
    
    # Test 1: Device compatibility
    try:
        device = test_device_compatibility()
        results['device'] = True
    except Exception as e:
        print(f"✗ Device test failed: {e}")
        results['device'] = False
        return results
    
    # Test 2: Predictor
    try:
        results['predictor'] = test_counterfactual_predictor(device)
    except Exception as e:
        print(f"✗ Predictor test failed: {e}")
        results['predictor'] = False
    
    # Test 3: Selector
    try:
        results['selector'] = test_action_selector(device)
    except Exception as e:
        print(f"✗ Selector test failed: {e}")
        results['selector'] = False
    
    # Test 4: Loss
    try:
        results['loss'] = test_counterfactual_loss(device)
    except Exception as e:
        print(f"✗ Loss test failed: {e}")
        results['loss'] = False
    
    # Test 5: Memory
    try:
        results['memory'] = test_memory_usage(device)
    except Exception as e:
        print(f"✗ Memory test failed: {e}")
        results['memory'] = False
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name.capitalize():20s}: {status}")
    
    all_passed = all(results.values())
    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED - MacBook compatible!")
    else:
        print("✗ SOME TESTS FAILED - Check errors above")
    print("="*60 + "\n")
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
