#!/usr/bin/env python3
"""
Example: How to integrate counterfactual reasoning into your training loop.

This is a minimal example showing the key integration points.
For full training, see main.py with --use-counterfactual flag.
"""

import torch
import numpy as np
from a2c_ppo_acktr.counterfactual import create_counterfactual_module


def example_basic_usage():
    """
    Example 1: Basic counterfactual prediction
    """
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Counterfactual Prediction")
    print("="*60)
    
    # Setup (mock objects for demonstration)
    device = 'cpu'  # MacBook compatible
    n_actions = 10
    
    # Mock FUTR model (in practice, use your trained JointFUTR)
    class MockFUTR:
        def __init__(self, device):
            self.device = device
            self.pad_idx = n_actions
            
        class MockModel:
            def __init__(self, device):
                self.device = device
                
            def __call__(self, visual_features, query=None, context=None, mode='test'):
                B = visual_features.size(0)
                return {
                    'action': torch.randn(B, 16, n_actions).to(self.device),
                    'seg': torch.randn(B, visual_features.size(1), n_actions).to(self.device),
                    'duration': torch.rand(B, 16).to(self.device)
                }
        
        def __init__(self, device):
            self.device = device
            self.pad_idx = n_actions
            self.model = self.MockModel(device)
    
    mock_futr = MockFUTR(device)
    
    # Create counterfactual components
    cf_predictor, cf_selector, cf_loss_fn = create_counterfactual_module(
        joint_model=mock_futr,
        n_actions=n_actions,
        device=device,
        safety_threshold=0.3,
        uncertainty_penalty=0.1
    )
    
    print("✓ Created counterfactual components")
    
    # Simulate agent observation
    batch_size = 2
    time_steps = 8
    feature_dim = 512
    
    visual_features = torch.randn(batch_size, time_steps, feature_dim).to(device)
    print(f"✓ Visual features: {visual_features.shape}")
    
    # Policy chooses an action
    policy_action = torch.tensor([3, 7]).to(device)  # Actions chosen by policy
    policy_logits = torch.randn(batch_size, n_actions).to(device)
    print(f"✓ Policy actions: {policy_action}")
    
    # Sample counterfactual actions
    K = 3  # Consider 3 alternatives
    cf_actions = cf_selector.sample_counterfactual_actions(
        policy_action,
        policy_logits,
        k=K
    )
    print(f"✓ Counterfactual actions: {cf_actions}")
    print(f"  Shape: {cf_actions.shape}")
    
    # Predict outcomes for each counterfactual
    cf_futures, outcome_scores, uncertainties = cf_predictor(
        visual_features,
        cf_actions,
        fg_embedding=None
    )
    
    print(f"✓ Predicted counterfactual futures:")
    print(f"  Futures shape: {cf_futures.shape}")
    print(f"  Outcome scores: {outcome_scores}")
    print(f"  Uncertainties: {uncertainties}")
    
    # Select best action based on counterfactual predictions
    selected_action, info = cf_selector.select_action(
        cf_actions,
        outcome_scores,
        uncertainties,
        exploration_rate=0.0
    )
    
    print(f"✓ Selected actions: {selected_action}")
    print(f"  Selection info: {info}")
    
    # Check if we overrode the policy
    overridden = (selected_action != policy_action).sum().item()
    print(f"  Policy overrides: {overridden}/{batch_size}")


def example_training_integration():
    """
    Example 2: How to integrate into training loop
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: Training Loop Integration")
    print("="*60)
    
    device = 'cpu'
    n_actions = 10
    
    # Mock components
    class MockFUTR:
        def __init__(self, device):
            self.device = device
            self.pad_idx = n_actions
            
        class MockModel:
            def __init__(self, device):
                self.device = device
                
            def __call__(self, visual_features, query=None, context=None, mode='test'):
                B = visual_features.size(0)
                return {
                    'action': torch.randn(B, 16, n_actions).to(self.device),
                    'seg': torch.randn(B, visual_features.size(1), n_actions).to(self.device),
                    'duration': torch.rand(B, 16).to(self.device)
                }
        
        def __init__(self, device):
            self.device = device
            self.pad_idx = n_actions
            self.model = self.MockModel(device)
    
    mock_futr = MockFUTR(device)
    
    cf_predictor, cf_selector, cf_loss_fn = create_counterfactual_module(
        joint_model=mock_futr,
        n_actions=n_actions,
        device=device
    )
    
    # Simulate training loop
    num_steps = 5
    batch_size = 2
    
    print(f"Simulating {num_steps} training steps...")
    
    for step in range(num_steps):
        print(f"\n--- Step {step+1}/{num_steps} ---")
        
        # 1. Get observation
        visual_features = torch.randn(batch_size, 8, 512).to(device)
        
        # 2. Policy predicts action
        policy_action = torch.randint(0, n_actions, (batch_size,)).to(device)
        policy_logits = torch.randn(batch_size, n_actions).to(device)
        
        # 3. Generate counterfactuals
        cf_actions = cf_selector.sample_counterfactual_actions(
            policy_action, policy_logits, k=3
        )
        
        # 4. Predict outcomes
        cf_futures, outcome_scores, uncertainties = cf_predictor(
            visual_features, cf_actions
        )
        
        # 5. Select action
        selected_action, info = cf_selector.select_action(
            cf_actions, outcome_scores, uncertainties
        )
        
        # 6. Execute action in environment (mock)
        actual_reward = torch.randn(batch_size, 1).to(device)
        actual_future = torch.randint(0, n_actions, (batch_size, 16)).to(device)
        
        # 7. Compute counterfactual loss
        cf_loss, cf_info = cf_loss_fn(
            cf_futures,
            actual_future,
            outcome_scores,
            actual_reward,
            action_taken_idx=0
        )
        
        print(f"  Policy action: {policy_action.tolist()}")
        print(f"  Selected action: {selected_action.tolist()}")
        print(f"  Outcome scores: {outcome_scores[0].tolist()}")
        print(f"  CF Loss: {cf_loss.item():.4f}")
        print(f"    - Supervised: {cf_info['supervised_loss']:.4f}")
        print(f"    - Outcome: {cf_info['outcome_loss']:.4f}")
        print(f"    - Contrastive: {cf_info['contrastive_loss']:.4f}")
        
        # 8. Backward pass (in real training)
        # cf_loss.backward()
        # optimizer.step()
    
    print("\n✓ Training loop simulation complete")


def example_safety_analysis():
    """
    Example 3: Analyzing safety with counterfactuals
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: Safety Analysis")
    print("="*60)
    
    device = 'cpu'
    n_actions = 10
    
    # Mock components
    class MockFUTR:
        def __init__(self, device):
            self.device = device
            self.pad_idx = n_actions
            
        class MockModel:
            def __init__(self, device):
                self.device = device
                
            def __call__(self, visual_features, query=None, context=None, mode='test'):
                B = visual_features.size(0)
                return {
                    'action': torch.randn(B, 16, n_actions).to(self.device),
                    'seg': torch.randn(B, visual_features.size(1), n_actions).to(self.device),
                    'duration': torch.rand(B, 16).to(self.device)
                }
        
        def __init__(self, device):
            self.device = device
            self.pad_idx = n_actions
            self.model = self.MockModel(device)
    
    mock_futr = MockFUTR(device)
    
    # Create selector with different safety thresholds
    thresholds = [0.1, 0.3, 0.5, 0.7]
    
    print("Testing different safety thresholds:")
    
    for threshold in thresholds:
        cf_predictor, cf_selector, _ = create_counterfactual_module(
            joint_model=mock_futr,
            n_actions=n_actions,
            device=device,
            safety_threshold=threshold
        )
        
        # Simulate scenario
        visual_features = torch.randn(10, 8, 512).to(device)
        policy_action = torch.randint(0, n_actions, (10,)).to(device)
        policy_logits = torch.randn(10, n_actions).to(device)
        
        cf_actions = cf_selector.sample_counterfactual_actions(
            policy_action, policy_logits, k=5
        )
        
        cf_futures, outcome_scores, uncertainties = cf_predictor(
            visual_features, cf_actions
        )
        
        selected_action, info = cf_selector.select_action(
            cf_actions, outcome_scores, uncertainties
        )
        
        print(f"\n  Threshold: {threshold:.1f}")
        print(f"    Safety violations: {info['safety_violations']}/10")
        print(f"    Policy overrides: {info['policy_action_selected']}/10 kept policy")
        print(f"    Mean outcome: {info['mean_outcome_score']:.3f}")
        print(f"    Mean uncertainty: {info['mean_uncertainty']:.3f}")


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print(" COUNTERFACTUAL ACTION ANTICIPATION - USAGE EXAMPLES")
    print("="*70)
    
    try:
        example_basic_usage()
    except Exception as e:
        print(f"\n✗ Example 1 failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        example_training_integration()
    except Exception as e:
        print(f"\n✗ Example 2 failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        example_safety_analysis()
    except Exception as e:
        print(f"\n✗ Example 3 failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print(" EXAMPLES COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("1. Run: python test_counterfactual_mac.py")
    print("2. Integrate into main.py with --use-counterfactual flag")
    print("3. Train on your dataset and analyze results")
    print("\nSee COUNTERFACTUAL_README.md for full documentation.")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
