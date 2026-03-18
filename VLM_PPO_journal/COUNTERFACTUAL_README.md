# Counterfactual Action Anticipation for VLM-based RL

This extension implements counterfactual reasoning for the RL4ActionAnticipation codebase, enabling VLM agents to predict "what would happen if I take action X" before committing to an action.

## 🎯 Key Innovation

Instead of just predicting "what will happen," the agent now:
1. Generates K alternative actions (counterfactuals)
2. Predicts the outcome for each counterfactual
3. Selects the action with the best anticipated outcome
4. Learns from the difference between factual and counterfactual futures

## 📁 New Files

```
VLM_PPO_journal/
├── a2c_ppo_acktr/
│   └── counterfactual.py          # Core counterfactual module
├── train_rl_counterfactual.py     # Enhanced training loop
├── test_counterfactual_mac.py     # MacBook compatibility tests
└── COUNTERFACTUAL_README.md       # This file
```

## 🖥️ MacBook Compatibility

The implementation is designed to work on MacBook (CPU-only):

- ✅ Automatic CPU/CUDA detection
- ✅ Efficient batching to minimize memory usage
- ✅ Fallback mechanisms for missing GPU operations
- ✅ Memory-efficient tensor operations
- ✅ No hard CUDA dependencies

## 🚀 Quick Start

### 1. Test MacBook Compatibility

```bash
cd RL4ActionAnticipation/VLM_PPO_journal
python test_counterfactual_mac.py
```

Expected output:
```
============================================================
COUNTERFACTUAL MODULE - MACBOOK COMPATIBILITY TEST
============================================================

TEST 1: Device Compatibility
✓ No GPU detected - will use CPU (MacBook mode)
✓ Successfully created tensor on cpu

TEST 2: Counterfactual Predictor
✓ Created CounterfactualPredictor on cpu
✓ Forward pass successful!
...

✓ ALL TESTS PASSED - MacBook compatible!
```

### 2. Run with Counterfactual Reasoning

Add the `--use-counterfactual` flag to your training command:

```bash
python main.py \
    --env-name gym_cards/NumberLine-v0 \
    --model-path /path/to/llava-model \
    --use-counterfactual \
    --num-counterfactuals 3 \
    --cf-safety-threshold 0.3 \
    --num-processes 1 \
    --num-steps 64 \
    --no-cuda  # For MacBook
```

### 3. Monitor Training

If using wandb, you'll see new metrics:
- `cf/mean_outcome_score`: Average predicted outcome quality
- `cf/mean_uncertainty`: Average prediction uncertainty
- `cf/policy_override_rate`: How often counterfactual selection overrides policy
- `cf/safety_violation_rate`: Frequency of unsafe action predictions
- `cf/supervised_loss`: Loss for predicting actual futures
- `cf/outcome_loss`: Loss for predicting actual rewards
- `cf/contrastive_loss`: Loss encouraging different actions → different futures

## 🔧 Configuration

### Command-line Arguments

```bash
--use-counterfactual              # Enable counterfactual reasoning
--num-counterfactuals 3           # Number of alternative actions to consider (default: 3)
--cf-safety-threshold 0.3         # Minimum outcome score for safe actions (default: 0.3)
--cf-uncertainty-penalty 0.1      # Penalty for high uncertainty (default: 0.1)
--cf-contrastive-weight 0.5       # Weight for contrastive loss (default: 0.5)
--cf-outcome-weight 0.3           # Weight for outcome prediction loss (default: 0.3)
--cf-exploration-rate 0.1         # Random exploration rate (default: 0.1)
```

### Recommended Settings

**For MacBook (CPU-only):**
```bash
--num-counterfactuals 3           # Keep low to reduce compute
--num-processes 1                 # Single process for CPU
--num-steps 64                    # Smaller rollouts
--mini-batch-size 2               # Small batches
--no-cuda                         # Force CPU
```

**For GPU Server:**
```bash
--num-counterfactuals 5           # More counterfactuals for better coverage
--num-processes 8                 # Parallel environments
--num-steps 256                   # Larger rollouts
--mini-batch-size 8               # Larger batches
```

## 📊 Architecture

### 1. CounterfactualPredictor

Extends FUTR to predict action-conditioned futures:

```python
class CounterfactualPredictor(nn.Module):
    def __init__(self, futr_model, n_class, hidden_dim=512):
        # Action encoder: maps action → embedding
        self.action_encoder = nn.Embedding(n_class, hidden_dim)
        
        # Outcome evaluator: predicts quality of future
        self.outcome_evaluator = nn.Sequential(...)
        
        # Uncertainty estimator: epistemic uncertainty
        self.uncertainty_head = nn.Sequential(...)
    
    def forward(self, visual_features, action_indices, fg_embedding):
        # For each counterfactual action:
        #   1. Encode action
        #   2. Condition FUTR on action
        #   3. Predict future action sequence
        #   4. Evaluate outcome quality
        #   5. Estimate uncertainty
        return cf_futures, outcome_scores, uncertainties
```

### 2. CounterfactualActionSelector

Selects actions based on anticipated outcomes:

```python
class CounterfactualActionSelector:
    def select_action(self, cf_actions, outcome_scores, uncertainties):
        # Compute utility = outcome - uncertainty_penalty * uncertainty
        utility = outcome_scores - self.uncertainty_penalty * uncertainties
        
        # Filter unsafe actions
        safe_mask = outcome_scores > self.safety_threshold
        
        # Select action with highest utility
        return best_action, selection_info
```

### 3. CounterfactualLoss

Trains the predictor with three objectives:

```python
class CounterfactualLoss(nn.Module):
    def forward(self, cf_futures, actual_future, outcome_scores, actual_reward):
        # 1. Supervised: predict actual future correctly
        supervised_loss = CrossEntropy(cf_futures[actual_action], actual_future)
        
        # 2. Outcome: predict actual reward correctly
        outcome_loss = MSE(outcome_scores[actual_action], actual_reward)
        
        # 3. Contrastive: different actions → different futures
        contrastive_loss = -CosineSimilarity(
            cf_futures[actual_action], 
            cf_futures[other_actions]
        )
        
        return supervised_loss + outcome_loss + contrastive_loss
```

## 🔬 Research Questions to Explore

1. **Safety**: Does counterfactual reasoning reduce unsafe actions?
2. **Sample Efficiency**: Does it improve learning speed?
3. **Generalization**: Does it transfer better to new scenarios?
4. **Interpretability**: Can we visualize why certain actions are chosen?
5. **Uncertainty**: How well calibrated are the uncertainty estimates?

## 📈 Expected Results

Based on the design, you should see:

- **Improved Safety**: Fewer actions with negative outcomes
- **Better Exploration**: More diverse action selection early in training
- **Faster Convergence**: Quicker learning from counterfactual reasoning
- **Higher Success Rate**: Better task completion rates

## 🐛 Troubleshooting

### Issue: Out of Memory on MacBook

**Solution**: Reduce batch sizes and number of counterfactuals
```bash
--num-counterfactuals 2
--num-processes 1
--mini-batch-size 1
--num-steps 32
```

### Issue: Slow Training on CPU

**Solution**: This is expected. For serious experiments, move to GPU server:
```bash
# On MacBook: Test implementation
python test_counterfactual_mac.py

# On Server: Full training
python main.py --use-counterfactual [other args]
```

### Issue: NaN in Counterfactual Loss

**Solution**: Check for numerical instability
```python
# In counterfactual.py, add gradient clipping:
torch.nn.utils.clip_grad_norm_(cf_predictor.parameters(), 1.0)
```

### Issue: Counterfactual predictions all the same

**Solution**: Increase contrastive loss weight
```bash
--cf-contrastive-weight 1.0  # Increase from default 0.5
```

## 🔄 Integration with Existing Code

The counterfactual module integrates seamlessly:

```python
# In main.py, add:
if args.use_counterfactual:
    from a2c_ppo_acktr.counterfactual import create_counterfactual_module
    
    cf_predictor, cf_selector, cf_loss_fn = create_counterfactual_module(
        joint_model=joint_model,
        n_actions=env.action_space.n,
        device=device,
        **vars(args)
    )
    
    # Use counterfactual training loop
    from train_rl_counterfactual import train_with_counterfactual as train
else:
    # Use original training loop
    from train_rl import train
```

## 📝 Citation

If you use this counterfactual extension in your research, please cite:

```bibtex
@inproceedings{counterfactual-vlm-rl-2024,
  title={Counterfactual Action Anticipation for Safe VLM-based RL Agents},
  author={Your Name},
  booktitle={NeurIPS},
  year={2024}
}
```

## 🤝 Contributing

To extend this work:

1. **Add new counterfactual sampling strategies** in `CounterfactualActionSelector.sample_counterfactual_actions()`
2. **Implement different outcome evaluators** in `CounterfactualPredictor.outcome_evaluator`
3. **Try alternative loss functions** in `CounterfactualLoss`
4. **Add visualization tools** for counterfactual predictions

## 📧 Support

For questions or issues:
1. Run `python test_counterfactual_mac.py` to verify setup
2. Check the troubleshooting section above
3. Review the inline code comments in `counterfactual.py`

## ✅ Next Steps

1. ✅ Test on MacBook: `python test_counterfactual_mac.py`
2. ✅ Verify integration: Run small experiment locally
3. ⬜ Move to GPU server: Transfer code and run full experiments
4. ⬜ Analyze results: Compare with baseline RL4VLM
5. ⬜ Write paper: Document findings and insights

Good luck with your NeurIPS submission! 🚀
