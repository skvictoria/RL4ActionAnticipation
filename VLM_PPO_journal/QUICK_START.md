# Counterfactual Action Anticipation - Quick Start Guide

## 🚀 30-Second Setup (MacBook)

```bash
cd RL4ActionAnticipation/VLM_PPO_journal
bash setup_counterfactual.sh
```

## 📝 Command Cheat Sheet

### Test Installation
```bash
# Run all compatibility tests
python3 test_counterfactual_mac.py

# Run usage examples
python3 example_counterfactual_usage.py
```

### Training Commands

#### MacBook (CPU, Testing)
```bash
python3 main.py \
    --env-name gym_cards/NumberLine-v0 \
    --model-path /path/to/llava-model \
    --use-counterfactual \
    --num-counterfactuals 3 \
    --num-processes 1 \
    --num-steps 32 \
    --mini-batch-size 1 \
    --no-cuda
```

#### GPU Server (Full Training)
```bash
python3 main.py \
    --env-name gym_cards/NumberLine-v0 \
    --model-path /path/to/llava-model \
    --use-counterfactual \
    --num-counterfactuals 5 \
    --num-processes 8 \
    --num-steps 256 \
    --mini-batch-size 8 \
    --use-wandb \
    --wandb-project "counterfactual-vlm"
```

#### Baseline (No Counterfactual)
```bash
python3 main.py \
    --env-name gym_cards/NumberLine-v0 \
    --model-path /path/to/llava-model \
    --num-processes 8 \
    --num-steps 256
```

## ⚙️ Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--use-counterfactual` | False | Enable counterfactual reasoning |
| `--num-counterfactuals` | 3 | Number of alternative actions (K) |
| `--cf-safety-threshold` | 0.3 | Min outcome score for safe actions |
| `--cf-uncertainty-penalty` | 0.1 | Penalty for high uncertainty |
| `--cf-contrastive-weight` | 0.5 | Weight for contrastive loss |
| `--cf-outcome-weight` | 0.3 | Weight for outcome prediction loss |
| `--cf-exploration-rate` | 0.1 | Random exploration probability |

## 📊 Monitoring (with wandb)

Key metrics to watch:
- `cf/mean_outcome_score`: Higher is better (0-1)
- `cf/mean_uncertainty`: Lower is better
- `cf/policy_override_rate`: % of times CF overrides policy
- `cf/safety_violation_rate`: % of unsafe predictions
- `eval/success_rate`: Task success rate

## 🔧 Troubleshooting

### Out of Memory
```bash
# Reduce batch sizes
--num-counterfactuals 2 \
--num-processes 1 \
--mini-batch-size 1 \
--num-steps 16
```

### Slow on CPU
```bash
# This is expected - move to GPU for real training
# On MacBook: just test that code works
# On Server: use full GPU training
```

### NaN in Loss
```bash
# Reduce learning rate
--init-lr 1e-7 \
--end-lr 1e-9
```

## 📁 File Structure

```
VLM_PPO_journal/
├── a2c_ppo_acktr/
│   └── counterfactual.py          # Core module
├── train_rl_counterfactual.py     # Training loop
├── test_counterfactual_mac.py     # Tests
├── example_counterfactual_usage.py # Examples
├── COUNTERFACTUAL_README.md       # Full docs
├── QUICK_START.md                 # This file
└── setup_counterfactual.sh        # Setup script
```

## 🎯 Typical Workflow

1. **MacBook**: Test implementation
   ```bash
   bash setup_counterfactual.sh
   python3 example_counterfactual_usage.py
   ```

2. **Transfer to Server**: Copy code
   ```bash
   rsync -avz RL4ActionAnticipation/ server:/path/to/project/
   ```

3. **Server**: Run experiments
   ```bash
   # Baseline
   python3 main.py [args]
   
   # Counterfactual
   python3 main.py --use-counterfactual [args]
   ```

4. **Analyze**: Compare results
   - Check wandb dashboard
   - Compare success rates
   - Analyze safety metrics

## 💡 Tips

- Start with K=3 counterfactuals, increase if needed
- Use `--cf-safety-threshold 0.3` for balanced safety/performance
- Monitor `cf/policy_override_rate` - should be 20-40%
- If override rate is too high/low, adjust safety threshold
- Use wandb for easy comparison between runs

## 📚 More Info

- Full documentation: `COUNTERFACTUAL_README.md`
- Implementation details: `COUNTERFACTUAL_IMPLEMENTATION_SUMMARY.md`
- Code examples: `example_counterfactual_usage.py`

## ✅ Verification Checklist

Before running experiments:
- [ ] Tests pass: `python3 test_counterfactual_mac.py`
- [ ] Examples run: `python3 example_counterfactual_usage.py`
- [ ] Baseline works: `python3 main.py [args without --use-counterfactual]`
- [ ] CF works: `python3 main.py --use-counterfactual [args]`
- [ ] Wandb logging works (if using)

## 🆘 Quick Help

```bash
# See all counterfactual arguments
python3 main.py --help | grep cf

# Test specific component
python3 -c "from a2c_ppo_acktr.counterfactual import *; print('✓ Import successful')"

# Check GPU availability
python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

---

**Ready to go!** Start with `bash setup_counterfactual.sh` 🚀
