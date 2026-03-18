# Getting Started Checklist - Counterfactual Action Anticipation

## ✅ Implementation Status

- [x] Core counterfactual module implemented
- [x] Training loop integration complete
- [x] MacBook compatibility verified
- [x] Tests passing (5/5)
- [x] Documentation complete
- [x] Examples provided

## 📋 Your Next Steps

### Phase 1: Local Verification (MacBook) - 30 minutes

- [ ] **Step 1.1**: Navigate to project
  ```bash
  cd RL4ActionAnticipation/VLM_PPO_journal
  ```

- [ ] **Step 1.2**: Run setup script
  ```bash
  bash setup_counterfactual.sh
  ```
  Expected: All tests pass ✅

- [ ] **Step 1.3**: Run examples
  ```bash
  python3 example_counterfactual_usage.py
  ```
  Expected: 3 examples run successfully ✅

- [ ] **Step 1.4**: Read documentation
  - [ ] Quick start: `QUICK_START.md`
  - [ ] Full guide: `COUNTERFACTUAL_README.md`
  - [ ] Implementation details: `../COUNTERFACTUAL_IMPLEMENTATION_SUMMARY.md`

### Phase 2: Small Integration Test (MacBook) - 1 hour

- [ ] **Step 2.1**: Prepare a small dataset
  - Use gym_cards environment (already included)
  - Or prepare a tiny subset of UTKinect

- [ ] **Step 2.2**: Test baseline (without counterfactual)
  ```bash
  python3 main.py \
      --env-name gym_cards/NumberLine-v0 \
      --model-path /path/to/llava \
      --num-processes 1 \
      --num-steps 16 \
      --no-cuda
  ```

- [ ] **Step 2.3**: Test with counterfactual
  ```bash
  python3 main.py \
      --env-name gym_cards/NumberLine-v0 \
      --model-path /path/to/llava \
      --use-counterfactual \
      --num-counterfactuals 3 \
      --num-processes 1 \
      --num-steps 16 \
      --no-cuda
  ```

- [ ] **Step 2.4**: Verify it runs without errors
  - Check logs for counterfactual metrics
  - Verify policy override rate is reasonable (20-40%)

### Phase 3: Transfer to GPU Server - 30 minutes

- [ ] **Step 3.1**: Transfer code to server
  ```bash
  # Option 1: rsync
  rsync -avz RL4ActionAnticipation/ user@server:/path/to/project/
  
  # Option 2: git
  git add .
  git commit -m "Add counterfactual implementation"
  git push
  # Then pull on server
  ```

- [ ] **Step 3.2**: Setup on server
  ```bash
  # SSH to server
  ssh user@server
  
  # Navigate to project
  cd /path/to/project/VLM_PPO_journal
  
  # Install dependencies
  pip install -r requirements.txt
  
  # Run tests (should work on GPU too)
  python3 test_counterfactual_mac.py
  ```

- [ ] **Step 3.3**: Verify GPU availability
  ```bash
  python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"
  ```
  Expected: `CUDA: True` ✅

### Phase 4: Baseline Experiments (Server) - 1-2 days

- [ ] **Step 4.1**: Run baseline experiments (no counterfactual)
  ```bash
  python3 main.py \
      --env-name gym_cards/NumberLine-v0 \
      --model-path /path/to/llava \
      --num-processes 8 \
      --num-steps 256 \
      --use-wandb \
      --wandb-project "counterfactual-vlm" \
      --wandb-run "baseline-run1"
  ```

- [ ] **Step 4.2**: Run multiple seeds (3-5 runs)
  ```bash
  for seed in 1 2 3; do
      python3 main.py [args] --seed $seed --wandb-run "baseline-seed$seed"
  done
  ```

- [ ] **Step 4.3**: Record baseline metrics
  - Success rate: _____%
  - Sample efficiency (steps to 80% success): _____
  - Safety violations: _____%

### Phase 5: Counterfactual Experiments (Server) - 1-2 days

- [ ] **Step 5.1**: Run counterfactual experiments
  ```bash
  python3 main.py \
      --env-name gym_cards/NumberLine-v0 \
      --model-path /path/to/llava \
      --use-counterfactual \
      --num-counterfactuals 5 \
      --num-processes 8 \
      --num-steps 256 \
      --use-wandb \
      --wandb-project "counterfactual-vlm" \
      --wandb-run "cf-run1"
  ```

- [ ] **Step 5.2**: Run multiple seeds (3-5 runs)
  ```bash
  for seed in 1 2 3; do
      python3 main.py [args] --use-counterfactual --seed $seed --wandb-run "cf-seed$seed"
  done
  ```

- [ ] **Step 5.3**: Record counterfactual metrics
  - Success rate: _____%
  - Sample efficiency: _____
  - Safety violations: _____%
  - Policy override rate: _____%
  - Mean outcome score: _____

### Phase 6: Ablation Studies (Server) - 2-3 days

- [ ] **Step 6.1**: Vary number of counterfactuals
  - [ ] K=1 (no alternatives)
  - [ ] K=3
  - [ ] K=5
  - [ ] K=7

- [ ] **Step 6.2**: Vary safety threshold
  - [ ] threshold=0.1 (permissive)
  - [ ] threshold=0.3 (default)
  - [ ] threshold=0.5
  - [ ] threshold=0.7 (strict)

- [ ] **Step 6.3**: Ablate loss components
  - [ ] Without contrastive loss (`--cf-contrastive-weight 0`)
  - [ ] Without outcome loss (`--cf-outcome-weight 0`)
  - [ ] Without uncertainty penalty (`--cf-uncertainty-penalty 0`)

### Phase 7: Analysis & Visualization - 1 week

- [ ] **Step 7.1**: Compare baseline vs counterfactual
  - [ ] Plot success rate over time
  - [ ] Plot sample efficiency
  - [ ] Plot safety violations
  - [ ] Statistical significance tests

- [ ] **Step 7.2**: Analyze counterfactual behavior
  - [ ] When does CF override policy?
  - [ ] What actions are filtered as unsafe?
  - [ ] Are uncertainty estimates calibrated?

- [ ] **Step 7.3**: Create visualizations
  - [ ] Architecture diagram
  - [ ] Result plots (success rate, efficiency, safety)
  - [ ] Ablation study plots
  - [ ] Example counterfactual predictions

- [ ] **Step 7.4**: Qualitative analysis
  - [ ] Show example scenarios where CF helps
  - [ ] Show failure cases
  - [ ] Interpret learned outcome predictor

### Phase 8: Paper Writing - 2-3 weeks

- [ ] **Step 8.1**: Write method section
  - [ ] Background on VLM-RL
  - [ ] Counterfactual predictor architecture
  - [ ] Training objective
  - [ ] Action selection mechanism

- [ ] **Step 8.2**: Write experiments section
  - [ ] Experimental setup
  - [ ] Main results (baseline vs CF)
  - [ ] Ablation studies
  - [ ] Analysis and discussion

- [ ] **Step 8.3**: Write introduction
  - [ ] Motivation
  - [ ] Problem statement
  - [ ] Contributions
  - [ ] Paper organization

- [ ] **Step 8.4**: Write related work
  - [ ] VLM-based RL
  - [ ] Action anticipation
  - [ ] Counterfactual reasoning
  - [ ] Safe RL

- [ ] **Step 8.5**: Write discussion & conclusion
  - [ ] Summary of findings
  - [ ] Limitations
  - [ ] Future work
  - [ ] Broader impact

- [ ] **Step 8.6**: Polish and proofread
  - [ ] Check formatting
  - [ ] Verify citations
  - [ ] Proofread for clarity
  - [ ] Get feedback from collaborators

### Phase 9: Submission - 1 week

- [ ] **Step 9.1**: Prepare submission materials
  - [ ] PDF of paper
  - [ ] Supplementary material
  - [ ] Code release (GitHub)
  - [ ] Anonymize if required

- [ ] **Step 9.2**: Submit to NeurIPS
  - [ ] Create OpenReview account
  - [ ] Upload paper
  - [ ] Fill out submission form
  - [ ] Submit before deadline

- [ ] **Step 9.3**: Prepare for rebuttal
  - [ ] Monitor reviews
  - [ ] Prepare responses
  - [ ] Run additional experiments if needed

## 📊 Success Metrics

Track these metrics to evaluate your implementation:

### Quantitative Metrics
- [ ] Success rate improvement: Target +20-30%
- [ ] Sample efficiency improvement: Target +20-40%
- [ ] Safety violation reduction: Target -50-70%
- [ ] Policy override rate: Target 20-40%

### Qualitative Metrics
- [ ] Counterfactual predictions are diverse
- [ ] Uncertainty estimates are calibrated
- [ ] Action selection is interpretable
- [ ] System is stable during training

## 🐛 Common Issues & Solutions

### Issue: Tests fail on MacBook
**Solution**: Check Python version and dependencies
```bash
python3 --version  # Need 3.8+
pip3 install torch numpy
```

### Issue: Out of memory on MacBook
**Solution**: This is expected - MacBook is for testing only
```bash
# Use minimal settings
--num-processes 1 --num-steps 16 --mini-batch-size 1
```

### Issue: Slow training on CPU
**Solution**: Transfer to GPU server for real experiments

### Issue: NaN in loss
**Solution**: Reduce learning rate
```bash
--init-lr 1e-7 --end-lr 1e-9
```

### Issue: Counterfactual predictions all the same
**Solution**: Increase contrastive loss weight
```bash
--cf-contrastive-weight 1.0
```

## 📚 Documentation Reference

- **Quick commands**: `VLM_PPO_journal/QUICK_START.md`
- **Full guide**: `VLM_PPO_journal/COUNTERFACTUAL_README.md`
- **Implementation details**: `COUNTERFACTUAL_IMPLEMENTATION_SUMMARY.md`
- **Test results**: `VLM_PPO_journal/TEST_RESULTS.md`
- **This checklist**: `GETTING_STARTED_CHECKLIST.md`

## ✅ Completion Checklist

Mark your progress:

- [ ] Phase 1: Local verification (MacBook)
- [ ] Phase 2: Small integration test
- [ ] Phase 3: Transfer to GPU server
- [ ] Phase 4: Baseline experiments
- [ ] Phase 5: Counterfactual experiments
- [ ] Phase 6: Ablation studies
- [ ] Phase 7: Analysis & visualization
- [ ] Phase 8: Paper writing
- [ ] Phase 9: Submission

## 🎯 Timeline Estimate

- **Week 1**: Phases 1-3 (Setup and verification)
- **Week 2-3**: Phases 4-5 (Main experiments)
- **Week 4**: Phase 6 (Ablations)
- **Week 5**: Phase 7 (Analysis)
- **Week 6-8**: Phase 8 (Paper writing)
- **Week 9**: Phase 9 (Submission)

**Total**: ~9 weeks from start to submission

## 🚀 Ready to Start?

Run this command to begin:

```bash
cd RL4ActionAnticipation/VLM_PPO_journal
bash setup_counterfactual.sh
```

Then check off items in this checklist as you complete them!

Good luck with your NeurIPS submission! 🎉
