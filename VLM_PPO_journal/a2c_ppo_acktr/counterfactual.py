"""
Counterfactual Action Anticipation Module

This module implements counterfactual reasoning for VLM-based RL agents.
It predicts "what would happen if I take action X" to enable safer decision-making.

MacBook Compatible: Uses CPU fallbacks and efficient batching.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
import warnings


class CounterfactualPredictor(nn.Module):
    """
    Extends FUTR to predict action-conditioned futures.
    
    Given:
        - Current observation (visual features)
        - Candidate action
    Predicts:
        - Future action sequence
        - Outcome quality (safety score)
    """
    
    def __init__(self, futr_model, n_class, hidden_dim=512, device='cpu'):
        super().__init__()
        self.futr_model = futr_model
        self.n_class = n_class
        self.device = device
        
        # Action conditioning network
        # Maps action index to embedding that modulates future prediction
        self.action_encoder = nn.Sequential(
            nn.Embedding(n_class, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        ).to(device)
        
        # Outcome evaluator: predicts quality of anticipated future
        self.outcome_evaluator = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # Single score: higher = better outcome
            nn.Sigmoid()  # Normalize to [0, 1]
        ).to(device)
        
        # Uncertainty estimator (for epistemic uncertainty)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Softplus()  # Ensure positive uncertainty
        ).to(device)
        
    def forward(self, visual_features, action_indices, fg_embedding=None):
        """
        Predict counterfactual futures conditioned on actions.
        
        Args:
            visual_features: [B, T, D] - observed visual features
            action_indices: [B, K] - K candidate actions per batch
            fg_embedding: [B, T, D_fg] - optional fine-grained embeddings from VLM
            
        Returns:
            cf_futures: [B, K, n_query, n_class] - predicted future actions
            outcome_scores: [B, K] - quality scores for each counterfactual
            uncertainties: [B, K] - uncertainty estimates
        """
        B = visual_features.size(0)
        K = action_indices.size(1)
        
        # Encode candidate actions
        action_embeds = self.action_encoder(action_indices)  # [B, K, hidden_dim]
        
        # Prepare outputs
        cf_futures_list = []
        outcome_scores_list = []
        uncertainties_list = []
        
        # Process each counterfactual action
        for k in range(K):
            action_embed = action_embeds[:, k, :]  # [B, hidden_dim]
            
            # Combine with fine-grained embedding if available
            if fg_embedding is not None:
                # Broadcast action embedding to sequence length
                T = fg_embedding.size(1)
                action_seq = action_embed.unsqueeze(1).expand(-1, T, -1)  # [B, T, hidden_dim]
                
                # Concatenate or add (we'll use addition for simplicity)
                # In practice, you might want a more sophisticated fusion
                context = fg_embedding + action_seq
            else:
                context = action_embed.unsqueeze(1)  # [B, 1, hidden_dim]
            
            # Predict future with FUTR conditioned on this action
            # Note: We pass context to FUTR's decoder
            outputs = self.futr_model.model(
                visual_features, 
                query=None, 
                context=context, 
                mode='test'
            )
            
            # Extract future action predictions
            future_actions = outputs['action']  # [B, n_query, n_class]
            cf_futures_list.append(future_actions)
            
            # Evaluate outcome quality
            # Use the last hidden state from FUTR decoder
            if hasattr(outputs, 'decoder_hidden'):
                decoder_hidden = outputs['decoder_hidden'][:, -1, :]  # [B, hidden_dim]
            else:
                # Fallback: use mean of predicted action logits
                decoder_hidden = future_actions.mean(dim=1)  # [B, n_class]
                # Project to hidden_dim if needed
                if decoder_hidden.size(-1) != action_embed.size(-1):
                    decoder_hidden = F.adaptive_avg_pool1d(
                        decoder_hidden.unsqueeze(1), 
                        action_embed.size(-1)
                    ).squeeze(1)
            
            outcome_score = self.outcome_evaluator(decoder_hidden)  # [B, 1]
            outcome_scores_list.append(outcome_score)
            
            # Estimate uncertainty
            uncertainty = self.uncertainty_head(decoder_hidden)  # [B, 1]
            uncertainties_list.append(uncertainty)
        
        # Stack results
        cf_futures = torch.stack(cf_futures_list, dim=1)  # [B, K, n_query, n_class]
        outcome_scores = torch.cat(outcome_scores_list, dim=1)  # [B, K]
        uncertainties = torch.cat(uncertainties_list, dim=1)  # [B, K]
        
        return cf_futures, outcome_scores, uncertainties


class CounterfactualActionSelector:
    """
    Selects actions based on counterfactual predictions.
    Balances exploration with safety.
    """
    
    def __init__(self, 
                 n_actions: int,
                 safety_threshold: float = 0.3,
                 uncertainty_penalty: float = 0.1,
                 device: str = 'cpu'):
        self.n_actions = n_actions
        self.safety_threshold = safety_threshold
        self.uncertainty_penalty = uncertainty_penalty
        self.device = device
        
    def sample_counterfactual_actions(self, 
                                     policy_action: torch.Tensor,
                                     policy_logits: torch.Tensor,
                                     k: int = 3,
                                     temperature: float = 1.0) -> torch.Tensor:
        """
        Sample K alternative actions for counterfactual reasoning.
        
        Args:
            policy_action: [B] - action chosen by policy
            policy_logits: [B, n_actions] - policy logits
            k: number of counterfactuals to generate
            temperature: sampling temperature
            
        Returns:
            cf_actions: [B, K] - K counterfactual actions per batch
        """
        B = policy_action.size(0)
        
        # Always include the policy's chosen action as first counterfactual
        cf_actions = torch.zeros(B, k, dtype=torch.long, device=self.device)
        cf_actions[:, 0] = policy_action
        
        # Sample k-1 alternative actions from policy distribution
        if k > 1:
            probs = F.softmax(policy_logits / temperature, dim=-1)
            
            for i in range(1, k):
                # Sample without replacement (avoid duplicates)
                sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
                cf_actions[:, i] = sampled
                
                # Reduce probability of sampled action to encourage diversity
                probs[torch.arange(B), sampled] *= 0.1
                probs = probs / probs.sum(dim=-1, keepdim=True)
        
        return cf_actions
    
    def select_action(self,
                     cf_actions: torch.Tensor,
                     outcome_scores: torch.Tensor,
                     uncertainties: torch.Tensor,
                     exploration_rate: float = 0.0) -> Tuple[torch.Tensor, Dict]:
        """
        Select best action based on counterfactual predictions.
        
        Args:
            cf_actions: [B, K] - candidate actions
            outcome_scores: [B, K] - predicted outcome quality
            uncertainties: [B, K] - prediction uncertainties
            exploration_rate: probability of random exploration
            
        Returns:
            selected_actions: [B] - chosen actions
            info: dict with selection statistics
        """
        B, K = cf_actions.shape
        
        # Compute utility: outcome score - uncertainty penalty
        utility = outcome_scores - self.uncertainty_penalty * uncertainties
        
        # Apply safety filter: mask out actions with low outcome scores
        safe_mask = outcome_scores > self.safety_threshold
        utility = torch.where(safe_mask, utility, torch.full_like(utility, -1e9))
        
        # Select action with highest utility
        best_indices = utility.argmax(dim=1)  # [B]
        selected_actions = cf_actions[torch.arange(B), best_indices]
        
        # Exploration: randomly override with probability exploration_rate
        if exploration_rate > 0:
            explore_mask = torch.rand(B, device=self.device) < exploration_rate
            random_actions = torch.randint(0, self.n_actions, (B,), device=self.device)
            selected_actions = torch.where(explore_mask, random_actions, selected_actions)
        
        # Gather statistics
        info = {
            'mean_outcome_score': outcome_scores.mean().item(),
            'mean_uncertainty': uncertainties.mean().item(),
            'safety_violations': (~safe_mask.any(dim=1)).sum().item(),
            'policy_action_selected': (best_indices == 0).sum().item(),  # How often we stick with policy
        }
        
        return selected_actions, info


class CounterfactualLoss(nn.Module):
    """
    Loss function for training counterfactual prediction.
    Combines supervised learning with contrastive objectives.
    """
    
    def __init__(self, 
                 n_class: int,
                 pad_idx: int,
                 contrastive_weight: float = 0.5,
                 outcome_weight: float = 0.3):
        super().__init__()
        self.n_class = n_class
        self.pad_idx = pad_idx
        self.contrastive_weight = contrastive_weight
        self.outcome_weight = outcome_weight
        
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=pad_idx)
        self.mse_loss = nn.MSELoss()
        
    def forward(self,
                cf_futures: torch.Tensor,
                actual_future: torch.Tensor,
                outcome_scores: torch.Tensor,
                actual_reward: torch.Tensor,
                action_taken_idx: int = 0) -> Tuple[torch.Tensor, Dict]:
        """
        Compute counterfactual training loss.
        
        Args:
            cf_futures: [B, K, n_query, n_class] - predicted futures
            actual_future: [B, n_query] - ground truth future actions
            outcome_scores: [B, K] - predicted outcome scores
            actual_reward: [B] - actual reward received
            action_taken_idx: which counterfactual corresponds to actual action
            
        Returns:
            loss: scalar loss
            info: dict with loss components
        """
        B, K, n_query, n_class = cf_futures.shape
        
        # 1. Supervised loss: predict actual future correctly
        actual_cf = cf_futures[:, action_taken_idx, :, :]  # [B, n_query, n_class]
        supervised_loss = self.ce_loss(
            actual_cf.reshape(-1, n_class),
            actual_future.reshape(-1)
        )
        
        # 2. Outcome prediction loss: predict actual reward
        actual_outcome = outcome_scores[:, action_taken_idx]  # [B]
        # Normalize reward to [0, 1] for comparison with sigmoid output
        normalized_reward = torch.sigmoid(actual_reward.squeeze())
        outcome_loss = self.mse_loss(actual_outcome, normalized_reward)
        
        # 3. Contrastive loss: different actions should lead to different futures
        # Compare actual action's future with counterfactual futures
        if K > 1:
            # Flatten futures for comparison
            actual_flat = actual_cf.reshape(B, -1)  # [B, n_query * n_class]
            
            contrastive_losses = []
            for k in range(1, K):
                cf_flat = cf_futures[:, k, :, :].reshape(B, -1)
                # Encourage difference (negative similarity)
                similarity = F.cosine_similarity(actual_flat, cf_flat, dim=1)
                contrastive_losses.append(-similarity.mean())  # Maximize difference
            
            contrastive_loss = torch.stack(contrastive_losses).mean()
        else:
            contrastive_loss = torch.tensor(0.0, device=cf_futures.device)
        
        # Total loss
        total_loss = (
            supervised_loss + 
            self.outcome_weight * outcome_loss +
            self.contrastive_weight * contrastive_loss
        )
        
        info = {
            'supervised_loss': supervised_loss.item(),
            'outcome_loss': outcome_loss.item(),
            'contrastive_loss': contrastive_loss.item(),
            'total_cf_loss': total_loss.item(),
        }
        
        return total_loss, info


def create_counterfactual_module(joint_model, 
                                 n_actions: int,
                                 device: str = 'cpu',
                                 **kwargs) -> Tuple[CounterfactualPredictor, 
                                                   CounterfactualActionSelector,
                                                   CounterfactualLoss]:
    """
    Factory function to create all counterfactual components.
    
    Args:
        joint_model: JointFUTR instance
        n_actions: number of possible actions
        device: 'cpu' or 'cuda'
        **kwargs: additional config options
        
    Returns:
        predictor: CounterfactualPredictor
        selector: CounterfactualActionSelector
        loss_fn: CounterfactualLoss
    """
    # MacBook compatibility: force CPU if CUDA not available
    if device == 'cuda' and not torch.cuda.is_available():
        warnings.warn("CUDA not available, falling back to CPU")
        device = 'cpu'
    
    predictor = CounterfactualPredictor(
        futr_model=joint_model,
        n_class=n_actions,
        hidden_dim=kwargs.get('hidden_dim', 512),
        device=device
    )
    
    selector = CounterfactualActionSelector(
        n_actions=n_actions,
        safety_threshold=kwargs.get('safety_threshold', 0.3),
        uncertainty_penalty=kwargs.get('uncertainty_penalty', 0.1),
        device=device
    )
    
    loss_fn = CounterfactualLoss(
        n_class=n_actions,
        pad_idx=joint_model.pad_idx,
        contrastive_weight=kwargs.get('contrastive_weight', 0.5),
        outcome_weight=kwargs.get('outcome_weight', 0.3)
    )
    
    return predictor, selector, loss_fn
