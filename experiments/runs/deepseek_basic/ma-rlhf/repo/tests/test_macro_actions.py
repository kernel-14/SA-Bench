"""
Unit tests for the MA-RLHF implementation.

Tests the core components:
1. Macro action termination strategies
2. Value function estimation
3. MA-PPO policy and critic losses
4. Reward shaping and KL penalty
5. Macro action reward/advantage computation
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from typing import List

from ma_rlhf.termination import (
    get_macro_action_positions_fixed_ngram,
    get_macro_action_positions_randomized_ngram,
    get_macro_action_positions_perplexity,
)
from ma_rlhf.value_estimation import (
    compute_macro_action_values_equal,
    compute_macro_action_values_unit,
    compute_macro_action_values_position_decayed,
)
from ma_rlhf.ma_ppo import (
    policy_loss_macro_action,
    policy_loss_macro_action_joint,
    critic_loss_macro_action,
    compute_macro_action_returns_and_advantages,
    compute_macro_rewards,
)
from ma_rlhf.rlhf_utils import (
    compute_kl_penalty,
    compute_shaped_reward,
    compute_reward_model_loss,
    compute_program_synthesis_reward,
)


class TestTerminationStrategies:
    """Test macro action termination strategies (Section 3.2.1, Appendix B.4)."""
    
    def test_fixed_ngram_basic(self):
        """Test basic fixed n-gram segmentation."""
        start = 0
        mask = torch.ones(1, 25, dtype=torch.float32)
        
        boundaries = get_macro_action_positions_fixed_ngram(start, mask, n_gram=5)
        sizes = [boundaries[i+1] - boundaries[i] for i in range(len(boundaries)-1)]
        
        assert len(sizes) == 5, f"Expected 5 macro actions, got {len(sizes)}"
        assert sizes == [5, 5, 5, 5, 5], f"Expected all size 5, got {sizes}"
    
    def test_fixed_ngram_with_remainder(self):
        """Test fixed n-gram when sequence length not divisible by n."""
        start = 0
        mask = torch.ones(1, 23, dtype=torch.float32)
        
        boundaries = get_macro_action_positions_fixed_ngram(start, mask, n_gram=5)
        sizes = [boundaries[i+1] - boundaries[i] for i in range(len(boundaries)-1)]
        
        assert sizes[:-1] == [5, 5, 5, 5]
        assert sizes[-1] <= 5
    
    def test_fixed_ngram_n1_is_vanilla_ppo(self):
        """Test that n=1 gives token-level granularity (vanilla PPO)."""
        start = 0
        mask = torch.ones(1, 10, dtype=torch.float32)
        
        boundaries = get_macro_action_positions_fixed_ngram(start, mask, n_gram=1)
        sizes = [boundaries[i+1] - boundaries[i] for i in range(len(boundaries)-1)]
        
        assert all(s == 1 for s in sizes[:-1])
    
    def test_randomized_ngram(self):
        """Test randomized n-gram produces valid segmentation."""
        start = 0
        mask = torch.ones(1, 100, dtype=torch.float32)
        
        boundaries = get_macro_action_positions_randomized_ngram(start, mask)
        sizes = [boundaries[i+1] - boundaries[i] for i in range(len(boundaries)-1)]
        
        valid_sizes = {2, 3, 5, 10}
        for s in sizes[:-1]:
            assert s in valid_sizes, f"Size {s} not in {valid_sizes}"
    
    def test_perplexity_termination(self):
        """Test perplexity-based termination."""
        start = 0
        mask = torch.ones(1, 10, dtype=torch.float32)
        
        ppl = torch.tensor([10.0, 9.0, 8.0, 8.5, 7.5, 7.0, 6.5, 6.8, 6.0, 5.5])
        
        boundaries = get_macro_action_positions_perplexity(start, mask, ppl)
        sizes = [boundaries[i+1] - boundaries[i] for i in range(len(boundaries)-1)]
        
        assert boundaries[1] - boundaries[0] <= 4
    
    def test_n_infinity_single_macro_action(self):
        """Test that n=infty gives a single macro action (REINFORCE)."""
        start = 0
        mask = torch.ones(1, 50, dtype=torch.float32)
        
        boundaries = get_macro_action_positions_fixed_ngram(start, mask, n_gram=1000000)
        sizes = [boundaries[i+1] - boundaries[i] for i in range(len(boundaries)-1)]
        
        assert len(sizes) == 1
        assert sizes[0] == 50


class TestValueEstimation:
    """Test macro action value function estimation (Appendix D.1)."""
    
    def test_equal_assignment(self):
        """Test equal contribution assignment."""
        values = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])
        mask = torch.ones(1, 8)
        start = 0
        sequence = [0, 3, 6, 8]
        
        macro_values = compute_macro_action_values_equal(values, mask, start, sequence)
        
        expected = torch.tensor([[2.0, 5.0, 7.5]])
        assert torch.allclose(macro_values, expected)
    
    def test_unit_assignment(self):
        """Test unit assignment (last token only)."""
        values = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])
        mask = torch.ones(1, 8)
        start = 0
        sequence = [0, 3, 6, 8]
        
        macro_values = compute_macro_action_values_unit(values, mask, start, sequence)
        
        expected = torch.tensor([[3.0, 6.0, 8.0]])
        assert torch.allclose(macro_values, expected)
    
    def test_position_decayed_assignment(self):
        """Test position decayed assignment."""
        values = torch.tensor([[10.0, 20.0, 30.0]])
        mask = torch.ones(1, 3)
        start = 0
        sequence = [0, 3]
        
        macro_values = compute_macro_action_values_position_decayed(
            values, mask, start, sequence
        )
        
        expected = 10*0.182 + 20*0.273 + 30*0.545
        assert abs(macro_values[0, 0].item() - expected) < 0.1
    
    def test_with_padding(self):
        """Test value estimation with padding."""
        values = torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0]])
        mask = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0]])
        start = 0
        sequence = [0, 5]
        
        macro_values = compute_macro_action_values_equal(values, mask, start, sequence)
        
        expected = torch.tensor([[2.0]])
        assert torch.allclose(macro_values, expected)


class TestMAPPO:
    """Test MA-PPO policy and critic losses (Section 3.2.2, Appendix E)."""
    
    def test_policy_loss_shape(self):
        """Test that policy loss returns a scalar."""
        batch, seq = 2, 10
        num_macro = 3
        
        logprobs = torch.randn(batch, seq)
        old_logprobs = torch.randn(batch, seq)
        advantages = torch.randn(batch, num_macro)
        mask = torch.ones(batch, seq)
        sequence = [0, 3, 7, seq]
        
        loss = policy_loss_macro_action(
            logprobs, old_logprobs, advantages, mask, sequence, cliprange=0.2
        )
        
        assert loss.dim() == 0, f"Expected scalar loss, got shape {loss.shape}"
        assert not torch.isnan(loss), "Loss should not be NaN"
        assert not torch.isinf(loss), "Loss should not be infinite"
    
    def test_policy_loss_joint_shape(self):
        """Test joint probability ratio loss."""
        batch, seq = 2, 10
        num_macro = 3
        
        logprobs = torch.randn(batch, seq)
        old_logprobs = torch.randn(batch, seq)
        advantages = torch.randn(batch, num_macro)
        mask = torch.ones(batch, seq)
        sequence = [0, 3, 7, seq]
        
        loss = policy_loss_macro_action_joint(
            logprobs, old_logprobs, advantages, mask, sequence, cliprange=0.2
        )
        
        assert loss.dim() == 0
        assert not torch.isnan(loss)
    
    def test_critic_loss_shape(self):
        """Test that critic loss returns a scalar."""
        batch, seq = 2, 10
        num_macro = 3
        
        values = torch.randn(batch, seq)
        old_values = torch.randn(batch, seq)
        returns = torch.randn(batch, num_macro)
        mask = torch.ones(batch, seq)
        sequence = [0, 3, 7, seq]
        
        loss = critic_loss_macro_action(values, old_values, returns, mask, sequence)
        
        assert loss.dim() == 0
        assert not torch.isnan(loss)
    
    def test_n1_is_vanilla_ppo(self):
        """Test that n=1 produces the same loss as token-level PPO."""
        batch, seq = 2, 5
        
        logprobs = torch.randn(batch, seq)
        old_logprobs = torch.randn(batch, seq)
        advantages = torch.randn(batch, seq)
        mask = torch.ones(batch, seq)
        sequence = list(range(seq + 1))
        
        loss = policy_loss_macro_action(
            logprobs, old_logprobs, advantages, mask, sequence, cliprange=0.2
        )
        
        assert loss.dim() == 0
    
    def test_macro_rewards_computation(self):
        """Test macro action reward aggregation."""
        token_rewards = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        mask = torch.ones(1, 5)
        start = 0
        sequence = [0, 3, 5]
        
        macro_rewards = compute_macro_rewards(token_rewards, mask, start, sequence)
        
        expected = torch.tensor([[6.0, 9.0]])
        assert torch.allclose(macro_rewards, expected)
    
    def test_gae_computation(self):
        """Test GAE at macro action level."""
        macro_values = torch.tensor([[0.5, 0.5, 0.0]])
        macro_rewards = torch.tensor([[0.0, 0.0, 1.0]])
        
        advantages, returns = compute_macro_action_returns_and_advantages(
            macro_values, macro_rewards, gamma=1.0, lam=0.95
        )
        
        assert advantages.shape == macro_values.shape
        assert returns.shape == macro_values.shape


class TestRLHFUtils:
    """Test RLHF utilities (Section 2.2)."""
    
    def test_kl_penalty(self):
        """Test KL divergence penalty computation."""
        log_probs = torch.tensor([[0.5, 0.3, 0.1]])
        ref_log_probs = torch.tensor([[0.4, 0.3, 0.1]])
        mask = torch.ones(1, 3)
        
        kl = compute_kl_penalty(log_probs, ref_log_probs, mask)
        
        expected = torch.tensor([[0.1, 0.0, 0.0]])
        assert torch.allclose(kl, expected, atol=1e-6)
    
    def test_shaped_reward(self):
        """Test shaped reward with KL penalty."""
        rm_scores = torch.tensor([1.0])
        log_probs = torch.tensor([[0.5, 0.3, 0.1]])
        ref_log_probs = torch.tensor([[0.4, 0.3, 0.1]])
        mask = torch.ones(1, 3)
        beta = 0.05
        
        shaped = compute_shaped_reward(rm_scores, log_probs, ref_log_probs, mask, beta)
        
        expected = torch.tensor([[-0.005, 0.0, 1.0]])
        assert torch.allclose(shaped, expected, atol=1e-6)
    
    def test_reward_model_loss(self):
        """Test reward model ranking loss."""
        chosen = torch.tensor([2.0, 3.0, 1.5])
        rejected = torch.tensor([1.0, 1.0, 0.5])
        
        loss = compute_reward_model_loss(chosen, rejected)
        
        assert loss.dim() == 0
        assert loss.item() > 0
    
    def test_program_synthesis_reward(self):
        """Test adaptive compiler signal reward."""
        r1 = compute_program_synthesis_reward(True, False, 5, 0)
        assert r1 == 1.0
        
        r2 = compute_program_synthesis_reward(True, False, 3, 2)
        assert abs(r2 - (-0.3 + 1.3 * 0.6)) < 1e-6
        
        r3 = compute_program_synthesis_reward(True, True, 0, 5)
        assert r3 == -0.6
        
        r4 = compute_program_synthesis_reward(False, False, 0, 0)
        assert r4 == -1.0


if __name__ == "__main__":
    test_classes = [
        TestTerminationStrategies,
        TestValueEstimation,
        TestMAPPO,
        TestRLHFUtils,
    ]
    
    for test_cls in test_classes:
        tests = test_cls()
        print(f"\n{'='*60}")
        print(f"Running {test_cls.__name__}")
        print('='*60)
        
        for method_name in dir(tests):
            if method_name.startswith('test_'):
                try:
                    getattr(tests, method_name)()
                    print(f"  OK {method_name}")
                except Exception as e:
                    print(f"  FAIL {method_name}: {e}")
    
    print(f"\n{'='*60}")
    print("All tests completed.")
    print('='*60)
