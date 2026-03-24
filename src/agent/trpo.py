"""
TRPO agent (without value network baseline).
"""

from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from run.config import TRPOConfig

class _PolicyMLP(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)

class TRPOAgent:
    def __init__(
        self,
        state_dim: int = TRPOConfig.state_dim,
        action_dim: int = TRPOConfig.action_dim,
        hidden_dim: int = TRPOConfig.hidden_dim,
        gamma: float = TRPOConfig.gamma,
        max_kl: float = TRPOConfig.max_kl,
        entropy_coeff: float = TRPOConfig.entropy_coeff,
        grad_clip_norm: float = TRPOConfig.grad_clip_norm,
        damping: float = TRPOConfig.damping,
        cg_iters: int = TRPOConfig.cg_iters,
        backtrack_iters: int = TRPOConfig.backtrack_iters,
        backtrack_coeff: float = TRPOConfig.backtrack_coeff,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.gamma = gamma
        self.max_kl = max_kl
        self.entropy_coeff = entropy_coeff
        self.grad_clip_norm = grad_clip_norm
        self.damping = damping
        self.cg_iters = cg_iters
        self.backtrack_iters = backtrack_iters
        self.backtrack_coeff = backtrack_coeff

        self.policy = _PolicyMLP(state_dim, hidden_dim, action_dim).to(self.device)

        self._states: List[np.ndarray] = []
        self._actions: List[int] = []
        self._rewards: List[float] = []
        self._log_probs: List[float] = []

        self.batch_size = 10
        self._episodes_in_batch = 0
        self._batch_states: List[np.ndarray] = []
        self._batch_actions: List[int] = []
        self._batch_returns: List[float] = []
        self._batch_log_probs: List[float] = []

    def select_action(self, state: np.ndarray) -> int:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits = self.policy(state_t)
        dist = Categorical(logits=logits)
        action_t = dist.sample()
        log_prob_t = dist.log_prob(action_t)

        self._states.append(np.asarray(state, dtype=np.float32))
        self._actions.append(int(action_t.item()))
        self._log_probs.append(float(log_prob_t.item()))
        return int(action_t.item())

    def store_reward(self, reward: float) -> None:
        self._rewards.append(float(reward))

    def update(self) -> Dict[str, float]:
        if not self._rewards:
            return {}

        returns = self._compute_returns(self._rewards)
        self._batch_states.extend(self._states)
        self._batch_actions.extend(self._actions)
        self._batch_returns.extend(returns)
        self._batch_log_probs.extend(self._log_probs)
        self._episodes_in_batch += 1

        self._states.clear()
        self._actions.clear()
        self._rewards.clear()
        self._log_probs.clear()

        if self._episodes_in_batch < self.batch_size:
            return {}

        states = torch.as_tensor(np.asarray(self._batch_states), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(np.asarray(self._batch_actions), dtype=torch.long, device=self.device)
        old_log_probs = torch.as_tensor(np.asarray(self._batch_log_probs), dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(np.asarray(self._batch_returns), dtype=torch.float32, device=self.device)

        advantages = returns_t - returns_t.mean()
        if advantages.std() > 1e-5:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        with torch.no_grad():
            old_logits = self.policy(states)
            old_dist = Categorical(logits=old_logits)

        def surrogate_loss() -> torch.Tensor:
            logits = self.policy(states)
            dist = Categorical(logits=logits)
            log_probs = dist.log_prob(actions)
            ratio = torch.exp(log_probs - old_log_probs)
            entropy_bonus = dist.entropy().mean()
            return -(ratio * advantages).mean() - self.entropy_coeff * entropy_bonus

        loss_before = surrogate_loss()
        grads = torch.autograd.grad(loss_before, self.policy.parameters())
        loss_grad = self._flat_concat(grads).detach()
        loss_grad_norm = torch.norm(loss_grad)
        
        if self.grad_clip_norm is not None and self.grad_clip_norm > 0:
            clip_coef = min(1.0, self.grad_clip_norm / (loss_grad_norm.item() + 1e-8))
            loss_grad = loss_grad * clip_coef

        if torch.norm(loss_grad) < 1e-10:
            self._clear_batch()
            return {"surrogate_before": float(-loss_before.item()), "kl": 0.0}

        step_dir = self._conjugate_gradient(
            lambda v: self._fisher_vector_product(v, states, old_dist),
            b=-loss_grad,
            nsteps=self.cg_iters,
        )

        fvp_step = self._fisher_vector_product(step_dir, states, old_dist)
        shs = 0.5 * torch.dot(step_dir, fvp_step)
        scale = torch.sqrt(torch.tensor(self.max_kl, device=self.device) / (shs + 1e-8))
        full_step = step_dir * scale

        old_params = self._get_flat_params().detach()
        expected_improve = torch.dot(-loss_grad, full_step).item()

        success = False
        kl_after = 0.0
        loss_after = loss_before

        for i in range(self.backtrack_iters):
            frac = self.backtrack_coeff ** i
            new_params = old_params + frac * full_step
            self._set_flat_params(new_params)

            with torch.no_grad():
                cur_loss = surrogate_loss()
                cur_kl = self._mean_kl(states, old_dist).item()

            actual_improve = (loss_before - cur_loss).item()
            if actual_improve > 0 and cur_kl <= self.max_kl:
                success = True
                loss_after = cur_loss
                kl_after = cur_kl
                break

        if not success:
            self._set_flat_params(old_params)
            kl_after = self._mean_kl(states, old_dist).item()
            loss_after = loss_before

        with torch.no_grad():
            entropy_after = Categorical(logits=self.policy(states)).entropy().mean().item()

        self._clear_batch()
        return {
            "surrogate_before": float(-loss_before.item()),
            "surrogate_after": float(-loss_after.item()),
            "kl": float(kl_after),
            "expected_improve": float(expected_improve),
            "entropy": float(entropy_after),
            "grad_norm": float(loss_grad_norm.item()),
        }

    def _clear_batch(self):
        self._batch_states.clear()
        self._batch_actions.clear()
        self._batch_returns.clear()
        self._batch_log_probs.clear()
        self._episodes_in_batch = 0

    def save(self, filepath: str) -> None:
        torch.save({"policy_state_dict": self.policy.state_dict()}, filepath)

    def load(self, filepath: str) -> None:
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])

    def _compute_returns(self, rewards: List[float]) -> np.ndarray:
        returns = np.zeros(len(rewards), dtype=np.float32)
        running = 0.0
        for t in reversed(range(len(rewards))):
            running = rewards[t] + self.gamma * running
            returns[t] = running
        return returns

    def _fisher_vector_product(self, vector: torch.Tensor, states: torch.Tensor, old_dist: Categorical) -> torch.Tensor:
        new_logits = self.policy(states)
        new_dist = Categorical(logits=new_logits)
        kl = torch.distributions.kl.kl_divergence(old_dist, new_dist).mean()
        kl_grads = torch.autograd.grad(kl, self.policy.parameters(), create_graph=True)
        flat_kl_grads = self._flat_concat(kl_grads)
        kl_v = (flat_kl_grads * vector).sum()
        hvp = torch.autograd.grad(kl_v, self.policy.parameters(), retain_graph=False)
        flat_hvp = self._flat_concat(hvp).detach()
        return flat_hvp + self.damping * vector

    def _mean_kl(self, states: torch.Tensor, old_dist: Categorical) -> torch.Tensor:
        new_dist = Categorical(logits=self.policy(states))
        return torch.distributions.kl.kl_divergence(old_dist, new_dist).mean()

    def _conjugate_gradient(self, Avp_fn, b: torch.Tensor, nsteps: int) -> torch.Tensor:
        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        rdotr = torch.dot(r, r)
        for _ in range(nsteps):
            Avp = Avp_fn(p)
            alpha = rdotr / (torch.dot(p, Avp) + 1e-8)
            x = x + alpha * p
            r = r - alpha * Avp
            new_rdotr = torch.dot(r, r)
            if new_rdotr < 1e-10: break
            beta = new_rdotr / (rdotr + 1e-8)
            p = r + beta * p
            rdotr = new_rdotr
        return x

    def _flat_concat(self, tensors) -> torch.Tensor:
        return torch.cat([t.reshape(-1) for t in tensors])

    def _get_flat_params(self) -> torch.Tensor:
        return torch.cat([p.data.view(-1) for p in self.policy.parameters()])

    def _set_flat_params(self, flat_params: torch.Tensor) -> None:
        idx = 0
        for p in self.policy.parameters():
            n = p.numel()
            p.data.copy_(flat_params[idx : idx + n].view_as(p))
            idx += n