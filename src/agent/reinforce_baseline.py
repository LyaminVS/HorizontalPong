"""
REINFORCE with heuristic baseline.
Uses an EMA of past episode mean returns. The scalar subtracted in the current episode
is the EMA *before* that episode is folded in (avoids same-episode leakage into b).
"""

import torch
import numpy as np
from typing import Dict, List
from torch.distributions import Categorical

from src.agent.networks import ActorNetwork
from run.config import ReinforceBaselineConfig


class ReinforceBaselineAgent:
    def __init__(
        self,
        state_dim: int = ReinforceBaselineConfig.state_dim,
        action_dim: int = ReinforceBaselineConfig.action_dim,
        hidden_dim: int = ReinforceBaselineConfig.hidden_dim,
        gamma: float = ReinforceBaselineConfig.gamma,
        lr_actor: float = ReinforceBaselineConfig.lr_actor,
        device: str = "cpu",
    ) -> None:
        self.gamma = gamma
        self.device = torch.device(device)
        self.entropy_coeff = 0.01
        
        self.actor = ActorNetwork(state_dim, hidden_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)

        # Статистический бейзлайн (EMA)
        self.baseline_ema = 0.0

        # Буферы
        self.ep_log_probs: List[torch.Tensor] = []
        self.ep_entropies: List[torch.Tensor] = []
        self.ep_rewards: List[float] = []

        self.batch_size = 10
        self.episodes_in_batch = 0
        self.batch_log_probs: List[torch.Tensor] = []
        self.batch_entropies: List[torch.Tensor] = []
        self.batch_advantages: List[float] = []

    def select_action(self, state: np.ndarray) -> int:
        state_ts = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        logits = self.actor(state_ts)
        dist = Categorical(logits=logits)
        action = dist.sample()
        
        self.ep_log_probs.append(dist.log_prob(action).squeeze())
        self.ep_entropies.append(dist.entropy().squeeze())
        return action.item()

    def store_reward(self, reward: float) -> None:
        self.ep_rewards.append(reward)

    def update(self) -> Dict[str, float]:
        if len(self.ep_rewards) == 0:
            return {}

        returns = self._compute_returns(self.ep_rewards)

        ep_mean = float(np.mean(returns))
        # Use baseline from *before* this episode (no contribution from current returns).
        baseline_old = self.baseline_ema
        advantages = [G - baseline_old for G in returns]

        # Then update EMA for future episodes.
        if self.baseline_ema == 0.0:
            self.baseline_ema = ep_mean
        else:
            self.baseline_ema = 0.99 * self.baseline_ema + 0.01 * ep_mean

        self.batch_log_probs.extend(self.ep_log_probs)
        self.batch_entropies.extend(self.ep_entropies)
        self.batch_advantages.extend(advantages)
        self.episodes_in_batch += 1

        self.ep_log_probs.clear()
        self.ep_entropies.clear()
        self.ep_rewards.clear()

        if self.episodes_in_batch < self.batch_size:
            return {}

        adv_ts = torch.FloatTensor(self.batch_advantages).to(self.device)
        
        # Нормализуем Advantage для стабильности
        if adv_ts.std() > 1e-5:
            adv_ts = adv_ts / (adv_ts.std() + 1e-8)
            
        log_probs_ts = torch.stack(self.batch_log_probs)
        entropies_ts = torch.stack(self.batch_entropies)
        
        policy_loss = -(log_probs_ts * adv_ts).mean()
        loss = policy_loss - self.entropy_coeff * entropies_ts.mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.optimizer.step()

        self.batch_log_probs.clear()
        self.batch_entropies.clear()
        self.batch_advantages.clear()
        self.episodes_in_batch = 0

        return {"actor_loss": loss.item()}

    def _compute_returns(self, rewards: List[float]) -> List[float]:
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        return returns

    def save(self, filepath: str) -> None:
        torch.save({"actor": self.actor.state_dict(), "baseline": self.baseline_ema}, filepath)

    def load(self, filepath: str) -> None:
        ckpt = torch.load(filepath, map_location=self.device, weights_only=False)
        if isinstance(ckpt, dict):
            self.actor.load_state_dict(ckpt["actor"])
            self.baseline_ema = ckpt.get("baseline", 0.0)
        else:
            self.actor.load_state_dict(ckpt)