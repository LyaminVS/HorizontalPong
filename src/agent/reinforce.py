"""
REINFORCE (Monte Carlo Policy Gradient) agent.
"""

import torch
import numpy as np
from typing import Dict, List
from torch.distributions import Categorical

from src.agent.networks import ActorNetwork
from run.config import ReinforceConfig


class ReinforceAgent:
    """REINFORCE agent with Monte Carlo return estimation."""

    def __init__(
        self,
        state_dim: int = ReinforceConfig.state_dim,
        action_dim: int = ReinforceConfig.action_dim,
        hidden_dim: int = ReinforceConfig.hidden_dim,
        gamma: float = ReinforceConfig.gamma,
        lr_actor: float = ReinforceConfig.lr_actor,
        device: str = "cpu",
    ) -> None:
        self.gamma = gamma
        self.device = torch.device(device)
        self.entropy_coeff = 0.01
        
        self.actor = ActorNetwork(state_dim, hidden_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)

        # Буферы текущего эпизода
        self.ep_log_probs: List[torch.Tensor] = []
        self.ep_entropies: List[torch.Tensor] = []
        self.ep_rewards: List[float] = []

        # Батч из 10 эпизодов для стабильности
        self.batch_size = 10
        self.episodes_in_batch = 0
        self.batch_log_probs: List[torch.Tensor] = []
        self.batch_entropies: List[torch.Tensor] = []
        self.batch_returns: List[float] = []

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
        self.batch_log_probs.extend(self.ep_log_probs)
        self.batch_entropies.extend(self.ep_entropies)
        self.batch_returns.extend(returns)
        self.episodes_in_batch += 1

        self.ep_log_probs.clear()
        self.ep_entropies.clear()
        self.ep_rewards.clear()

        # Ждем накопления батча (10 эпизодов)
        if self.episodes_in_batch < self.batch_size:
            return {}

        returns_ts = torch.FloatTensor(self.batch_returns).to(self.device)
        
        # ЧИСТЫЙ REINFORCE: мы НЕ вычитаем среднее (нет бейзлайна).
        # Но делим на std, чтобы градиенты не взорвались от наград в +/- 100
        if returns_ts.std() > 1e-5:
            returns_ts = returns_ts / (returns_ts.std() + 1e-8)
            
        log_probs_ts = torch.stack(self.batch_log_probs)
        entropies_ts = torch.stack(self.batch_entropies)
        
        policy_loss = -(log_probs_ts * returns_ts).mean()
        loss = policy_loss - self.entropy_coeff * entropies_ts.mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.optimizer.step()

        self.batch_log_probs.clear()
        self.batch_entropies.clear()
        self.batch_returns.clear()
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
        torch.save(self.actor.state_dict(), filepath)

    def load(self, filepath: str) -> None:
        self.actor.load_state_dict(torch.load(filepath, map_location=self.device, weights_only=False))