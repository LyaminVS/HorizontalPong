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
        
        self.actor = ActorNetwork(state_dim, hidden_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)

        # Buffers
        self.log_probs: List[torch.Tensor] = []
        self.entropies: List[torch.Tensor] = []
        self.rewards: List[float] = []

    def select_action(self, state: np.ndarray) -> int:
        state_ts = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        logits = self.actor(state_ts)
        dist = Categorical(logits=logits)
        action = dist.sample()
        
        self.log_probs.append(dist.log_prob(action))
        self.entropies.append(dist.entropy())
        
        return action.item()

    def store_reward(self, reward: float) -> None:
        self.rewards.append(reward)

    def update(self) -> Dict[str, float]:
        if len(self.rewards) == 0:
            return {"actor_loss": 0.0}

        returns = self._compute_returns(self.rewards)
        returns_ts = torch.FloatTensor(returns).to(self.device)
        
        # НИКАКОЙ НОРМАЛИЗАЦИИ! Используем чистый Return.

        policy_loss = []
        entropy_bonus = []
        
        for log_prob, G_t, entropy in zip(self.log_probs, returns_ts, self.entropies):
            policy_loss.append(-log_prob * G_t)
            entropy_bonus.append(entropy)
            
        # Используем .sum() ! Длинный успешный эпизод должен давать сильный сигнал
        policy_loss_sum = torch.stack(policy_loss).sum()
        entropy_loss_sum = torch.stack(entropy_bonus).sum()
        
        beta = 0.01  # Коэффициент энтропии
        loss = policy_loss_sum - beta * entropy_loss_sum

        self.optimizer.zero_grad()
        loss.backward()
        
        # Клиппинг градиентов для защиты от слишком длинных эпизодов
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
        
        self.optimizer.step()

        # Очищаем буферы
        self.log_probs.clear()
        self.entropies.clear()
        self.rewards.clear()

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
        self.actor.load_state_dict(torch.load(filepath, map_location=self.device))