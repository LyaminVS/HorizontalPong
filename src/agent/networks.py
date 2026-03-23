"""
Neural network architectures for Actor and Critic.
"""

import torch
import torch.nn as nn
import numpy as np
from run.config import ActorCriticConfig


class ActorNetwork(nn.Module):
    """Policy network pi(a | s)."""

    def __init__(
        self,
        state_dim: int = ActorCriticConfig.state_dim,
        hidden_dim: int = ActorCriticConfig.hidden_dim,
        action_dim: int = ActorCriticConfig.action_dim,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        """Ортогональная инициализация для стабильного старта RL агента."""
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0.0)
            
            # Последний слой делаем с маленьким gain, чтобы вероятности 
            # на старте были почти одинаковыми (равномерными)
            if m == self.net[-1]:
                nn.init.orthogonal_(m.weight, gain=0.01)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class CriticNetwork(nn.Module):
    """Q-value network q_hat(s, a)."""

    def __init__(
        self,
        state_dim: int = ActorCriticConfig.state_dim,
        action_dim: int = ActorCriticConfig.action_dim,
        hidden_dim: int = ActorCriticConfig.hidden_dim,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.action_dim = action_dim
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0.0)
            if m == self.net[-1]:
                nn.init.orthogonal_(m.weight, gain=1.0)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        action_one_hot = nn.functional.one_hot(action.long(), num_classes=self.action_dim).float()
        x = torch.cat([state, action_one_hot], dim=-1)
        return self.net(x)