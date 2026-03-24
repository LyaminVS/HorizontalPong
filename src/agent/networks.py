"""
Neural network architectures for RL agents.
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
            if m == self.net[-1]:
                nn.init.orthogonal_(m.weight, gain=0.01)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class ActorCriticNetwork(nn.Module):
    """
    Unified Actor-Critic network with shared MLP backbone.

    Architecture:
        state -> [self.net  shared MLP] -> features
        features -> [self.actor  head]  -> action logits   (action_dim)
        features -> [self.critic head]  -> Q(s, ·)         (action_dim)
    """

    def __init__(
        self,
        state_dim: int = ActorCriticConfig.state_dim,
        action_dim: int = ActorCriticConfig.action_dim,
        hidden_dim: int = ActorCriticConfig.hidden_dim,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, action_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        """Ортогональная инициализация для стабильного старта RL агента."""
        if isinstance(m, nn.Linear):
            if m is self.actor:
                nn.init.orthogonal_(m.weight, gain=0.01)
            elif m is self.critic:
                nn.init.orthogonal_(m.weight, gain=1.0)
            else:
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0.0)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Q(s, a) for given state-action pairs.

        Args:
            state:  (batch, state_dim)
            action: (batch,) integer actions

        Returns:
            Q-values of shape (batch,).
        """
        features = self.net(state)
        q_all = self.critic(features)
        return q_all.gather(1, action.long().unsqueeze(-1)).squeeze(-1)

    def get_action(self, state: torch.Tensor) -> torch.Tensor:
        """
        Action logits for given states.

        Returns:
            Logits of shape (batch, action_dim).
        """
        features = self.net(state)
        return self.actor(features)