"""
Neural network architectures for Actor and Critic.

Actor:  MLP  5 -> 128 -> 128 -> 3, outputs action probabilities via Softmax.
Critic: MLP  8 -> 128 -> 128 -> 1, inputs [s_norm; one_hot(a)], outputs Q(s,a).

Both methods (Actor-Critic and REINFORCE) share the same Actor architecture
to ensure fair comparison.
"""

import torch
import torch.nn as nn


class ActorNetwork(nn.Module):
    """
    Policy network pi(a | s).

    Architecture:
        Linear(5 -> 128) -> ReLU -> Linear(128 -> 128) -> ReLU -> Linear(128 -> 3)
        Output passed through Softmax to produce action probabilities.

    Input:  normalized state vector s_norm of shape (batch, 5).
    Output: action probability distribution of shape (batch, 3).
    """

    def __init__(self, state_dim: int = 5, hidden_dim: int = 128, action_dim: int = 3) -> None:
        """
        Initialize the actor MLP layers.

        Args:
            state_dim: dimensionality of the normalized state (default 5).
            hidden_dim: number of hidden units per layer (default 128).
            action_dim: number of discrete actions (default 3).
        """
        raise NotImplementedError

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute action probabilities.

        Args:
            state: normalized state tensor of shape (batch, state_dim).

        Returns:
            probs: action probability tensor of shape (batch, action_dim),
                   each row sums to 1.
        """
        raise NotImplementedError


class CriticNetwork(nn.Module):
    """
    Q-value network q_hat(s, a).

    Architecture:
        Linear(8 -> 128) -> ReLU -> Linear(128 -> 128) -> ReLU -> Linear(128 -> 1)

    Input:  concatenation [s_norm; one_hot(a)] of shape (batch, 8).
    Output: scalar Q-value estimate of shape (batch, 1).
    """

    def __init__(
        self, state_dim: int = 5, action_dim: int = 3, hidden_dim: int = 128
    ) -> None:
        """
        Initialize the critic MLP layers.

        Args:
            state_dim: dimensionality of the normalized state (default 5).
            action_dim: number of discrete actions (default 3), used for one-hot size.
            hidden_dim: number of hidden units per layer (default 128).
        """
        raise NotImplementedError

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute Q(s, a).

        Args:
            state: normalized state tensor of shape (batch, state_dim).
            action: integer action tensor of shape (batch,). Will be one-hot encoded
                    internally and concatenated with state.

        Returns:
            q_value: tensor of shape (batch, 1).
        """
        raise NotImplementedError
