"""
Replay buffer for off-policy training of the Critic in Actor-Critic.

Stores SARSA-style transitions (s, a, r, s', a') and supports
uniform random sampling of mini-batches.

Capacity M = 10,000 transitions (configurable).
"""

import numpy as np
from typing import Tuple, Dict
from run.config import ActorCriticConfig


class ReplayBuffer:
    """
    Fixed-size circular replay buffer storing (s, a, r, s_next, a_next) transitions.

    Attributes:
        capacity: maximum number of transitions to store (default 10_000).
        size: current number of stored transitions.
    """

    def __init__(
        self,
        capacity: int = ActorCriticConfig.buffer_capacity,
        state_dim: int = ActorCriticConfig.state_dim,
    ) -> None:
        """
        Allocate pre-sized numpy arrays for each component of the transition.

        Args:
            capacity: maximum buffer size M.
            state_dim: dimensionality of the state vector.
        """
        raise NotImplementedError

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        next_action: int,
    ) -> None:
        """
        Add a single SARSA transition (s, a, r, s', a') to the buffer.

        Overwrites the oldest entry when the buffer is full (circular).

        Args:
            state: normalized state vector (5,).
            action: integer action taken.
            reward: received reward.
            next_state: normalized next state vector (5,).
            next_action: action taken in the next state.
        """
        raise NotImplementedError

    def sample(self, batch_size: int = ActorCriticConfig.batch_size) -> Dict[str, np.ndarray]:
        """
        Sample a random mini-batch of transitions.

        Args:
            batch_size: number of transitions to sample (B=64).

        Returns:
            dict with keys:
                "states":       np.ndarray (batch_size, state_dim)
                "actions":      np.ndarray (batch_size,)  int
                "rewards":      np.ndarray (batch_size,)  float
                "next_states":  np.ndarray (batch_size, state_dim)
                "next_actions": np.ndarray (batch_size,)  int
        """
        raise NotImplementedError

    def __len__(self) -> int:
        """
        Return the current number of stored transitions.
        """
        raise NotImplementedError
