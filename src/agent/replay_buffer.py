"""
Replay buffer for off-policy Actor-Critic training.

Stores transitions (s, a, r, s', done) in a fixed-size circular buffer
and supports uniform random mini-batch sampling.
"""

import numpy as np
from typing import Dict
from run.config import ActorCriticConfig


class ReplayBuffer:
    """
    Fixed-size circular replay buffer.

    Attributes:
        capacity: maximum number of stored transitions.
        size: current number of stored transitions.
    """

    def __init__(
        self,
        capacity: int = ActorCriticConfig.buffer_capacity,
        state_dim: int = ActorCriticConfig.state_dim,
    ) -> None:
        self.capacity = capacity
        self.size = 0
        self._pos = 0

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add a single transition; overwrites oldest entry when full."""
        self.states[self._pos] = state
        self.actions[self._pos] = action
        self.rewards[self._pos] = reward
        self.next_states[self._pos] = next_state
        self.dones[self._pos] = float(done)

        self._pos = (self._pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self, batch_size: int = ActorCriticConfig.batch_size
    ) -> Dict[str, np.ndarray]:
        """Uniform random mini-batch of transitions."""
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "states": self.states[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_states": self.next_states[idx],
            "dones": self.dones[idx],
        }

    def __len__(self) -> int:
        return self.size
