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
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.size = 0
        self.ptr = 0

        self.states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity,), dtype=np.int64)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.next_states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.next_actions = np.zeros((self.capacity,), dtype=np.int64)
        self.dones = np.zeros((self.capacity,), dtype=np.float32)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        next_action: int,
        done: bool = False,
    ) -> None:
        """
        Add a transition (s, a, r, s', a', done) to the buffer.

        Overwrites the oldest entry when the buffer is full (circular).

        Args:
            state: normalized state vector (5,).
            action: integer action taken.
            reward: received reward.
            next_state: normalized next state vector (5,).
            next_action: action taken in the next state.
            done: whether s' is a terminal state (no bootstrapping).
        """
        self.states[self.ptr] = np.asarray(state, dtype=np.float32)
        self.actions[self.ptr] = int(action)
        self.rewards[self.ptr] = float(reward)
        self.next_states[self.ptr] = np.asarray(next_state, dtype=np.float32)
        self.next_actions[self.ptr] = int(next_action)
        self.dones[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

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
        if self.size == 0:
            raise ValueError("ReplayBuffer is empty.")
        bs = min(int(batch_size), self.size)
        idx = np.random.randint(0, self.size, size=bs)
        return {
            "states": self.states[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_states": self.next_states[idx],
            "next_actions": self.next_actions[idx],
            "dones": self.dones[idx],
        }

    def __len__(self) -> int:
        """
        Return the current number of stored transitions.
        """
        return self.size
