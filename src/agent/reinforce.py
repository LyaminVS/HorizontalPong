"""
REINFORCE (Monte Carlo Policy Gradient) agent.

Uses the same ActorNetwork MLP(5->128->128->3) as Actor-Critic
for fair comparison. No critic, no replay buffer.

Update: after each complete episode using accumulated returns:
    G_t = sum_{k=0}^{T-t} gamma^k * r_{t+k}
    L = -sum_t log pi(a_t | s_t) * G_t
"""

import torch
import numpy as np
from typing import Dict, List

from src.agent.networks import ActorNetwork
from run.config import ReinforceConfig


class ReinforceAgent:
    """
    REINFORCE agent with Monte Carlo return estimation.

    Attributes:
        actor: ActorNetwork instance (shared architecture with Actor-Critic).
        gamma: discount factor (0.99).
        lr_actor: learning rate (3e-4, same as Actor-Critic for fair comparison).
    """

    def __init__(
        self,
        state_dim: int = ReinforceConfig.state_dim,
        action_dim: int = ReinforceConfig.action_dim,
        hidden_dim: int = ReinforceConfig.hidden_dim,
        gamma: float = ReinforceConfig.gamma,
        lr_actor: float = ReinforceConfig.lr_actor,
        device: str = "cpu",
    ) -> None:
        """
        Initialize actor network, Adam optimizer, and episode trajectory storage.

        Args:
            state_dim: state vector dimensionality.
            action_dim: number of discrete actions.
            hidden_dim: hidden layer size.
            gamma: discount factor.
            lr_actor: learning rate for Adam optimizer.
            device: torch device ("cpu" or "cuda").
        """
        raise NotImplementedError

    def select_action(self, state: np.ndarray) -> int:
        """
        Sample an action from the current policy pi(a | s).

        Also stores the log-probability of the selected action for the
        subsequent policy gradient update.

        Args:
            state: normalized state vector (5,).

        Returns:
            action: sampled integer action.
        """
        raise NotImplementedError

    def store_reward(self, reward: float) -> None:
        """
        Append the reward received at the current step to the episode trajectory.

        Args:
            reward: scalar reward.
        """
        raise NotImplementedError

    def update(self) -> Dict[str, float]:
        """
        Compute the REINFORCE policy gradient update after a complete episode.

        1. Compute discounted returns G_t for each step t:
           G_t = sum_{k=0}^{T-t} gamma^k * r_{t+k}
        2. Normalize returns (subtract mean, divide by std) for stability.
        3. Compute loss: L = -sum_t log pi(a_t | s_t) * G_t.
        4. Backpropagate and update actor weights.
        5. Clear the episode trajectory buffers.

        Returns:
            dict with "actor_loss": scalar loss value.
        """
        raise NotImplementedError

    def _compute_returns(self, rewards: List[float]) -> List[float]:
        """
        Compute discounted cumulative returns from a list of rewards.

        G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...

        Args:
            rewards: list of rewards [r_0, r_1, ..., r_{T-1}].

        Returns:
            list of discounted returns [G_0, G_1, ..., G_{T-1}].
        """
        raise NotImplementedError

    def save(self, filepath: str) -> None:
        """
        Save actor network weights to a file.

        Args:
            filepath: path to the checkpoint file (e.g. "artifacts/reinforce_model.pt").
        """
        raise NotImplementedError

    def load(self, filepath: str) -> None:
        """
        Load actor network weights from a file.

        Args:
            filepath: path to the checkpoint file.
        """
        raise NotImplementedError
