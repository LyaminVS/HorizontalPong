"""
REINFORCE with heuristic baseline.

Same as vanilla REINFORCE, but subtracts a hand-crafted heuristic baseline b(s)
from the returns to reduce gradient variance:
    L_actor = -sum_t log pi(a_t | s_t) * (G_t - b(s_t))

The baseline is NOT a neural network — it is a scalar or a simple
function of the state computed by a user-defined heuristic.
No additional learnable parameters, no extra optimizer.

Uses ActorNetwork MLP(5->128->128->3) for the policy only.
"""

import torch
import numpy as np
from typing import Dict, List

from src.agent.networks import ActorNetwork
from run.config import ReinforceBaselineConfig


class ReinforceBaselineAgent:
    """
    REINFORCE agent with a heuristic (non-learned) baseline b(s).

    The baseline reduces variance of the policy gradient without
    introducing bias and without adding trainable parameters.

    The concrete heuristic is defined in compute_baseline() and can be
    swapped or tuned without changing the rest of the algorithm.
    Possible heuristic strategies (to be chosen during implementation):
        - Running average of recent episode returns.
        - Simple function of state features (e.g. distance-based estimate).
        - Exponential moving average of per-step rewards.
        - Constant fitted to historical data.

    Attributes:
        actor: ActorNetwork instance.
        gamma: discount factor (0.99).
        lr_actor: actor learning rate (3e-4).
    """

    def __init__(
        self,
        state_dim: int = ReinforceBaselineConfig.state_dim,
        action_dim: int = ReinforceBaselineConfig.action_dim,
        hidden_dim: int = ReinforceBaselineConfig.hidden_dim,
        gamma: float = ReinforceBaselineConfig.gamma,
        lr_actor: float = ReinforceBaselineConfig.lr_actor,
        device: str = "cpu",
    ) -> None:
        """
        Initialize actor network, Adam optimizer, episode trajectory storage,
        and any internal state needed by the heuristic baseline
        (e.g. running mean accumulator).

        Args:
            state_dim: state vector dimensionality.
            action_dim: number of discrete actions.
            hidden_dim: hidden layer size.
            gamma: discount factor.
            lr_actor: learning rate for actor optimizer (Adam).
            device: torch device ("cpu" or "cuda").
        """
        raise NotImplementedError

    def select_action(self, state: np.ndarray) -> int:
        """
        Sample an action from the current policy pi(a | s).

        Stores the log-probability and the state for the subsequent update.

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

    def compute_baseline(self, states: List[np.ndarray]) -> np.ndarray:
        """
        Compute the heuristic baseline value b(s_t) for each state in the episode.

        This is a non-learned, hand-crafted function. The exact formula
        is to be determined; it should provide a reasonable estimate of
        expected return given the state, using only simple computations
        (no neural networks, no gradient).

        Args:
            states: list of normalized state vectors from the episode,
                    each of shape (5,).

        Returns:
            baselines: np.ndarray of shape (T,), one scalar baseline per step.
        """
        raise NotImplementedError

    def update(self) -> Dict[str, float]:
        """
        Compute the REINFORCE + heuristic baseline update after a complete episode.

        1. Compute discounted returns G_t for each step t.
        2. Compute heuristic baselines b(s_t) via compute_baseline().
        3. Compute advantages: A_t = G_t - b(s_t).
        4. Update actor: L_actor = -sum_t log pi(a_t | s_t) * A_t.
        5. Optionally update internal baseline state (e.g. running mean).
        6. Clear episode trajectory buffers.

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
        Save actor network weights (and baseline internal state if any) to a file.

        Args:
            filepath: path to the checkpoint file (e.g. "artifacts/reinforce_bl_model.pt").
        """
        raise NotImplementedError

    def load(self, filepath: str) -> None:
        """
        Load actor network weights (and baseline internal state if any) from a file.

        Args:
            filepath: path to the checkpoint file.
        """
        raise NotImplementedError
