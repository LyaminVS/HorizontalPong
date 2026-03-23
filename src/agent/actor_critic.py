"""
Actor-Critic agent with Q(s,a)-critic (MLP).

Critic: trained off-policy via replay buffer using SARSA-style TD error.
Actor:  trained on-policy using advantage = q_hat(s,a) - V^pi(s) with entropy bonus.

Update schedule:
    - Every N=10 steps: sample batch B=64 from replay buffer, update critic.
    - Every N=10 steps: update actor on accumulated on-policy transitions.
"""

import torch
import numpy as np
from typing import Dict, Optional

from src.agent.networks import ActorNetwork, CriticNetwork
from src.agent.replay_buffer import ReplayBuffer


class ActorCriticAgent:
    """
    Actor-Critic agent combining on-policy actor updates with
    off-policy critic training via replay buffer.

    Attributes:
        actor: ActorNetwork instance.
        critic: CriticNetwork instance.
        replay_buffer: ReplayBuffer instance.
        gamma: discount factor (0.99).
        lr_actor: actor learning rate (3e-4).
        lr_critic: critic learning rate (1e-4).
        entropy_coeff: entropy bonus coefficient beta (0.01).
        update_every: number of steps between updates N (10).
        batch_size: critic mini-batch size B (64).
    """

    def __init__(
        self,
        state_dim: int = 5,
        action_dim: int = 3,
        hidden_dim: int = 128,
        gamma: float = 0.99,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-4,
        entropy_coeff: float = 0.01,
        buffer_capacity: int = 10_000,
        batch_size: int = 64,
        update_every: int = 10,
        device: str = "cpu",
    ) -> None:
        """
        Initialize actor network, critic network, replay buffer,
        Adam optimizers, and hyperparameters.

        Args:
            state_dim: state vector dimensionality.
            action_dim: number of discrete actions.
            hidden_dim: hidden layer size for both networks.
            gamma: discount factor.
            lr_actor: learning rate for actor optimizer (Adam).
            lr_critic: learning rate for critic optimizer (Adam).
            entropy_coeff: entropy regularization coefficient beta.
            buffer_capacity: replay buffer capacity M.
            batch_size: mini-batch size B for critic updates.
            update_every: update frequency N (in environment steps).
            device: torch device ("cpu" or "cuda").
        """
        raise NotImplementedError

    def select_action(self, state: np.ndarray) -> int:
        """
        Sample an action from the current policy pi(a | s).

        Args:
            state: normalized state vector (5,).

        Returns:
            action: sampled integer action from Categorical(pi(. | s)).
        """
        raise NotImplementedError

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        next_action: int,
    ) -> None:
        """
        Store a SARSA transition (s, a, r, s', a') in both:
        - The replay buffer (for off-policy critic training).
        - The on-policy trajectory buffer (for actor updates).

        Args:
            state, action, reward, next_state, next_action: transition components.
        """
        raise NotImplementedError

    def update(self, step: int) -> Optional[Dict[str, float]]:
        """
        Perform a learning update if step is a multiple of update_every.

        1. Sample batch from replay buffer.
        2. Compute SARSA-style TD error: delta = q_hat(s,a) - r - gamma * q_hat(s',a').
        3. Compute normalized critic loss: L = mean(delta^2 / (||d||^2 + 1)^2).
        4. Update critic weights.
        5. Compute actor loss: L = -sum(log pi(a|s) * A(s,a)) - beta * H(pi).
        6. Update actor weights.
        7. Clear on-policy buffer.

        Args:
            step: current environment step counter within episode.

        Returns:
            dict with "critic_loss" and "actor_loss" if update occurred, else None.
        """
        raise NotImplementedError

    def _update_critic(self) -> float:
        """
        Sample a batch from the replay buffer and perform one gradient step
        on the critic using the normalized SARSA TD loss.

        Returns:
            critic_loss: scalar loss value.
        """
        raise NotImplementedError

    def _update_actor(self) -> float:
        """
        Compute the policy gradient loss over the accumulated on-policy
        transitions, using advantage estimated from the critic, plus
        an entropy bonus.

        Advantage: A(s,a) = q_hat(s,a) - V^pi(s),
        where V^pi(s) = sum_a pi(a|s) * q_hat(s,a).

        Returns:
            actor_loss: scalar loss value.
        """
        raise NotImplementedError

    def save(self, filepath: str) -> None:
        """
        Save actor and critic network weights to a file.

        Args:
            filepath: path to the checkpoint file (e.g. "artifacts/ac_model.pt").
        """
        raise NotImplementedError

    def load(self, filepath: str) -> None:
        """
        Load actor and critic network weights from a file.

        Args:
            filepath: path to the checkpoint file.
        """
        raise NotImplementedError
