"""
Off-policy Actor-Critic agent with shared backbone network.

Critic:  Q(s,a) trained via TD error with Expected-SARSA target V(s') = Σ_a π(a|s')·Q(s',a).
Actor:   analytical policy gradient maximizing E_s[Σ_a π(a|s)·Q(s,a)].
Entropy: optional regularisation encouraging exploration.

Combined loss:  L = critic_coeff · L_critic  +  L_actor  +  entropy_coeff · L_entropy
Updated every `update_every` environment steps from replay-buffer mini-batches.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional

from src.agent.networks import ActorCriticNetwork
from src.agent.replay_buffer import ReplayBuffer
from run.config import ActorCriticConfig


class ActorCriticAgent:
    """Off-policy Actor-Critic with a single shared ActorCriticNetwork."""

    def __init__(
        self,
        state_dim: int = ActorCriticConfig.state_dim,
        action_dim: int = ActorCriticConfig.action_dim,
        hidden_dim: int = ActorCriticConfig.hidden_dim,
        gamma: float = ActorCriticConfig.gamma,
        lr: float = ActorCriticConfig.lr,
        critic_coeff: float = ActorCriticConfig.critic_coeff,
        entropy_coeff: float = ActorCriticConfig.entropy_coeff,
        use_entropy: bool = ActorCriticConfig.use_entropy,
        grad_clip_norm: float = ActorCriticConfig.grad_clip_norm,
        buffer_capacity: int = ActorCriticConfig.buffer_capacity,
        batch_size: int = ActorCriticConfig.batch_size,
        update_every: int = ActorCriticConfig.update_every,
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.gamma = gamma
        self.critic_coeff = critic_coeff
        self.entropy_coeff = entropy_coeff
        self.use_entropy = use_entropy
        self.grad_clip_norm = grad_clip_norm
        self.batch_size = batch_size
        self.update_every = update_every
        self.action_dim = action_dim

        self.network = ActorCriticNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_capacity, state_dim)

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray) -> int:
        """Sample action from π(·|s)."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.network.get_action(state_t)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.sample().item()

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Push (s, a, r, s', done) into the replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def update(self, step: int) -> Optional[Dict[str, float]]:
        """
        Perform one gradient step if conditions are met:
        1) step is a multiple of update_every,
        2) replay buffer has enough samples.
        """
        if step % self.update_every != 0:
            return None
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = self.replay_buffer.sample(self.batch_size)
        states = torch.FloatTensor(batch["states"]).to(self.device)
        actions = torch.LongTensor(batch["actions"]).to(self.device)
        rewards = torch.FloatTensor(batch["rewards"]).to(self.device)
        next_states = torch.FloatTensor(batch["next_states"]).to(self.device)
        dones = torch.FloatTensor(batch["dones"]).to(self.device)

        critic_loss = self._loss_critic(states, actions, rewards, next_states, dones)
        actor_loss = self._loss_actor(states)

        total_loss = self.critic_coeff * critic_loss + actor_loss

        entropy_loss_val = 0.0
        if self.use_entropy:
            entropy_loss = self._entropy_loss(states)
            total_loss = total_loss + self.entropy_coeff * entropy_loss
            entropy_loss_val = entropy_loss.item()

        self.optimizer.zero_grad()
        total_loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.network.parameters(), self.grad_clip_norm
        ).item()

        self.optimizer.step()

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "entropy_loss": entropy_loss_val,
            "total_loss": total_loss.item(),
            "grad_norm": grad_norm,
        }

    # ------------------------------------------------------------------
    # Loss components
    # ------------------------------------------------------------------

    def _loss_critic(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """MSE TD loss: (Q(s,a) − [r + γ·V(s')])²  where V(s') = Σ_a π(a|s')·Q(s',a)."""
        q_sa = self.network(states, actions)

        with torch.no_grad():
            next_action_logits = self.network.get_action(next_states)
            next_action_probs = F.softmax(next_action_logits, dim=-1) # π(a|s')
            q_all_next_s_a = self.network.critic(self.network.net(next_states))
            v_next = (next_action_probs * q_all_next_s_a).sum(dim=-1) # Expected-SARSA target
            targets = rewards + self.gamma * v_next * (1.0 - dones)

        return F.mse_loss(q_sa, targets)

    def _loss_actor(self, states: torch.Tensor) -> torch.Tensor:
        """Analytical policy gradient: −E_s[Σ_a π(a|s)·Q(s,a)]."""
        logits = self.network.get_action(states)
        probs = F.softmax(logits, dim=-1)

        with torch.no_grad():
            features = self.network.net(states)
            q_all = self.network.critic(features)

        return -(probs * q_all).sum(dim=-1).mean()

    def _entropy_loss(self, states: torch.Tensor) -> torch.Tensor:
        """Negative entropy of π(·|s); minimise to encourage exploration."""
        logits = self.network.get_action(states)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        return -entropy

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, filepath: str) -> None:
        torch.save(
            {
                "network": self.network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            filepath,
        )

    def load(self, filepath: str) -> None:
        ckpt = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(ckpt["network"])
        self.optimizer.load_state_dict(ckpt["optimizer"])