"""
Actor-Critic agent with Q(s,a)-critic (MLP), per specification v5.

Critic:
    Off-policy via replay buffer. SARSA-style TD error (a+ is the realised
    next action, NOT argmax). Normalised TD loss from the lecture:
        L_critic = mean( delta^2 / (||d||^2 + 1)^2 )
    where d = phi(s,a) - phi(s+,a+) and phi = [s_norm ; one_hot(a)].

Actor:
    On-policy using advantage A(s,a) = q_hat(s,a) - V^pi(s) with entropy bonus.

Both networks updated every N=10 environment steps.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Optional
from torch.distributions import Categorical

from src.agent.networks import ActorNetwork, CriticNetwork
from src.agent.replay_buffer import ReplayBuffer
from run.config import ActorCriticConfig


class ActorCriticAgent:
    """Actor-Critic with SARSA Q-critic and on-policy actor (spec v5)."""

    def __init__(
        self,
        state_dim: int = ActorCriticConfig.state_dim,
        action_dim: int = ActorCriticConfig.action_dim,
        hidden_dim: int = ActorCriticConfig.hidden_dim,
        gamma: float = ActorCriticConfig.gamma,
        lr_actor: float = ActorCriticConfig.lr_actor,
        lr_critic: float = ActorCriticConfig.lr_critic,
        entropy_coeff: float = ActorCriticConfig.entropy_coeff,
        grad_clip_norm: float = ActorCriticConfig.grad_clip_norm,
        buffer_capacity: int = ActorCriticConfig.buffer_capacity,
        batch_size: int = ActorCriticConfig.batch_size,
        update_every: int = ActorCriticConfig.update_every,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.gamma = float(gamma)
        self.entropy_coeff = float(entropy_coeff)
        self.grad_clip_norm = float(grad_clip_norm)
        self.batch_size = int(batch_size)
        self.update_every = int(update_every)
        self.action_dim = int(action_dim)

        self.actor = ActorNetwork(state_dim, hidden_dim, action_dim).to(self.device)
        self.critic = CriticNetwork(state_dim, action_dim, hidden_dim).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity, state_dim=state_dim)
        self._on_policy: list[tuple[np.ndarray, int]] = []

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray) -> int:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits = self.actor(state_t)
        dist = Categorical(logits=logits)
        return int(dist.sample().item())

    # ------------------------------------------------------------------
    # Data storage
    # ------------------------------------------------------------------

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        next_action: int,
        done: bool = False,
    ) -> None:
        """Store (s, a, r, s+, a+) in replay buffer and (s, a) on-policy."""
        self.replay_buffer.push(state, action, reward, next_state, next_action, done=done)
        self._on_policy.append((np.asarray(state, dtype=np.float32), int(action)))

    # ------------------------------------------------------------------
    # Joint update (every N steps)
    # ------------------------------------------------------------------

    def update(self, step: int) -> Optional[Dict[str, float]]:
        """
        Every N steps: update critic from replay buffer, then update actor
        from accumulated on-policy transitions. Clear on-policy buffer.
        """
        if step % self.update_every != 0:
            return None
        if len(self.replay_buffer) < max(2, self.batch_size):
            return None
        if len(self._on_policy) == 0:
            return None

        critic_loss = self._update_critic()
        actor_loss = self._update_actor()
        self._on_policy.clear()
        return {"critic_loss": float(critic_loss), "actor_loss": float(actor_loss)}

    # ------------------------------------------------------------------
    # Critic: SARSA TD with normalised loss (spec section 7.1)
    # ------------------------------------------------------------------

    def _update_critic(self) -> float:
        """
        SARSA-style TD update with normalised loss:
            delta = q_hat(s,a) - r - gamma * q_hat(s+, a+)
            d = phi(s,a) - phi(s+, a+)   (input feature vectors)
            L = mean( delta^2 / (||d||^2 + 1)^2 )
        """
        batch = self.replay_buffer.sample(self.batch_size)
        states = torch.as_tensor(batch["states"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(batch["next_states"], dtype=torch.float32, device=self.device)
        next_actions = torch.as_tensor(batch["next_actions"], dtype=torch.long, device=self.device)
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device)

        q_sa = self.critic(states, actions).squeeze(-1)
        with torch.no_grad():
            q_next = self.critic(next_states, next_actions).squeeze(-1)
            target = rewards + self.gamma * q_next * (1.0 - dones)

        td_error = q_sa - target

        phi = torch.cat([states, F.one_hot(actions, self.action_dim).float()], dim=-1)
        phi_next = torch.cat([next_states, F.one_hot(next_actions, self.action_dim).float()], dim=-1)
        d = phi - phi_next
        d_norm_sq = (d * d).sum(dim=-1)
        normalizer = (d_norm_sq + 1.0).pow(2)

        loss = (td_error.pow(2) / normalizer).mean()

        self.critic_optimizer.zero_grad()
        loss.backward()
        if self.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_norm)
        self.critic_optimizer.step()
        return float(loss.item())

    # ------------------------------------------------------------------
    # Actor: on-policy with advantage from critic (spec section 7.2)
    # ------------------------------------------------------------------

    def _update_actor(self) -> float:
        """
        Policy gradient with advantage A(s,a) = q_hat(s,a) - V^pi(s),
        where V^pi(s) = sum_a pi(a|s) * q_hat(s,a), plus entropy bonus.
            L_actor = - sum log pi(a|s) * A(s,a) - beta * H(pi)
        """
        states_np = np.stack([s for s, _ in self._on_policy], axis=0)
        actions_np = np.array([a for _, a in self._on_policy], dtype=np.int64)
        states = torch.as_tensor(states_np, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions_np, dtype=torch.long, device=self.device)

        logits = self.actor(states)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        with torch.no_grad():
            q_sa = self.critic(states, actions).squeeze(-1)
            q_all = []
            for a in range(self.action_dim):
                a_batch = torch.full((states.size(0),), a, device=self.device, dtype=torch.long)
                q_all.append(self.critic(states, a_batch).squeeze(-1))
            q_all = torch.stack(q_all, dim=1)
            pi = torch.softmax(logits.detach(), dim=-1)
            v_pi = (pi * q_all).sum(dim=1)
            advantage = q_sa - v_pi
            if advantage.numel() > 1:
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        loss = -(log_probs * advantage).mean() - self.entropy_coeff * entropy
        self.actor_optimizer.zero_grad()
        loss.backward()
        if self.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_norm)
        self.actor_optimizer.step()
        return float(loss.item())

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, filepath: str) -> None:
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            },
            filepath,
        )

    def load(self, filepath: str) -> None:
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        if "actor_optimizer_state_dict" in checkpoint:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        if "critic_optimizer_state_dict" in checkpoint:
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
