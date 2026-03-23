"""
Training script for Actor-Critic and REINFORCE agents on Horizontal Pong.

Usage:
    python -m src.run.train --agent actor_critic --steps 500000 --seed 42
    python -m src.run.train --agent reinforce    --steps 500000 --seed 42

Saves model checkpoints and training logs (CSV) to artifacts/.
"""

import argparse
import numpy as np
from typing import Dict, List

from src.environment.pong_env import PongEnv
from src.agent.actor_critic import ActorCriticAgent
from src.agent.reinforce import ReinforceAgent
from src.run.config import TrainConfig, ActorCriticConfig, ReinforceConfig


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the training script.

    Arguments:
        --agent:  agent type, one of {"actor_critic", "reinforce"}.
        --steps:  total number of environment steps (default 500_000).
        --seed:   random seed for reproducibility (default 42).
        --device: torch device, "cpu" or "cuda" (default "cpu").

    Returns:
        Parsed argparse.Namespace.
    """
    raise NotImplementedError


def set_all_seeds(seed: int) -> None:
    """
    Set random seeds for numpy, torch, and the environment for full reproducibility.

    Args:
        seed: integer seed value.
    """
    raise NotImplementedError


def create_agent(agent_type: str, config: Dict, device: str):
    """
    Factory function: instantiate the appropriate agent based on type string.

    Args:
        agent_type: "actor_critic" or "reinforce".
        config: dict of hyperparameters from the corresponding config dataclass.
        device: torch device string.

    Returns:
        An instance of ActorCriticAgent or ReinforceAgent.
    """
    raise NotImplementedError


def train_actor_critic(
    env: PongEnv, agent: ActorCriticAgent, total_steps: int, config: TrainConfig
) -> Dict[str, List]:
    """
    Training loop for the Actor-Critic agent.

    For each step:
        1. Select action a ~ pi(.|s).
        2. Execute env.step(a), receive (s', r, terminated, truncated, info).
        3. Select next action a' ~ pi(.|s') for SARSA target.
        4. Store transition (s, a, r, s', a') in replay buffer.
        5. Every N steps: update critic and actor.
        6. On episode end: log metrics, reset environment.

    Args:
        env: PongEnv instance.
        agent: ActorCriticAgent instance.
        total_steps: total environment steps to train for.
        config: TrainConfig with logging/saving intervals.

    Returns:
        history: dict with keys:
            "episode_rewards": list of total rewards per episode.
            "episode_hits": list of ball hits per episode.
            "episode_lengths": list of episode step counts.
            "critic_losses": list of critic loss values.
            "actor_losses": list of actor loss values.
    """
    raise NotImplementedError


def train_reinforce(
    env: PongEnv, agent: ReinforceAgent, total_steps: int, config: TrainConfig
) -> Dict[str, List]:
    """
    Training loop for the REINFORCE agent.

    For each episode:
        1. Reset environment.
        2. Collect full trajectory: for each step, select action and store reward.
        3. After episode ends: compute returns G_t, update actor.
        4. Log metrics.
        5. Repeat until total_steps exhausted.

    Args:
        env: PongEnv instance.
        agent: ReinforceAgent instance.
        total_steps: total environment steps to train for.
        config: TrainConfig with logging/saving intervals.

    Returns:
        history: dict with same keys as train_actor_critic.
    """
    raise NotImplementedError


def save_training_log(history: Dict[str, List], filepath: str) -> None:
    """
    Save training metrics to a CSV file for later analysis.

    Columns: episode, reward, hits, length, critic_loss, actor_loss.

    Args:
        history: training history dict.
        filepath: output CSV path (e.g. "artifacts/train_log_ac.csv").
    """
    raise NotImplementedError


def main() -> None:
    """
    Entry point: parse arguments, create environment and agent,
    run the appropriate training loop, save model and logs.
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
