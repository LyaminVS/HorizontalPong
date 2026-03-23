"""
Evaluation script for trained RL agents on Horizontal Pong.

Supported agents: actor_critic, reinforce, reinforce_baseline.

Loads a saved model checkpoint, runs evaluation episodes, computes
performance metrics, and optionally records a GIF rollout.

Usage:
    python -m run.eval --agent actor_critic        --checkpoint artifacts/ac_model.pt
    python -m run.eval --agent reinforce           --checkpoint artifacts/reinforce_model.pt
    python -m run.eval --agent reinforce_baseline  --checkpoint artifacts/reinforce_bl_model.pt
"""

import argparse
import numpy as np
from typing import Dict, List

from src.environment.pong_env import PongEnv
from src.environment.renderer import PongRenderer
from src.agent.actor_critic import ActorCriticAgent
from src.agent.reinforce import ReinforceAgent
from src.agent.reinforce_baseline import ReinforceBaselineAgent
from run.config import EvalConfig


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the evaluation script.

    Arguments:
        --agent:      agent type, one of {"actor_critic", "reinforce",
                      "reinforce_baseline"}.
        --checkpoint: path to the saved model checkpoint.
        --episodes:   number of evaluation episodes (default 100).
        --seed:       random seed (default 123).
        --render:     flag to enable live rendering.
        --save-gif:   flag to save a GIF of one rollout.

    Returns:
        Parsed argparse.Namespace.
    """
    raise NotImplementedError


def load_agent(agent_type: str, checkpoint_path: str, device: str = "cpu"):
    """
    Instantiate the agent and load weights from a checkpoint file.

    Args:
        agent_type: one of {"actor_critic", "reinforce", "reinforce_baseline"}.
        checkpoint_path: path to the .pt checkpoint file.
        device: torch device string.

    Returns:
        Agent instance with loaded weights.
    """
    raise NotImplementedError


def evaluate(
    env: PongEnv, agent, num_episodes: int, deterministic: bool = True
) -> Dict[str, float]:
    """
    Run evaluation episodes and compute aggregate metrics.

    For each episode:
        1. Reset environment.
        2. Run policy until termination/truncation (greedy or stochastic).
        3. Collect episode reward, hits, and length.

    Args:
        env: PongEnv instance.
        agent: agent with a select_action method.
        num_episodes: number of episodes to evaluate.
        deterministic: if True, use argmax action instead of sampling.

    Returns:
        metrics: dict with keys:
            "mean_reward": average total reward per episode.
            "std_reward": standard deviation of rewards.
            "mean_hits": average number of ball hits per episode.
            "mean_length": average episode length.
            "max_hits": maximum hits in a single episode.
    """
    raise NotImplementedError


def record_rollout(
    env: PongEnv,
    agent,
    renderer: PongRenderer,
    filepath: str = "artifacts/rollout.gif",
) -> None:
    """
    Record a single evaluation episode and save as a GIF.

    Args:
        env: PongEnv instance.
        agent: agent with a select_action method.
        renderer: PongRenderer instance for frame generation.
        filepath: output path for the GIF file.
    """
    raise NotImplementedError


def print_metrics(metrics: Dict[str, float], agent_type: str) -> None:
    """
    Print evaluation metrics to stdout in a formatted table.

    Args:
        metrics: evaluation metrics dict.
        agent_type: string name of the agent for display.
    """
    raise NotImplementedError


def main() -> None:
    """
    Entry point: parse arguments, load agent, run evaluation,
    print metrics, and optionally save rollout GIF.
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
