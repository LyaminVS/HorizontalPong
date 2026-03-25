"""
Evaluation script for trained RL agents on Horizontal Pong.

Supported agents: actor_critic, reinforce, reinforce_baseline.

Loads a saved model checkpoint, runs evaluation episodes, computes
performance metrics, and optionally records a GIF rollout.

Usage:
    python -m run.eval --agent reinforce --checkpoint artifacts/reinforce_model.pt
    python -m run.eval --agent reinforce --checkpoint artifacts/reinforce_model.pt --render
    python -m run.eval --agent reinforce --checkpoint artifacts/reinforce_model.pt --save-gif
"""

import argparse
import numpy as np
import torch
from typing import Dict, List

from src.environment.pong_env import PongEnv
from src.environment.renderer import PongRenderer
from run.config import EvalConfig


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate trained RL agents on Pong.")
    parser.add_argument(
        "--agent", 
        type=str, 
        required=True, 
        choices=["actor_critic", "reinforce", "reinforce_baseline", "trpo"],
        help="Agent type."
    )
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        required=True,
        help="Path to the saved model checkpoint (.pt)."
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=None,
        help="Override hidden layer width for the selected agent (must match checkpoint).",
    )
    parser.add_argument(
        "--episodes", 
        type=int, 
        default=EvalConfig.num_episodes,
        help="Number of evaluation episodes."
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=EvalConfig.seed,
        help="Random seed."
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional per-episode step limit (overrides env time limit for eval/recording).",
    )
    parser.add_argument(
        "--render", 
        action="store_true",
        help="Flag to enable live pygame rendering."
    )
    parser.add_argument(
        "--save-gif", 
        action="store_true",
        help="Flag to save a GIF of one rollout."
    )
    return parser.parse_args()


def load_agent(
    agent_type: str,
    checkpoint_path: str,
    device: str = "cpu",
    hidden_dim_override: int | None = None,
):
    """Instantiate the agent and load weights from a checkpoint file."""
    if agent_type == "reinforce":
        from src.agent.reinforce import ReinforceAgent
        agent = ReinforceAgent(hidden_dim=hidden_dim_override or 256, device=device)
    elif agent_type == "actor_critic":
        from src.agent.actor_critic import ActorCriticAgent
        agent = ActorCriticAgent(hidden_dim=hidden_dim_override or 256, device=device)
    elif agent_type == "reinforce_baseline":
        from src.agent.reinforce_baseline import ReinforceBaselineAgent
        agent = ReinforceBaselineAgent(hidden_dim=hidden_dim_override or 256, device=device)
    elif agent_type == "trpo":
        from src.agent.trpo import TRPOAgent
        agent = TRPOAgent(hidden_dim=hidden_dim_override or 256, device=device)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
        
    print(f"Loading checkpoint from: {checkpoint_path}")
    agent.load(checkpoint_path)
    return agent


def get_deterministic_action(agent, state: np.ndarray) -> int:
    """Helper function to get the argmax action from the policy network."""
    with torch.no_grad():
        state_ts = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
        if hasattr(agent, "network"):
            logits = agent.network.get_action(state_ts)
            action = torch.argmax(logits, dim=-1).item()
        elif hasattr(agent, "actor"):
            logits = agent.actor(state_ts)
            action = torch.argmax(logits, dim=-1).item()
        elif hasattr(agent, "policy"):
            logits = agent.policy(state_ts)
            action = torch.argmax(logits, dim=-1).item()
        else:
            raise AttributeError("Agent has no recognized policy network attribute.")
    return action


def evaluate(
    env: PongEnv, agent, num_episodes: int, max_hits: int = EvalConfig.max_hits,
    deterministic: bool = True,
    max_steps: int | None = None,
) -> Dict[str, float]:
    """Run evaluation episodes and compute aggregate metrics."""
    rewards = []
    hits = []
    lengths = []

    for ep in range(num_episodes):
        state = env.reset()
        done = False
        ep_reward = 0.0
        ep_steps = 0

        while not done:
            if deterministic:
                action = get_deterministic_action(agent, state)
            else:
                action = agent.select_action(state)

            state, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_steps += 1
            if max_steps is not None and ep_steps >= max_steps:
                done = True
            else:
                done = terminated or info.get("hits", 0) >= max_hits

        rewards.append(ep_reward)
        hits.append(info["hits"])
        lengths.append(info["step_count"])

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_hits": float(np.mean(hits)),
        "mean_length": float(np.mean(lengths)),
        "max_hits": float(np.max(hits)),
    }


def record_rollout(
    env: PongEnv,
    agent,
    renderer: PongRenderer,
    filepath: str = "artifacts/rollout.gif",
    max_steps: int | None = None,
) -> None:
    """Record a single evaluation episode and save as a GIF."""
    state = env.reset()
    done = False
    frames = []
    ep_steps = 0

    while not done:
        action = get_deterministic_action(agent, state)
        state, reward, terminated, truncated, info = env.step(action)
        ep_steps += 1
        if max_steps is not None and ep_steps >= max_steps:
            done = True
        else:
            done = terminated or info.get("hits", 0) >= EvalConfig.max_hits

        frame = renderer.capture_frame(
            bx=env.bx,
            by=env.by,
            py_agent=env.py,
            py_opponent=env.ly,
            score_agent=info["hits"],
            score_opponent=0,
        )
        frames.append(frame)

    renderer.save_gif(frames, filepath)
    print(f"Rollout saved successfully to {filepath}")


def print_metrics(metrics: Dict[str, float], agent_type: str) -> None:
    """Print evaluation metrics to stdout in a formatted table."""
    print("=" * 40)
    print(f" Evaluation Metrics: {agent_type.upper()}")
    print("=" * 40)
    print(f" Mean Reward:  {metrics['mean_reward']:.3f} ± {metrics['std_reward']:.3f}")
    print(f" Mean Hits:    {metrics['mean_hits']:.2f}")
    print(f" Max Hits:     {metrics['max_hits']:.0f}")
    print(f" Mean Length:  {metrics['mean_length']:.1f} steps")
    print("=" * 40)


def main() -> None:
    """Entry point for the evaluation script."""
    args = parse_args()
    
    # Environment setup
    env = PongEnv()
    env.seed(args.seed)
    env.set_random_bounce(True)
    
    # During evaluation we want the hardest opponent level (sigma=2)
    # So we simulate being late in the training process
    env.set_global_step(200_000) 
    
    # Load model
    agent = load_agent(args.agent, args.checkpoint, hidden_dim_override=args.hidden_dim)

    # Mode 1: Live Rendering
    if args.render:
        print("Starting live rendering... Close the game window to stop.")
        renderer = PongRenderer()
        app_running = True
        
        while app_running:
            state = env.reset()
            done = False
            
            while not done and app_running:
                app_running = renderer.handle_events()
                action = get_deterministic_action(agent, state)
                state, reward, terminated, truncated, info = env.step(action)
                done = terminated or info.get("hits", 0) >= EvalConfig.max_hits
                
                renderer.render_frame(
                    bx=env.bx,
                    by=env.by,
                    py_agent=env.py,
                    py_opponent=env.ly,
                    score_agent=info["hits"],
                    score_opponent=0,
                )
        renderer.close()

    # Mode 2: Silent Evaluation Metrics
    else:
        print(f"Running evaluation over {args.episodes} episodes...")
        metrics = evaluate(env, agent, args.episodes, deterministic=True, max_steps=args.max_steps)
        print_metrics(metrics, args.agent)

    # Mode 3: Save a GIF
    if args.save_gif:
        print("Recording a GIF rollout...")
        renderer = PongRenderer()
        gif_path = f"artifacts/{args.agent}_rollout.gif"
        record_rollout(env, agent, renderer, filepath=gif_path, max_steps=args.max_steps)
        renderer.close()


if __name__ == "__main__":
    main()