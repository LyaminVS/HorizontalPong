"""
Training script for RL agents on Horizontal Pong.
"""

import argparse
import os
import random
import numpy as np
import pandas as pd
import torch
from typing import Dict, List

from src.environment.pong_env import PongEnv
from src.agent.reinforce import ReinforceAgent
from src.agent.trpo import TRPOAgent
# Import other agents when implemented
# from src.agent.actor_critic import ActorCriticAgent
# from src.agent.reinforce_baseline import ReinforceBaselineAgent

from run.config import TrainConfig, ReinforceConfig, TRPOConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RL Agents on Pong.")
    parser.add_argument("--agent", type=str, required=True, 
                        choices=["actor_critic", "reinforce", "reinforce_baseline", "trpo"])
    parser.add_argument("--steps", type=int, default=TrainConfig.total_steps)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--device", type=str, default=TrainConfig.device, 
                        choices=["cpu", "cuda"])
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default=None,
        help="Path to an existing .pt checkpoint to continue training from.",
    )
    return parser.parse_args()


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_agent(agent_type: str, device: str):
    if agent_type == "reinforce":
        config = ReinforceConfig()
        return ReinforceAgent(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim,
            gamma=config.gamma,
            lr_actor=config.lr_actor,
            device=device
        )
    if agent_type == "trpo":
        config = TRPOConfig()
        return TRPOAgent(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim,
            gamma=config.gamma,
            max_kl=config.max_kl,
            entropy_coeff=config.entropy_coeff,
            grad_clip_norm=config.grad_clip_norm,
            damping=config.damping,
            cg_iters=config.cg_iters,
            backtrack_iters=config.backtrack_iters,
            backtrack_coeff=config.backtrack_coeff,
            device=device,
        )
    # elif agent_type == "actor_critic": ...
    # elif agent_type == "reinforce_baseline": ...
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def train_reinforce(
    env: PongEnv, agent: ReinforceAgent, total_steps: int, config: TrainConfig
) -> Dict[str, List]:
    history = {
        "episode": [], "reward": [], "hits": [], 
        "length": [], "actor_loss": []
    }
    
    global_step = 0
    episodes = 0
    
    print(f"Starting REINFORCE training for {total_steps} steps...")
    
    state = env.reset()
    ep_reward = 0.0
    ep_steps = 0
    ep_hits = 0
    info = {"hits": 0, "step_count": 0, "opponent_hit": False, "agent_hit": False}

    while global_step < total_steps:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)

        agent.store_reward(reward)
        state = next_state
        ep_reward += reward
        ep_steps += 1
        if info.get("agent_hit", False):
            ep_hits += 1
        global_step += 1

        short_done = config.short_episode_on_opponent_hit and info.get("opponent_hit", False)
        env_done = terminated or truncated
        done = short_done or env_done or (global_step >= total_steps)

        if not done:
            continue

        update_info = agent.update()
        episodes += 1
        
        # Logging
        history["episode"].append(episodes)
        history["reward"].append(ep_reward)
        history["hits"].append(ep_hits)
        history["length"].append(ep_steps)
        history["actor_loss"].append(update_info.get("actor_loss", 0.0))
        
        if episodes % config.log_interval == 0:
            avg_rew = np.mean(history["reward"][-config.log_interval:])
            avg_hits = np.mean(history["hits"][-config.log_interval:])
            print(f"Step: {global_step}/{total_steps} | Episode: {episodes} | "
                  f"Avg Reward (last {config.log_interval}): {avg_rew:.3f} | "
                  f"Avg Hits: {avg_hits:.2f} | Loss: {update_info.get('actor_loss', 0.0):.4f}")
            
        if global_step % config.save_interval == 0 or global_step >= total_steps:
            save_checkpoint(agent, config.artifacts_dir, args.agent, global_step)

        if env_done:
            state = env.reset()
        ep_reward = 0.0
        ep_steps = 0
        ep_hits = 0

    return history


def train_trpo(env: PongEnv, agent: TRPOAgent, total_steps: int, config: TrainConfig) -> Dict[str, List]:
    history = {
        "episode": [], "reward": [], "hits": [],
        "length": [], "surrogate_before": [], "surrogate_after": [], "kl": [], "entropy": [], "grad_norm": []
    }

    global_step = 0
    episodes = 0
    print(f"Starting TRPO training for {total_steps} steps...")

    state = env.reset()
    ep_reward = 0.0
    ep_steps = 0
    ep_hits = 0
    info = {"hits": 0, "step_count": 0, "opponent_hit": False, "agent_hit": False}

    while global_step < total_steps:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        agent.store_reward(reward)
        state = next_state
        ep_reward += reward
        ep_steps += 1
        if info.get("agent_hit", False):
            ep_hits += 1
        global_step += 1

        short_done = config.short_episode_on_opponent_hit and info.get("opponent_hit", False)
        env_done = terminated or truncated
        done = short_done or env_done or (global_step >= total_steps)

        if not done:
            continue

        update_info = agent.update()
        episodes += 1

        history["episode"].append(episodes)
        history["reward"].append(ep_reward)
        history["hits"].append(ep_hits)
        history["length"].append(ep_steps)
        history["surrogate_before"].append(update_info.get("surrogate_before", 0.0))
        history["surrogate_after"].append(update_info.get("surrogate_after", 0.0))
        history["kl"].append(update_info.get("kl", 0.0))
        history["entropy"].append(update_info.get("entropy", 0.0))
        history["grad_norm"].append(update_info.get("grad_norm", 0.0))

        if episodes % config.log_interval == 0:
            avg_rew = np.mean(history["reward"][-config.log_interval:])
            avg_hits = np.mean(history["hits"][-config.log_interval:])
            print(
                f"Step: {global_step}/{total_steps} | Episode: {episodes} | "
                f"Avg Reward: {avg_rew:.3f} | Avg Hits: {avg_hits:.2f} | "
                f"KL: {update_info.get('kl', 0.0):.5f} | "
                f"Entropy: {update_info.get('entropy', 0.0):.4f} | "
                f"GradNorm: {update_info.get('grad_norm', 0.0):.4f}"
            )

        if global_step % config.save_interval == 0 or global_step >= total_steps:
            save_checkpoint(agent, config.artifacts_dir, args.agent, global_step)

        if env_done:
            state = env.reset()
        ep_reward = 0.0
        ep_steps = 0
        ep_hits = 0

    return history


def save_training_log(history: Dict[str, List], filepath: str) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df = pd.DataFrame(history)
    df.to_csv(filepath, index=False)
    print(f"Training log saved to {filepath}")


def save_checkpoint(agent, artifacts_dir: str, agent_name: str, global_step: int) -> None:
    """Save rolling + intermediate checkpoint snapshot."""
    os.makedirs(artifacts_dir, exist_ok=True)
    rolling_path = os.path.join(artifacts_dir, f"{agent_name}_model.pt")
    agent.save(rolling_path)

    ckpt_dir = os.path.join(artifacts_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    step_path = os.path.join(ckpt_dir, f"{agent_name}_step_{global_step}.pt")
    agent.save(step_path)


def main() -> None:
    global args
    args = parse_args()
    
    set_all_seeds(args.seed)
    
    env = PongEnv()
    env.seed(args.seed)
    
    train_config = TrainConfig(total_steps=args.steps, seed=args.seed, device=args.device)
    agent = create_agent(args.agent, args.device)

    if args.resume_checkpoint is not None:
        if not os.path.exists(args.resume_checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {args.resume_checkpoint}")
        print(f"Resuming training from checkpoint: {args.resume_checkpoint}")
        agent.load(args.resume_checkpoint)
    
    if args.agent == "reinforce":
        history = train_reinforce(env, agent, args.steps, train_config)
    elif args.agent == "trpo":
        history = train_trpo(env, agent, args.steps, train_config)
    else:
        raise NotImplementedError(f"Training loop for {args.agent} is not yet implemented.")
        
    # Final save of the log
    log_path = os.path.join(train_config.artifacts_dir, f"train_log_{args.agent}.csv")
    save_training_log(history, log_path)
    print("Training finished successfully!")


if __name__ == "__main__":
    main()