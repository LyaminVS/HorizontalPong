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
# Import other agents when implemented
# from src.agent.actor_critic import ActorCriticAgent
# from src.agent.reinforce_baseline import ReinforceBaselineAgent

from run.config import TrainConfig, ReinforceConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RL Agents on Pong.")
    parser.add_argument("--agent", type=str, required=True, 
                        choices=["actor_critic", "reinforce", "reinforce_baseline"])
    parser.add_argument("--steps", type=int, default=TrainConfig.total_steps)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--device", type=str, default=TrainConfig.device, 
                        choices=["cpu", "cuda"])
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
    
    while global_step < total_steps:
        state = env.reset()
        done = False
        ep_reward = 0.0
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            agent.store_reward(reward)
            state = next_state
            ep_reward += reward
            global_step += 1
            
            done = terminated or truncated
            
            if global_step >= total_steps:
                break
                
        # Episode is finished (or max steps reached), compute MC update
        update_info = agent.update()
        episodes += 1
        
        # Logging
        history["episode"].append(episodes)
        history["reward"].append(ep_reward)
        history["hits"].append(info["hits"])
        history["length"].append(info["step_count"])
        history["actor_loss"].append(update_info.get("actor_loss", 0.0))
        
        if episodes % config.log_interval == 0:
            avg_rew = np.mean(history["reward"][-config.log_interval:])
            avg_hits = np.mean(history["hits"][-config.log_interval:])
            print(f"Step: {global_step}/{total_steps} | Episode: {episodes} | "
                  f"Avg Reward (last {config.log_interval}): {avg_rew:.3f} | "
                  f"Avg Hits: {avg_hits:.2f} | Loss: {update_info.get('actor_loss', 0.0):.4f}")
            
        if global_step % config.save_interval == 0 or global_step >= total_steps:
            os.makedirs(config.artifacts_dir, exist_ok=True)
            save_path = os.path.join(config.artifacts_dir, f"{args.agent}_model.pt")
            agent.save(save_path)
            
    return history


def save_training_log(history: Dict[str, List], filepath: str) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df = pd.DataFrame(history)
    df.to_csv(filepath, index=False)
    print(f"Training log saved to {filepath}")


def main() -> None:
    global args
    args = parse_args()
    
    set_all_seeds(args.seed)
    
    env = PongEnv()
    env.seed(args.seed)
    
    train_config = TrainConfig(total_steps=args.steps, seed=args.seed, device=args.device)
    agent = create_agent(args.agent, args.device)
    
    if args.agent == "reinforce":
        history = train_reinforce(env, agent, args.steps, train_config)
    else:
        raise NotImplementedError(f"Training loop for {args.agent} is not yet implemented.")
        
    # Final save of the log
    log_path = os.path.join(train_config.artifacts_dir, f"train_log_{args.agent}.csv")
    save_training_log(history, log_path)
    print("Training finished successfully!")


if __name__ == "__main__":
    main()