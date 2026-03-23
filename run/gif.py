"""
GIF recording script for trained Actor-Critic agent.

Records one or more episodes as an animated GIF.
Each episode is saved as a separate file, or all episodes
can be stitched into a single GIF.

Usage:
    python -m run.gif --checkpoint artifacts/actor_critic_model.pt
    python -m run.gif --checkpoint artifacts/checkpoints/actor_critic_step_500000.pt --episodes 3
    python -m run.gif --checkpoint artifacts/actor_critic_model.pt --out artifacts/my_run.gif --fps 30
"""

import argparse
import os
import numpy as np
import torch
from PIL import Image

from src.environment.pong_env import PongEnv
from src.environment.renderer import PongRenderer
from src.agent.actor_critic import ActorCriticAgent
from run.config import RenderConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record Actor-Critic gameplay as GIF.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to .pt checkpoint file.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes to record (default: 1). "
             "All episodes are stitched into one GIF.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="artifacts/actor_critic_rollout.gif",
        help="Output GIF file path (default: artifacts/actor_critic_rollout.gif).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=RenderConfig.gif_fps,
        help=f"GIF frames per second (default: {RenderConfig.gif_fps}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=313,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=RenderConfig.gif_frame_skip,
        help=f"Record every Nth frame (default: {RenderConfig.gif_frame_skip}). "
             "Higher value = faster GIF + smaller file.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy (sample from distribution). "
             "Default: deterministic (argmax).",
    )
    return parser.parse_args()


def select_action(agent: ActorCriticAgent, state: np.ndarray, stochastic: bool) -> int:
    state_t = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
    with torch.no_grad():
        logits = agent.network.get_action(state_t)
    if stochastic:
        action = torch.distributions.Categorical(logits=logits).sample().item()
    else:
        action = torch.argmax(logits, dim=-1).item()
    return action


def record_episodes(
    env: PongEnv,
    agent: ActorCriticAgent,
    renderer: PongRenderer,
    num_episodes: int,
    stochastic: bool,
    frame_skip: int = 1,
) -> list:
    """Run episodes and collect frames into a single list.

    Args:
        frame_skip: record every Nth step. frame_skip=2 means half the frames
                    at the same playback fps → 2x faster perceived speed.
    """
    all_frames = []
    frame_skip = max(1, frame_skip)

    for ep in range(1, num_episodes + 1):
        state = env.reset()
        done = False
        ep_frames = []
        ep_hits = 0
        step = 0

        while not done:
            action = select_action(agent, state, stochastic)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_hits = info["hits"]

            if step % frame_skip == 0:
                frame = renderer.capture_frame(
                    bx=env.bx,
                    by=env.by,
                    py_agent=env.py,
                    py_opponent=env.ly,
                    score_agent=info["hits"],
                    score_opponent=0,
                )
                ep_frames.append(frame)
            step += 1

        all_frames.extend(ep_frames)
        print(f"  Episode {ep:>2}: {len(ep_frames):>4} frames recorded "
              f"({step} steps, skip={frame_skip}) | hits: {ep_hits}")

    return all_frames


def save_gif(frames: list, filepath: str, fps: int) -> None:
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    duration_ms = int(1000 / max(1, fps))
    pil_frames = [Image.fromarray(f) for f in frames]
    pil_frames[0].save(
        filepath,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
    )


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    print(f"Loading checkpoint: {args.checkpoint}")
    agent = ActorCriticAgent()
    agent.load(args.checkpoint)
    agent.network.eval()

    env = PongEnv()
    env.seed(args.seed)

    renderer = PongRenderer()

    mode = "stochastic" if args.stochastic else "deterministic"
    print(f"Recording {args.episodes} episode(s) | policy: {mode} | fps: {args.fps}")

    frames = record_episodes(
        env, agent, renderer, args.episodes, args.stochastic, args.frame_skip
    )

    renderer.close()

    print(f"Saving GIF ({len(frames)} frames) → {args.out}")
    save_gif(frames, args.out, args.fps)
    print("Done.")


if __name__ == "__main__":
    main()
