"""
Manual play mode for Horizontal Pong.

Control the right paddle (agent side) with keyboard:
- Up Arrow or W: move up
- Down Arrow or S: move down
- No key: stay

Run:
    python -m run.play
"""

import argparse
import pygame

from src.environment.pong_env import PongEnv
from src.environment.renderer import PongRenderer


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for manual play."""
    parser = argparse.ArgumentParser(description="Play Horizontal Pong manually.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for environment.")
    parser.add_argument(
        "--episodes",
        type=int,
        default=0,
        help="Number of episodes to play. 0 means infinite.",
    )
    return parser.parse_args()


def get_action_from_keyboard() -> int:
    """
    Map keyboard state to environment action.

    Returns:
        0 for up, 1 for down, 2 for stay.
    """
    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP] or keys[pygame.K_w]:
        return 0
    if keys[pygame.K_DOWN] or keys[pygame.K_s]:
        return 1
    return 2


def main() -> None:
    """Entry point for manual play loop."""
    args = parse_args()

    env = PongEnv()
    env.seed(args.seed)
    renderer = PongRenderer()

    observation = env.reset()
    del observation  # manual mode does not need observation directly

    app_running = True
    episodes_played = 0
    episode_reward = 0.0
    score_left = 0
    score_right = 0

    try:
        while app_running and (args.episodes == 0 or episodes_played < args.episodes):
            app_running = renderer.handle_events()
            action = get_action_from_keyboard()

            _, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            winner = info.get("rally_winner")
            if winner == "left":
                score_left += 1
            elif winner == "right":
                score_right += 1

            renderer.render_frame(
                bx=env.bx,
                by=env.by,
                py_agent=env.py,
                py_opponent=env.ly,
                score_agent=score_right,
                score_opponent=score_left,
            )

            if terminated or truncated:
                episodes_played += 1
                print(
                    f"[Episode {episodes_played}] reward={episode_reward:.3f}, "
                    f"hits={info['hits']}, steps={info['step_count']}, "
                    f"terminated={terminated}, truncated={truncated}, "
                    f"score_left={score_left}, score_right={score_right}"
                )
                env.reset()
                episode_reward = 0.0
    finally:
        renderer.close()


if __name__ == "__main__":
    main()
