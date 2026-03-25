"""
PvP Tournament: TRPO vs Actor-Critic
"""

import argparse
import os
import numpy as np
import torch
import pygame
from typing import Dict, Tuple

# ============================================================================
# ИСПРАВЛЕНИЕ ДЛЯ PYTORCH 2.6+ (Отключаем параноидальную проверку weights_only)
# ============================================================================
import functools
original_torch_load = torch.load
torch.load = functools.partial(original_torch_load, weights_only=False)
# ============================================================================

from src.environment.pong_env import PongEnv
from src.environment.renderer import PongRenderer

# ============================================================================
# 1. Среда для двух агентов (PvP)
# ============================================================================

class PongVersusEnv(PongEnv):
    def _get_left_observation(self) -> np.ndarray:
        obs = np.array([
            1.0 - (self.bx / float(self.W)),               
            self.by / float(self.H),                       
            -(self.vx / float(max(1, self.MAX_BALL_SPEED_X))), 
            self.vy / float(max(1, self.MAX_BALL_SPEED_Y)),    
            self.ly / float(self.H),                       
        ], dtype=np.float32)
        return obs

    def reset_vs(self) -> Tuple[np.ndarray, np.ndarray]:
        self.reset()
        self.total_hits = 0 
        return self._get_observation(), self._get_left_observation()

    def step_vs(self, action_right: int, action_left: int) -> Tuple[np.ndarray, np.ndarray, bool, bool, Dict]:
        self._step_count += 1
        self._prev_bx = self.bx
        self._prev_by = self.by

        # Двигаем правую ракетку
        self._move_paddle(action_right)

        # Двигаем левую ракетку
        if action_left == 0:
            self.ly -= self.PADDLE_SPEED
        elif action_left == 1:
            self.ly += self.PADDLE_SPEED
        self.ly = int(np.clip(self.ly, self.PH // 2, self.H - 1 - self.PH // 2))

        self._move_ball()

        hit_right = self._check_agent_paddle_hit()

        hit_left = False
        crossed_left = self._prev_bx > self.LX >= self.bx and self.vx < 0
        if crossed_left:
            dx = self.bx - self._prev_bx
            t = 0.0 if dx == 0 else (self.LX - self._prev_bx) / float(dx)
            y_cross = self._prev_by + t * (self.by - self._prev_by)

            if abs(y_cross - self.ly) <= self.PH / 2.0:
                self.vx = abs(self.vx)
                self.bx = self.LX + 1
                self.by = int(np.clip(round(y_cross), 0, self.H - 1))
                
                half_ph = max(1, self.PH // 2)
                offset = self.by - self.ly
                normalized_offset = offset / float(half_ph)
                angle_boost = int(round(2.0 * normalized_offset))
                self.vy = int(np.clip(self.vy + angle_boost, -self.MAX_BALL_SPEED_Y, self.MAX_BALL_SPEED_Y))
                self._apply_random_bounce_noise() # Рандомный отскок ВКЛЮЧЕН
                
                hit_left = True

        if hit_right or hit_left:
            self.total_hits += 1

        terminated, truncated = self._check_terminal()

        winner = None
        if terminated:
            if self.bx < 0:
                # Ball exited on the left side -> right player scores.
                winner = "right"
            elif self.bx >= self.W:
                # Ball exited on the right side -> left player scores.
                winner = "left"

        info = {
            "hits": self.total_hits,
            "winner": winner,
            "step_count": self._step_count
        }

        return self._get_observation(), self._get_left_observation(), terminated, truncated, info


# ============================================================================
# 2. Утилиты
# ============================================================================

def load_agent(agent_type: str, checkpoint_path: str, device: str = "cpu"):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if agent_type == "trpo":
        from src.agent.trpo import TRPOAgent
        agent = TRPOAgent(device=device)
    elif agent_type == "actor_critic":
        from src.agent.actor_critic import ActorCriticAgent
        agent = ActorCriticAgent(device=device)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
        
    print(f"Loaded {agent_type.upper()} from {checkpoint_path}")
    agent.load(checkpoint_path)
    return agent


def get_deterministic_action(agent, state: np.ndarray) -> int:
    with torch.no_grad():
        state_ts = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
        if hasattr(agent, "network"):
            logits = agent.network.get_action(state_ts)
        elif hasattr(agent, "policy"):
            logits = agent.policy(state_ts)
        else:
            raise AttributeError("Agent policy not found.")
        action = torch.argmax(logits, dim=-1).item()
    return action


# ============================================================================
# 3. Главный цикл
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="TRPO vs Actor-Critic PvP Tournament")
    parser.add_argument("--episodes", type=int, default=100, help="Number of matches.")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts", help="Folder with weights.")
    parser.add_argument("--trpo-ckpt", type=str, default="trpo_model.pt", help="TRPO filename.")
    parser.add_argument("--ac-ckpt", type=str, default="actor_critic_model.pt", help="AC filename.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--render", action="store_true", help="Watch the games live (will be slower).")
    return parser.parse_args()


def main():
    args = parse_args()

    # Отключаем визуальное окно Pygame, если не просили рендер
    if not args.render:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    trpo_path = os.path.join(args.artifacts_dir, args.trpo_ckpt)
    ac_path = os.path.join(args.artifacts_dir, args.ac_ckpt)

    print("Initializing agents...")
    agent_trpo = load_agent("trpo", trpo_path, device=args.device)
    agent_ac = load_agent("actor_critic", ac_path, device=args.device)

    env = PongVersusEnv()
    env.seed(42)
    env.set_random_bounce(True)
    renderer = PongRenderer()

    trpo_wins = 0
    ac_wins = 0
    draws = 0
    hits_list = []
    
    best_hits = -1
    best_frames = []

    print(f"\nStarting TRPO vs Actor-Critic Tournament! ({args.episodes} episodes)")
    if not args.render:
        print("Running in fast background mode (No window)...")
    print("=" * 60)

    app_running = True

    for ep in range(args.episodes):
        if not app_running:
            break

        trpo_is_right = (ep < args.episodes // 2)
        agent_right = agent_trpo if trpo_is_right else agent_ac
        agent_left  = agent_ac if trpo_is_right else agent_trpo

        obs_right, obs_left = env.reset_vs()
        done = False
        ep_frames = []

        while not done and app_running:
            action_right = get_deterministic_action(agent_right, obs_right)
            action_left = get_deterministic_action(agent_left, obs_left)
            
            obs_right, obs_left, terminated, truncated, info = env.step_vs(action_right, action_left)
            done = terminated or truncated

            # --- СЧЕТЧИК ОТБИВАНИЙ (HITS) ---
            score_left = env.total_hits // 2
            score_right = (env.total_hits + 1) // 2

            if args.render:
                app_running = renderer.handle_events()
                renderer.render_frame(
                    bx=env.bx, by=env.by, 
                    py_agent=env.py, py_opponent=env.ly, 
                    score_agent=score_right, score_opponent=score_left
                )

            # Сохраняем все кадры для GIF
            frame = renderer.capture_frame(
                bx=env.bx, by=env.by, 
                py_agent=env.py, py_opponent=env.ly, 
                score_agent=score_right, score_opponent=score_left
            )
            ep_frames.append(frame)

        if info["winner"] == "right":
            if trpo_is_right: trpo_wins += 1
            else: ac_wins += 1
        elif info["winner"] == "left":
            if trpo_is_right: ac_wins += 1
            else: trpo_wins += 1
        else:
            draws += 1
            
        hits = info["hits"]
        hits_list.append(hits)
        
        # Обновляем лучший ралли
        if hits > best_hits:
            best_hits = hits
            best_frames = ep_frames.copy()

        if (ep + 1) % 10 == 0 and not args.render:
            print(f"Played {ep + 1}/{args.episodes} | TRPO wins: {trpo_wins} | AC wins: {ac_wins}")

    # ============================================================
    # СОХРАНЕНИЕ БЫСТРОЙ GIF
    # ============================================================
    gif_path = os.path.join(args.artifacts_dir, "trpo_vs_ac_best_rally_timelapse.gif")
    if best_frames:
        total_frames = len(best_frames)
        
        # Ускоряем GIF, пропуская кадры (сжимаем до ~60 кадров = 2 секунды)
        target_frames = 60
        skip_step = max(1, total_frames // target_frames)
        timelapse_frames = best_frames[::skip_step]
        
        print(f"\nSaving the BEST rally GIF ({best_hits} hits) to {gif_path} ...")
        renderer.save_gif(timelapse_frames, gif_path, fps=30)

    renderer.close()

    if len(hits_list) > 0:
        avg_hits = np.mean(hits_list)
        max_hits = np.max(hits_list)
        win_rate_trpo = (trpo_wins / len(hits_list)) * 100
        win_rate_ac = (ac_wins / len(hits_list)) * 100
        
        print("\n" + "=" * 40)
        print(" PVP TOURNAMENT RESULTS ")
        print("=" * 40)
        print(f" Total Matches : {len(hits_list)}")
        print(f" TRPO Wins     : {trpo_wins} ({win_rate_trpo:.1f}%)")
        print(f" AC Wins       : {ac_wins} ({win_rate_ac:.1f}%)")
        print(f" Draws         : {draws}")
        print("-" * 40)
        print(f" Avg Hits/Rally: {avg_hits:.2f}")
        print(f" Max Hits/Rally: {max_hits} (This is the GIF!)")
        print("=" * 40)

if __name__ == "__main__":
    main()