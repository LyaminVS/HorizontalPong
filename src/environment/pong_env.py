"""
Discrete Horizontal Pong environment.

Field: W x H = 160 x 120 pixels.
Paddles: PW x PH = 2 x 12 pixels, positioned vertically (left at x=6, right at x=153).
Ball speed: |vx| in {1, 2}, vy in {-3, ..., +3}.
State: (bx, by, vx, vy, py) — integer vector.
Actions: 0=Up, 1=Down, 2=Stay.
No gymnasium dependency.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np

from run.config import EnvConfig, RewardConfig


class PongEnv:
    """
    Custom RL environment for discrete horizontal Pong.
    Implements reset/step/seed API without gymnasium.
    Uses a dedicated opponent module (`LeftPaddleOpponent`) for
    the left paddle control and bounce-noise curriculum.

    Attributes:
        W, H: field dimensions (160 x 120)
        PW, PH: paddle dimensions (2 x 12)
        LX, RX: x-coordinates of left (6) and right (153) paddles
        T_max: maximum episode length (2000 steps)
        observation_space: dict describing state bounds
        action_space: dict describing valid actions {0, 1, 2}
    """

    # --- Constants ---
    W: int = EnvConfig.width
    H: int = EnvConfig.height
    PW: int = EnvConfig.paddle_width
    PH: int = EnvConfig.paddle_height
    LX: int = EnvConfig.left_x
    RX: int = EnvConfig.right_x
    T_MAX: int = EnvConfig.t_max

    def __init__(self) -> None:
        """
        Initialize the environment: set field parameters, define observation_space
        and action_space dicts, initialize internal state to None.
        """
        self.observation_space: Dict[str, Any] = {
            "shape": (5,),
            "low": np.array([0.0, 0.0, -2.0 / 3.0, -1.0, 0.0], dtype=np.float32),
            "high": np.array([1.0, 1.0, 2.0 / 3.0, 1.0, 1.0], dtype=np.float32),
            "dtype": np.float32,
        }
        self.action_space: Dict[str, Any] = {
            "n": 3,
            "actions": [0, 1, 2],
            "meaning": {0: "up", 1: "down", 2: "stay"},
        }

        self._rng = np.random.default_rng()
        self._global_step = 0
        self._step_count = 0
        self._hits = 0

        # Right paddle (agent) center.
        self.py = self.H // 2
        # Left paddle placeholder (static opponent).
        self.ly = self.H // 2

        # Ball state.
        self.bx = self.W // 2
        self.by = self.H // 2
        self.vx = 1
        self.vy = 0

    def seed(self, seed: Optional[int] = None) -> None:
        """
        Set the random seed for reproducibility.

        Args:
            seed: integer seed for numpy RNG. If None, use a random seed.
        """
        self._rng = np.random.default_rng(seed)

    def reset(self) -> np.ndarray:
        """
        Reset the environment to a starting state.

        - Place the ball at the center of the field.
        - Assign a random initial velocity to the ball (vx in {-1,-2,+1,+2}, vy in {-3..+3}).
        - Place both paddles at the vertical center.
        - Reset step counter to 0.
        - Reset episode statistics (hits, score).

        Returns:
            observation: normalized state vector np.ndarray of shape (5,).
                         [bx/W, by/H, vx/3, vy/3, py/H]
        """
        self._step_count = 0
        self._hits = 0

        self.py = self.H // 2
        self.ly = self.H // 2  # static placeholder opponent

        self.bx = self.W // 2
        self.by = self.H // 2
        self.vx = int(self._rng.choice([-2, -1, 1, 2]))
        self.vy = int(self._rng.integers(-3, 4))

        return self._get_observation()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one time step of the environment.

        1. Update agent paddle position based on action (0=Up, 1=Down, 2=Stay),
           clamping within [PH/2, H - 1 - PH/2].
        2. Update ball position: bx' = bx + vx, by' = by + vy.
        3. Handle wall bounces (top/bottom).
        4. Handle right paddle (agent) hit detection and bounce.
        5. Handle left paddle (opponent) hit detection and bounce with noise.
        6. Compute reward (sparse + dense components).
        7. Check terminal conditions (ball out of bounds, T_max reached).

        Args:
            action: integer in {0, 1, 2}.

        Returns:
            observation: normalized state vector (5,).
            reward: float scalar.
            terminated: True if the ball left the field (episode end by goal).
            truncated: True if T_max steps reached.
            info: dict with auxiliary data (e.g. {"hits": int, "score": str}).
        """
        if action not in (0, 1, 2):
            raise ValueError(f"Invalid action {action}, expected one of [0, 1, 2].")

        self._step_count += 1
        self._global_step += 1

        self._move_paddle(action)
        self._move_ball()
        hit = self._check_agent_paddle_hit()
        self._check_opponent_paddle_hit()

        reward = self._compute_reward(hit=hit)
        terminated, truncated = self._check_terminal()

        info = {
            "hits": self._hits,
            "step_count": self._step_count,
            "agent_paddle_y": self.py,
            "opponent_paddle_y": self.ly,
            "opponent_stub": True,
        }
        return self._get_observation(), reward, terminated, truncated, info

    def _move_paddle(self, action: int) -> None:
        """
        Update agent's paddle vertical position based on action.

        Action mapping:
            0 -> py -= 1 (up),   clamped at PH/2
            1 -> py += 1 (down), clamped at H - 1 - PH/2
            2 -> no change

        Args:
            action: integer action.
        """
        if action == 0:
            self.py -= 1
        elif action == 1:
            self.py += 1

        low = self.PH // 2
        high = self.H - 1 - self.PH // 2
        self.py = int(np.clip(self.py, low, high))

    def _move_ball(self) -> None:
        """
        Advance the ball by its velocity: bx += vx, by += vy.
        Handle top/bottom wall bounces:
            if by <= 0:  by = 0,     vy = +|vy|
            if by >= H-1: by = H-1,  vy = -|vy|
        """
        self.bx += self.vx
        self.by += self.vy

        if self.by <= 0:
            self.by = 0
            self.vy = abs(self.vy)
        elif self.by >= self.H - 1:
            self.by = self.H - 1
            self.vy = -abs(self.vy)

    def _check_agent_paddle_hit(self) -> bool:
        """
        Check if the ball hits the agent's (right) paddle.

        Condition: vx > 0 and bx >= RX and |by - py| <= PH/2.
        On hit: vx = -|vx|, bx = RX - 1.
        Angular bounce is applied based on impact point:
            - hit near center -> minimal vertical change,
            - hit near paddle edges -> stronger vertical deflection.

        Returns:
            True if the ball was deflected by the agent paddle.
        """
        if self.vx > 0 and self.bx >= self.RX and abs(self.by - self.py) <= self.PH // 2:
            self.vx = -abs(self.vx)
            self.bx = self.RX - 1

            # Angular bounce: edge hits produce stronger vertical deflection.
            half_ph = max(1, self.PH // 2)
            offset = self.by - self.py  # negative: upper edge, positive: lower edge
            normalized_offset = offset / float(half_ph)  # in [-1, 1] approximately
            angle_boost = int(round(2.0 * normalized_offset))  # map to {-2, -1, 0, 1, 2}
            self.vy = int(np.clip(self.vy + angle_boost, -3, 3))

            self._hits += 1
            return True
        return False

    def _check_opponent_paddle_hit(self) -> None:
        """
        Check if the ball hits the opponent's (left) paddle.

        The left paddle behavior is delegated to `LeftPaddleOpponent`:
        - paddle center ly is provided by opponent.compute_paddle_center(...),
        - bounce noise is applied by opponent.apply_bounce_noise(...).

        On hit: vx = +|vx|, bx = LX + 1.
        """
        # Opponent stub: left paddle is fixed at center and has no bounce noise.
        self.ly = self.H // 2
        if self.vx < 0 and self.bx <= self.LX and abs(self.by - self.ly) <= self.PH // 2:
            self.vx = abs(self.vx)
            self.bx = self.LX + 1

    def _compute_reward(self, hit: bool) -> float:
        """
        Compute the reward for the current transition.

        Sparse component:
            +1 if the agent successfully deflected the ball (hit=True)
            -1 if the ball passed the agent (bx < 0)
             0 otherwise

        Dense component (only when vx > 0, i.e. ball approaching agent):
            r_dense = -alpha * |py - by| / H
            alpha decays linearly from 0.01 to 0 over the first 100k global steps.

        Returns:
            reward: r_sparse + r_dense.
        """
        reward_sparse = 1.0 if hit else 0.0
        # Agent miss: ball exited right side.
        if self.bx >= self.W:
            reward_sparse = -1.0

        reward_dense = 0.0
        if self.vx > 0:
            alpha = RewardConfig.alpha_initial * max(
                0.0, 1.0 - self._global_step / float(RewardConfig.alpha_decay_steps)
            )
            reward_dense = -alpha * abs(self.py - self.by) / float(self.H)

        return float(reward_sparse + reward_dense)

    def _get_observation(self) -> np.ndarray:
        """
        Return the normalized observation vector.

        Returns:
            np.array([bx/W, by/H, vx/3, vy/3, py/H], dtype=float32)
        """
        obs = np.array(
            [
                self.bx / float(self.W),
                self.by / float(self.H),
                self.vx / 3.0,
                self.vy / 3.0,
                self.py / float(self.H),
            ],
            dtype=np.float32,
        )
        return obs

    def _check_terminal(self) -> Tuple[bool, bool]:
        """
        Check terminal conditions.

        Returns:
            terminated: True if bx < 0 (agent lost) or bx >= W (opponent lost).
            truncated: True if step count >= T_MAX.
        """
        terminated = bool(self.bx < 0 or self.bx >= self.W)
        truncated = bool(self._step_count >= self.T_MAX)
        return terminated, truncated

    def set_global_step(self, step: int) -> None:
        """
        Set the global training step counter (used for curriculum scheduling
        in environment dynamics, including dense reward alpha decay.
        This step should also be forwarded to `LeftPaddleOpponent`.

        Args:
            step: current global training step.
        """
        self._global_step = int(max(0, step))
