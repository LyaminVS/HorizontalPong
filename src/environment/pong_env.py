"""
Discrete Horizontal Pong environment.

Field: W x H = 160 x 120 pixels.
Paddles: PW x PH = 2 x 12 pixels, positioned vertically (left at x=6, right at x=153).
Ball speed: |vx| in {1, 2}, vy in {-3, ..., +3}.
State: (bx, by, vx, vy, py) — integer vector.
Actions: 0=Up, 1=Down, 2=Stay.
No gymnasium dependency.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any


class PongEnv:
    """
    Custom RL environment for discrete horizontal Pong.
    Implements reset/step/seed API without gymnasium.

    Attributes:
        W, H: field dimensions (160 x 120)
        PW, PH: paddle dimensions (2 x 12)
        LX, RX: x-coordinates of left (6) and right (153) paddles
        T_max: maximum episode length (2000 steps)
        observation_space: dict describing state bounds
        action_space: dict describing valid actions {0, 1, 2}
    """

    # --- Constants ---
    W: int = 160
    H: int = 120
    PW: int = 2
    PH: int = 12
    LX: int = 6
    RX: int = 153
    T_MAX: int = 2000

    def __init__(self) -> None:
        """
        Initialize the environment: set field parameters, define observation_space
        and action_space dicts, initialize internal state to None.
        """
        raise NotImplementedError

    def seed(self, seed: Optional[int] = None) -> None:
        """
        Set the random seed for reproducibility.

        Args:
            seed: integer seed for numpy RNG. If None, use a random seed.
        """
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

    def _move_ball(self) -> None:
        """
        Advance the ball by its velocity: bx += vx, by += vy.
        Handle top/bottom wall bounces:
            if by <= 0:  by = 0,     vy = +|vy|
            if by >= H-1: by = H-1,  vy = -|vy|
        """
        raise NotImplementedError

    def _check_agent_paddle_hit(self) -> bool:
        """
        Check if the ball hits the agent's (right) paddle.

        Condition: vx > 0 and bx >= RX and |by - py| <= PH/2.
        On hit: vx = -|vx|, bx = RX - 1.
        Optional angular bounce: vy += sign(by - py).

        Returns:
            True if the ball was deflected by the agent paddle.
        """
        raise NotImplementedError

    def _check_opponent_paddle_hit(self) -> None:
        """
        Check if the ball hits the opponent's (left) paddle.

        The opponent always reaches the ball (ly = by).
        On hit: vx = +|vx|, bx = LX + 1.
        Add random noise delta to vy: vy = clamp(vy + delta, -3, +3),
        where delta ~ Uniform{-sigma, ..., +sigma}.

        Sigma follows curriculum:
            steps 0–50k:      sigma = 0
            steps 50k–150k:   sigma = 1
            steps 150k+:      sigma = 2
        """
        raise NotImplementedError

    def _get_opponent_sigma(self) -> int:
        """
        Return the opponent noise level sigma based on the global step count
        (curriculum schedule).

        Returns:
            sigma: 0, 1, or 2.
        """
        raise NotImplementedError

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
        raise NotImplementedError

    def _get_observation(self) -> np.ndarray:
        """
        Return the normalized observation vector.

        Returns:
            np.array([bx/W, by/H, vx/3, vy/3, py/H], dtype=float32)
        """
        raise NotImplementedError

    def _check_terminal(self) -> Tuple[bool, bool]:
        """
        Check terminal conditions.

        Returns:
            terminated: True if bx < 0 (agent lost) or bx >= W (opponent lost).
            truncated: True if step count >= T_MAX.
        """
        raise NotImplementedError

    def set_global_step(self, step: int) -> None:
        """
        Set the global training step counter (used for curriculum scheduling
        of opponent sigma and dense reward alpha decay).

        Args:
            step: current global training step.
        """
        raise NotImplementedError
