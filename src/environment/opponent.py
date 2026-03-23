"""
Opponent algorithm for the left paddle in Horizontal Pong.

This module isolates all logic related to the left paddle (algorithmic opponent):
- how the paddle tracks the ball,
- how opponent difficulty changes over training (sigma curriculum),
- how vertical noise is applied on opponent bounce.
"""

from typing import Optional

import numpy as np

from run.config import EnvConfig
from run.config import CurriculumConfig


class LeftPaddleOpponent:
    """
    Rule-based controller for the left paddle.

    The opponent is designed to always intercept the incoming ball and then
    perturb the reflected trajectory via controlled noise.

    Curriculum schedule for noise sigma:
        steps 0–50k:      sigma = 0
        steps 50k–150k:   sigma = 1
        steps 150k+:      sigma = 2
    """

    def __init__(
        self,
        field_height: int = EnvConfig.height,
        paddle_height: int = EnvConfig.paddle_height,
        paddle_speed: int = EnvConfig.paddle_speed,
    ) -> None:
        """
        Initialize opponent geometry and internal global step.

        Args:
            field_height: environment height H.
            paddle_height: paddle height PH.
            paddle_speed: paddle movement speed in pixels per step.
        """
        self.field_height = int(field_height)
        self.paddle_height = int(paddle_height)
        self.paddle_speed = int(max(1, paddle_speed))
        self._global_step = 0

    def set_global_step(self, step: int) -> None:
        """
        Set current global training step for curriculum-based behavior.

        Args:
            step: current global training step.
        """
        self._global_step = int(max(0, step))

    def get_sigma(self) -> int:
        """
        Return current noise level sigma according to curriculum.

        Returns:
            sigma: integer in {0, 1, 2}.
        """
        if self._global_step < CurriculumConfig.sigma_0_until:
            return 0
        if self._global_step < CurriculumConfig.sigma_1_until:
            return 1
        return 2

    def compute_paddle_center(
        self,
        current_ly: int,
        ball_x: int,
        ball_y: int,
        vx: int,
        vy: int,
        target_x: int,
    ) -> int:
        """
        Compute vertical center of the left paddle for current step.

        Uses anticipatory control without teleportation:
        - predict y-coordinate where ball will reach target_x (left paddle line),
        - move paddle center toward that target with configured paddle speed.
        - if ball is moving away from opponent (vx >= 0), keep paddle at center.

        Args:
            current_ly: current vertical center of the left paddle.
            ball_x: current ball x-coordinate.
            ball_y: current ball y-coordinate.
            vx: current ball x-velocity.
            vy: current ball y-velocity.
            target_x: x-coordinate of the left paddle.

        Returns:
            ly: vertical center position for the left paddle.
        """
        low = self.paddle_height // 2
        high = self.field_height - 1 - self.paddle_height // 2

        if vx >= 0:
            y_target = self.field_height // 2
        else:
            y_target = self._predict_intercept_y(
                ball_x=ball_x, ball_y=ball_y, vx=vx, vy=vy, target_x=target_x
            )

        if current_ly < y_target:
            next_ly = current_ly + self.paddle_speed
        elif current_ly > y_target:
            next_ly = current_ly - self.paddle_speed
        else:
            next_ly = current_ly

        return int(np.clip(next_ly, low, high))

    def apply_bounce_noise(self, vy: int, rng) -> int:
        """
        Apply opponent bounce noise to vertical velocity.

        Noise rule:
            delta ~ Uniform{-sigma, ..., +sigma}
            vy_new = clamp(vy + delta, -3, +3)

        Args:
            vy: current vertical ball velocity.
            rng: random generator (e.g., numpy Generator).

        Returns:
            vy_new: velocity after stochastic perturbation.
        """
        sigma = self.get_sigma()
        if sigma == 0:
            return int(np.clip(vy, -3, 3))
        delta = int(rng.integers(-sigma, sigma + 1))
        return int(np.clip(vy + delta, -3, 3))

    def _predict_intercept_y(
        self, ball_x: int, ball_y: int, vx: int, vy: int, target_x: int
    ) -> int:
        """
        Predict y-position of the ball when it reaches target_x.

        Simulates integer dynamics with wall bounces exactly as environment does.
        """
        x = int(ball_x)
        y = int(ball_y)
        vx_cur = int(vx)
        vy_cur = int(vy)

        # Safety cap in case of unexpected dynamics.
        for _ in range(10000):
            if vx_cur < 0 and x <= target_x:
                break
            x += vx_cur
            y += vy_cur

            if y <= 0:
                y = 0
                vy_cur = abs(vy_cur)
            elif y >= self.field_height - 1:
                y = self.field_height - 1
                vy_cur = -abs(vy_cur)

        return int(y)
