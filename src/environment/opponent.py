"""
Opponent algorithm for the left paddle in Horizontal Pong.

This module isolates all logic related to the left paddle (algorithmic opponent):
- how the paddle tracks the ball,
- how opponent difficulty changes over training (sigma curriculum),
- how vertical noise is applied on opponent bounce.
"""

from typing import Optional
from run.config import EnvConfig


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
    ) -> None:
        """
        Initialize opponent geometry and internal global step.

        Args:
            field_height: environment height H.
            paddle_height: paddle height PH.
        """
        raise NotImplementedError

    def set_global_step(self, step: int) -> None:
        """
        Set current global training step for curriculum-based behavior.

        Args:
            step: current global training step.
        """
        raise NotImplementedError

    def get_sigma(self) -> int:
        """
        Return current noise level sigma according to curriculum.

        Returns:
            sigma: integer in {0, 1, 2}.
        """
        raise NotImplementedError

    def compute_paddle_center(self, ball_y: int) -> int:
        """
        Compute vertical center of the left paddle for current step.

        Baseline behavior from spec: opponent always reaches the ball, so
        this method can return a center aligned to the ball trajectory.

        Args:
            ball_y: current ball y-coordinate.

        Returns:
            ly: vertical center position for the left paddle.
        """
        raise NotImplementedError

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
        raise NotImplementedError
