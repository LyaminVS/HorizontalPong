"""
Visualization and rendering utilities for the Pong environment.

Uses pygame for real-time window rendering and frame capture.
Captured frames can be exported as GIF/video for qualitative demonstration.
"""

import pygame
import numpy as np
from typing import List, Optional


class PongRenderer:
    """
    Renders the Pong game state using a pygame window.

    Supports:
    - Real-time rendering in a pygame display at a configurable scale.
    - Capturing frames as numpy arrays for recording.
    - Saving episode rollouts as GIF animations.
    """

    def __init__(
        self,
        width: int = 160,
        height: int = 120,
        scale: int = 4,
        fps: int = 60,
    ) -> None:
        """
        Initialize pygame, create a display window, and set up a clock.

        The window size is (width * scale) x (height * scale) so the
        160x120 field is clearly visible.

        Args:
            width: field width in pixels.
            height: field height in pixels.
            scale: integer multiplier for the display window size.
            fps: target frames per second for real-time rendering.
        """
        raise NotImplementedError

    def render_frame(
        self,
        bx: int,
        by: int,
        py_agent: int,
        py_opponent: int,
        score_agent: int,
        score_opponent: int,
    ) -> None:
        """
        Draw a single game frame onto the pygame surface and flip the display.

        Draws on the internal surface at native resolution, then scales
        up to the display window.

        Elements drawn:
        - Black background.
        - Dashed center line.
        - Left paddle (opponent) and right paddle (agent) as white rectangles.
        - Ball as a white square.
        - Score text rendered with pygame.font at the top center.

        Ticks the clock to maintain the target FPS.

        Args:
            bx, by: ball position.
            py_agent: vertical center of the agent's paddle.
            py_opponent: vertical center of the opponent's paddle.
            score_agent: agent's current score (hits).
            score_opponent: opponent's current score.
        """
        raise NotImplementedError

    def capture_frame(
        self,
        bx: int,
        by: int,
        py_agent: int,
        py_opponent: int,
        score_agent: int,
        score_opponent: int,
    ) -> np.ndarray:
        """
        Draw a frame onto an off-screen surface and return it as a numpy array.

        Used for recording rollouts without displaying a window.

        Args:
            bx, by: ball position.
            py_agent: vertical center of the agent's paddle.
            py_opponent: vertical center of the opponent's paddle.
            score_agent: agent's current score (hits).
            score_opponent: opponent's current score.

        Returns:
            frame: np.ndarray of shape (H, W, 3), dtype uint8 (RGB).
        """
        raise NotImplementedError

    def handle_events(self) -> bool:
        """
        Process pygame events (quit, keypress, etc.).

        Should be called each frame to keep the window responsive.

        Returns:
            running: False if the user closed the window, True otherwise.
        """
        raise NotImplementedError

    def save_gif(self, frames: List[np.ndarray], filepath: str, fps: int = 30) -> None:
        """
        Save a sequence of captured frames as an animated GIF file using Pillow.

        Args:
            frames: list of RGB arrays, each of shape (H, W, 3).
            filepath: output path for the GIF file (e.g. "artifacts/rollout.gif").
            fps: frames per second.
        """
        raise NotImplementedError

    def close(self) -> None:
        """
        Quit pygame and release all display resources.
        """
        raise NotImplementedError
