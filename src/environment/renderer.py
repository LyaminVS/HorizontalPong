"""
Visualization and rendering utilities for the Pong environment.

Provides real-time rendering via matplotlib and GIF/video export
for qualitative demonstration of trained agents.
"""

import numpy as np
from typing import List, Optional


class PongRenderer:
    """
    Renders the Pong game state for visualization and recording.

    Supports:
    - Frame-by-frame rendering via matplotlib.
    - Saving episode rollouts as GIF animations.
    """

    def __init__(self, width: int = 160, height: int = 120) -> None:
        """
        Initialize the renderer with field dimensions.

        Args:
            width: field width in pixels.
            height: field height in pixels.
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
    ) -> np.ndarray:
        """
        Draw a single game frame as a numpy RGB array.

        Draws:
        - Black background.
        - White top/bottom borders.
        - Left paddle (opponent) and right paddle (agent) as white rectangles.
        - Ball as a white square.
        - Optional score overlay.

        Args:
            bx, by: ball position.
            py_agent: vertical center of the agent's paddle.
            py_opponent: vertical center of the opponent's paddle.
            score_agent: agent's current score (hits).
            score_opponent: opponent's current score.

        Returns:
            frame: np.ndarray of shape (H, W, 3), dtype uint8.
        """
        raise NotImplementedError

    def show_frame(self, frame: np.ndarray) -> None:
        """
        Display a single frame in a matplotlib window (non-blocking).

        Args:
            frame: RGB image array of shape (H, W, 3).
        """
        raise NotImplementedError

    def save_gif(self, frames: List[np.ndarray], filepath: str, fps: int = 30) -> None:
        """
        Save a sequence of frames as an animated GIF file.

        Args:
            frames: list of RGB arrays, each of shape (H, W, 3).
            filepath: output path for the GIF file (e.g. "artifacts/rollout.gif").
            fps: frames per second.
        """
        raise NotImplementedError

    def save_video(self, frames: List[np.ndarray], filepath: str, fps: int = 30) -> None:
        """
        Save a sequence of frames as an MP4 video file.

        Args:
            frames: list of RGB arrays, each of shape (H, W, 3).
            filepath: output path for the video file (e.g. "artifacts/rollout.mp4").
            fps: frames per second.
        """
        raise NotImplementedError

    def close(self) -> None:
        """
        Close any open rendering windows and release resources.
        """
        raise NotImplementedError
