"""
Visualization and rendering utilities for the Pong environment.

Uses pygame for real-time window rendering and frame capture.
Captured frames can be exported as GIF/video for qualitative demonstration.
"""

import pygame
import numpy as np
from typing import List, Optional
from run.config import EnvConfig, RenderConfig
from PIL import Image


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
        width: int = EnvConfig.width,
        height: int = EnvConfig.height,
        scale: int = RenderConfig.scale,
        fps: int = RenderConfig.fps,
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
        self.width = width
        self.height = height
        self.scale = scale
        self.fps = fps

        self.paddle_w = EnvConfig.paddle_width
        self.paddle_h = EnvConfig.paddle_height
        self.left_x = EnvConfig.left_x
        self.right_x = EnvConfig.right_x

        pygame.init()
        pygame.font.init()
        self._clock = pygame.time.Clock()
        self._surface = pygame.Surface((self.width, self.height))
        self._screen = pygame.display.set_mode((self.width * self.scale, self.height * self.scale))
        pygame.display.set_caption("Horizontal Pong")
        self._font = pygame.font.SysFont("Arial", 12)
        self._big_font = pygame.font.SysFont("Arial", 18, bold=True)

    def render_frame(
        self,
        bx: int,
        by: int,
        py_agent: int,
        py_opponent: int,
        score_agent: int,
        score_opponent: int,
        overlay_text: Optional[str] = None,
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
        self._draw_on_surface(
            self._surface,
            bx,
            by,
            py_agent,
            py_opponent,
            score_agent,
            score_opponent,
            overlay_text=overlay_text,
        )
        scaled = pygame.transform.scale(
            self._surface, (self.width * self.scale, self.height * self.scale)
        )
        self._screen.blit(scaled, (0, 0))
        pygame.display.flip()
        self._clock.tick(self.fps)

    def capture_frame(
        self,
        bx: int,
        by: int,
        py_agent: int,
        py_opponent: int,
        score_agent: int,
        score_opponent: int,
        overlay_text: Optional[str] = None,
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
        offscreen = pygame.Surface((self.width, self.height))
        self._draw_on_surface(
            offscreen,
            bx,
            by,
            py_agent,
            py_opponent,
            score_agent,
            score_opponent,
            overlay_text=overlay_text,
        )
        arr = pygame.surfarray.array3d(offscreen)  # (W, H, 3)
        return np.transpose(arr, (1, 0, 2)).copy()  # (H, W, 3)

    def handle_events(self) -> bool:
        """
        Process pygame events (quit, keypress, etc.).

        Should be called each frame to keep the window responsive.

        Returns:
            running: False if the user closed the window, True otherwise.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        return True

    def save_gif(
        self, frames: List[np.ndarray], filepath: str, fps: int = RenderConfig.gif_fps
    ) -> None:
        """
        Save a sequence of captured frames as an animated GIF file using Pillow.

        Args:
            frames: list of RGB arrays, each of shape (H, W, 3).
            filepath: output path for the GIF file (e.g. "artifacts/rollout.gif").
            fps: frames per second.
        """
        if not frames:
            raise ValueError("frames list is empty")

        pil_frames = [Image.fromarray(frame) for frame in frames]
        duration_ms = int(1000 / max(1, fps))
        pil_frames[0].save(
            filepath,
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration_ms,
            loop=0,
        )

    def close(self) -> None:
        """
        Quit pygame and release all display resources.
        """
        pygame.quit()

    def _draw_on_surface(
        self,
        surface: pygame.Surface,
        bx: int,
        by: int,
        py_agent: int,
        py_opponent: int,
        score_agent: int,
        score_opponent: int,
        overlay_text: Optional[str] = None,
    ) -> None:
        """Draw current game state to a target surface."""
        black = (0, 0, 0)
        white = (255, 255, 255)

        surface.fill(black)

        # Dashed center line.
        for y in range(0, self.height, 8):
            pygame.draw.line(
                surface, white, (self.width // 2, y), (self.width // 2, min(y + 4, self.height))
            )

        # Paddles.
        left_rect = pygame.Rect(
            self.left_x,
            int(py_opponent - self.paddle_h // 2),
            self.paddle_w,
            self.paddle_h,
        )
        right_rect = pygame.Rect(
            self.right_x,
            int(py_agent - self.paddle_h // 2),
            self.paddle_w,
            self.paddle_h,
        )
        pygame.draw.rect(surface, white, left_rect)
        pygame.draw.rect(surface, white, right_rect)

        # Ball.
        ball_rect = pygame.Rect(int(bx), int(by), 2, 2)
        pygame.draw.rect(surface, white, ball_rect)

        # Score.
        # Show a single hits counter (agent side) on the left so it doesn't overlap
        # with the dashed center line.
        score_text = f"Hits: {score_agent}"
        text = self._font.render(score_text, True, white)
        surface.blit(text, (4, 4))

        # Optional overlay message (e.g., "YOU WON").
        if overlay_text:
            msg = self._big_font.render(str(overlay_text), True, white)
            msg_x = self.width // 2 - msg.get_width() // 2
            msg_y = self.height // 2 - msg.get_height() // 2
            surface.blit(msg, (msg_x, msg_y))
