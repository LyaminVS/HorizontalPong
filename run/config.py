"""
Centralized hyperparameter configuration for training and evaluation.

All parameters from the specification are collected here as dataclass defaults.
"""

from dataclasses import dataclass, field


@dataclass
class EnvConfig:
    """Environment parameters."""
    width: int = 86
    height: int = 64
    paddle_width: int = 2
    paddle_height: int = 12
    paddle_speed: int = 1
    max_ball_speed_x: int = 1
    max_ball_speed_y: int = 1
    left_x: int = 6
    right_x: int = 79
    t_max: int = 2000


@dataclass
class ActorCriticConfig:
    """Hyperparameters for the Actor-Critic agent."""
    state_dim: int = 5
    action_dim: int = 3
    hidden_dim: int = 128
    gamma: float = 0.99
    lr_actor: float = 3e-4
    lr_critic: float = 1e-4
    entropy_coeff: float = 0.5
    buffer_capacity: int = 10_000
    batch_size: int = 64
    update_every: int = 10


@dataclass
class ReinforceConfig:
    """Hyperparameters for the REINFORCE agent."""
    state_dim: int = 5
    action_dim: int = 3
    hidden_dim: int = 128
    gamma: float = 0.99
    lr_actor: float = 3e-4


@dataclass
class ReinforceBaselineConfig:
    """Hyperparameters for the REINFORCE with heuristic baseline agent."""
    state_dim: int = 5
    action_dim: int = 3
    hidden_dim: int = 128
    gamma: float = 0.99
    lr_actor: float = 3e-4


@dataclass
class TRPOConfig:
    """Hyperparameters for the TRPO agent (without value network)."""
    state_dim: int = 5
    action_dim: int = 3
    hidden_dim: int = 256
    gamma: float = 0.99
    max_kl: float = 0.001
    entropy_coeff: float = 0.001
    grad_clip_norm: float = 3.0
    damping: float = 0.05
    cg_iters: int = 10
    backtrack_iters: int = 10
    backtrack_coeff: float = 0.8


@dataclass
class TrainConfig:
    """Training pipeline parameters."""
    total_steps: int = 500_000
    seed: int = 42
    log_interval: int = 100
    save_interval: int = 50_000
    short_episode_on_opponent_hit: bool = False
    device: str = "cpu"
    artifacts_dir: str = "artifacts"


@dataclass
class EvalConfig:
    """Evaluation pipeline parameters."""
    num_episodes: int = 100
    seed: int = 123
    render: bool = False
    save_gif: bool = True
    device: str = "cpu"
    artifacts_dir: str = "artifacts"


@dataclass
class RenderConfig:
    """Rendering parameters."""
    scale: int = 4
    fps: int = 60
    gif_fps: int = 30


@dataclass
class RewardConfig:
    """Reward shaping parameters."""
    alpha_initial: float = 0.01
    alpha_decay_steps: int = 100_000


@dataclass
class CurriculumConfig:
    """Opponent difficulty curriculum schedule."""
    enabled: bool = False
    fixed_sigma: int = 0
    sigma_0_until: int = 50_000
    sigma_1_until: int = 150_000
