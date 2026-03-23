# Horizontal Pong — Actor-Critic vs REINFORCE

Reinforcement learning project: training an agent to play discrete horizontal Pong using **Actor-Critic with Q(s,a)-critic** and **REINFORCE**, then comparing their performance.

---

## Problem Definition

The agent controls the **right paddle** in a 2D Pong game. The ball moves horizontally, bouncing off walls and paddles. The goal is to deflect the ball as many times as possible, maximizing cumulative reward. The opponent (left paddle) is controlled by a rule-based algorithm with increasing difficulty (curriculum).

**Transitions are stochastic**: the opponent adds random noise to the ball's vertical velocity upon deflection. The noise level follows a curriculum schedule that increases with training progress.

---

## Environment Specification

### State Space

Integer vector of 5 components, normalized for neural network input:

| Variable | Range | Description |
|----------|-------|-------------|
| `bx` | 0..159 | Ball horizontal position |
| `by` | 0..119 | Ball vertical position |
| `vx` | {-2,-1,+1,+2} | Ball horizontal velocity |
| `vy` | {-3..+3} | Ball vertical velocity |
| `py` | PH/2..H-PH/2 | Agent paddle vertical center |

Normalization: `[bx/W, by/H, vx/3, vy/3, py/H]`

### Action Space

| Code | Action | Effect |
|------|--------|--------|
| 0 | Up | `py -= 1`, clamped at `PH/2` |
| 1 | Down | `py += 1`, clamped at `H-1-PH/2` |
| 2 | Stay | No change |

### Transition Logic

1. Ball advances: `bx' = bx + vx`, `by' = by + vy`.
2. Wall bounce: if `by' <= 0` or `by' >= H-1`, vertical velocity reverses.
3. Agent paddle hit: if `vx > 0`, `bx' >= RX`, and `|by' - py| <= PH/2` — ball reflects.
4. Opponent paddle hit: opponent always intercepts; adds noise `delta ~ Uniform{-sigma..+sigma}` to `vy`.

Opponent curriculum:

| Training steps | sigma | Behavior |
|----------------|-------|----------|
| 0 – 50k | 0 | Straight returns |
| 50k – 150k | 1 | Slight angle variation |
| 150k+ | 2 | Strong angle variation |

### Episode Termination

- `bx < 0` — agent missed the ball (terminated, reward = -1).
- `bx >= W` — opponent missed (terminated).
- `t >= 2000` — time limit reached (truncated).

### Reward Function

| Reward | Event |
|--------|-------|
| +1 | Agent deflects the ball |
| -1 | Agent misses the ball (`bx < 0`) |
| 0 | All other steps |

**Dense shaping** (when `vx > 0`): `r_dense = -alpha * |py - by| / H`, where `alpha` decays linearly from 0.01 to 0 over the first 100k steps.

---

## Repository Structure

```
HorizontalPong/
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── src/                          # Source code
│   ├── __init__.py
│   ├── environment/              # Pong environment (no gymnasium)
│   │   ├── __init__.py
│   │   ├── pong_env.py           # PongEnv: reset, step, seed, reward, transitions
│   │   └── renderer.py           # PongRenderer: frame rendering, GIF/video export
│   ├── agent/                    # RL agents and neural networks
│   │   ├── __init__.py
│   │   ├── networks.py           # ActorNetwork, CriticNetwork (MLP)
│   │   ├── actor_critic.py       # ActorCriticAgent: on-policy actor + off-policy critic
│   │   ├── reinforce.py          # ReinforceAgent: Monte Carlo policy gradient
│   │   └── replay_buffer.py      # ReplayBuffer for SARSA transitions
│   └── run/                      # Training and evaluation scripts
│       ├── __init__.py
│       ├── config.py             # Hyperparameter dataclasses
│       ├── train.py              # Training entry point
│       └── eval.py               # Evaluation and rollout recording
├── artifacts/                    # Model checkpoints, training logs, GIFs
│   └── .gitkeep
└── analysis/                     # Evaluation and visualization
    └── visualize.ipynb           # Jupyter notebook for learning curves & comparison
```

---

## Methods

### Actor-Critic (Q-critic)

- **Actor**: MLP `5→128→128→3` with Softmax, updated on-policy via advantage.
- **Critic**: MLP `8→128→128→1` (input: `[s_norm; one_hot(a)]`), trained off-policy from replay buffer using SARSA TD-error.
- **Advantage**: `A(s,a) = q_hat(s,a) - V^pi(s)`, with entropy bonus `beta * H(pi)`.
- **Update frequency**: every 10 steps, batch size 64.

### REINFORCE

- **Actor**: identical MLP `5→128→128→3` for fair comparison.
- **Update**: after each complete episode using Monte Carlo returns `G_t = sum gamma^k r_{t+k}`.
- **No critic, no replay buffer.**

---

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| W x H | 160 x 120 | Field size (pixels) |
| PW x PH | 2 x 12 | Paddle size |
| Actor MLP | 5→128→128→3 | Same for both methods |
| Critic MLP | 8→128→128→1 | Actor-Critic only |
| gamma | 0.99 | Discount factor |
| lr_actor | 3e-4 (Adam) | Same for both methods |
| lr_critic | 1e-4 (Adam) | Actor-Critic only |
| M (buffer) | 10,000 | Replay buffer capacity |
| B (batch) | 64 | Critic batch size |
| beta (entropy) | 0.01 | Entropy bonus coefficient |
| alpha (dense) | 0.01 → 0 | Dense reward decay over 100k steps |
| Total steps | 500,000 | Per method |

---

## Reproducibility

### Install dependencies

```bash
pip install -r requirements.txt
```

### Train

```bash
# Actor-Critic
python -m src.run.train --agent actor_critic --steps 500000 --seed 42

# REINFORCE
python -m src.run.train --agent reinforce --steps 500000 --seed 42
```

### Evaluate

```bash
# Actor-Critic
python -m src.run.eval --agent actor_critic --checkpoint artifacts/ac_model.pt --episodes 100

# REINFORCE
python -m src.run.eval --agent reinforce --checkpoint artifacts/reinforce_model.pt --episodes 100
```

### Expected Output

- **Model checkpoints**: `artifacts/ac_model.pt`, `artifacts/reinforce_model.pt`
- **Training logs**: `artifacts/train_log_ac.csv`, `artifacts/train_log_reinforce.csv`
- **Rollout GIFs**: `artifacts/rollout_ac.gif`, `artifacts/rollout_reinforce.gif`
- **Learning curves**: generated in `analysis/visualize.ipynb`

---

## Comparison Metrics

- Mean episode reward (sliding window of 100 episodes).
- Mean ball hits per episode.
- Steps to reach threshold: mean > 15 hits.
- Reward variance (training stability).

**Expected result**: Actor-Critic converges faster than REINFORCE due to TD updates, off-policy experience reuse, and lower gradient variance.
