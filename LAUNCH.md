#Launch Guide: Horizontal Pong RL

This project is fully containerized using **Docker**. This ensures that all dependencies (PyTorch, Pygame, NumPy) are handled correctly, and training runs safely in a "headless" mode (without a physical monitor) using a virtual frame buffer.

---

## 1. Docker Setup

The build scripts automatically map your local **User ID (UID)** and **Group ID (GID)** into the container. This prevents permission issues where saved models or GIFs would otherwise be owned by `root`.

### Build the Image
Grant execution permissions to the scripts (one-time setup):
```bash
chmod +x build.sh launch_container.sh
```

**Build the Docker image:**
```bash
./build.sh
```

### Start the Container
**Launch the container and enter the interactive shell:**
```bash
./launch_container.sh
```
> **Note:** Your local project directory is mounted to `/app` inside the container. Any code changes made on your host machine are immediately reflected inside the container. All artifacts (checkpoints, logs, GIFs) generated inside will appear in your local `artifacts/` folder.

---

## 2. Training Agents

The `train.py` script handles the training loops for different algorithms.
Supported agents: `actor_critic`, `trpo`, `reinforce`, `reinforce_baseline`.

```bash
# Train the Actor-Critic agent (Recommended)
python -m run.train --agent actor_critic --steps 500000

# Train TRPO with a specific seed and device
python -m run.train --agent trpo --steps 200000 --seed 42 --device cpu
```
*   **Artifacts:** Model checkpoints (`.pt`) and training logs (`.csv`) are saved to the `artifacts/` directory.
*   **Headless:** Pygame rendering is disabled during training via `SDL_VIDEODRIVER=dummy` to prevent crashes on servers.

---

## 3. Evaluation & Visualization

Use `eval.py` to test a trained model, collect performance metrics, and record gameplay.

```bash
# Evaluate a model and save a GIF of the rollout
python -m run.eval --agent actor_critic --checkpoint artifacts/actor_critic_model.pt --save-gif

# Run evaluation over 100 episodes to get mean hits/reward
python -m run.eval --agent trpo --checkpoint artifacts/trpo_model.pt --episodes 100
```

---

## 4. Model vs Model (PvP Tournament)

This script pits two trained agents against each other. It swaps sides automatically to ensure fairness and identifies the superior policy.

```bash
# Run 100 matches between TRPO and Actor-Critic
python -m run.play_pvp --episodes 100
```
*   **Metrics:** Outputs a Win Rate matrix and average rally lengths to the console.
*   **Timelapse GIF:** Automatically records the longest rally of the tournament and saves it as a high-speed timelapse: `artifacts/trpo_vs_ac_best_rally_timelapse.gif`.

---

## 5. Analytics & Notebooks

A **Jupyter Notebook** server starts automatically when you run `./launch_container.sh`.

1.  Open your browser and navigate to: `http://localhost:8890`
2.  Use the notebooks in the `analysis/` folder to:
    *   Plot learning curves (Reward, Hits, Loss).
    *   Visualize Policy Decision Maps (Heatmaps).
    *   Compare ablation studies (Buffer capacity, Hidden dimensions).

---

## 6. Manual Play (Optional)

If you are on a local machine with a screen (and NOT inside a basic Docker container), you can play against the rule-based AI yourself:

```bash
# Control the right paddle with Up/Down arrows or W/S keys
python -m run.play
```
*Note: This requires a valid X11 display/monitor and will not work inside a standard headless Docker environment.*

---

## Summary of Results

*   **Actor-Critic:** Highest sample efficiency and final score (~33 hits/episode).
*   **TRPO:** High stability and reliable defense, though slower to train.
*   **REINFORCE:** Serves as a baseline; typically struggles with sparse rewards in this environment.