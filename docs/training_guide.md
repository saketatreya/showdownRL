# Training Guide 🏋️‍♂️

## 1. Quick Start

```bash
# Start local Showdown server (in pokemon-showdown directory)
node pokemon-showdown start --no-security

# Run training (in showdownbot directory)
python scripts/train.py --timesteps 10000000 --n-envs 7 --experiment-name my_run
```

## 2. Key Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--timesteps` | 10M | Total environment steps to train |
| `--n-envs` | 7 | Parallel environments (set to CPU cores - 1) |
| `--n-steps` | 2048 | Steps per rollout before update |
| `--experiment-name` | hybrid_ladder | Checkpoint directory name |
| `--resume` | None | Path to checkpoint to resume from |
| `--dry-run` | False | Short 1000-step test run |
| `--no-subproc` | False | Use DummyVecEnv (for debugging) |

## 3. Understanding Logs

### Terminal Output
```
=== Iteration 5 (Step 10000/10000000) ===
Phase:    EARLY
Progress: 0.10% (LR: 5.8e-05)
```
- **Phase**: Current curriculum phase (EARLY/MID/LATE)
- **Progress**: Percentage of total timesteps completed

### Key Metrics (Tensorboard)
```bash
tensorboard --logdir logs/
```

| Metric | Meaning |
|--------|---------|
| `rollout/ep_rew_mean` | Average episode reward (higher = better) |
| `rollout/ep_len_mean` | Average battle length in turns |
| `train/approx_kl` | Policy change magnitude (should stay < 0.05) |
| `train/entropy_loss` | Exploration level (decreases over time) |
| `curriculum/winrate_*` | Win rate against each agent type |

### Reward Interpretation
| Range | Meaning |
|-------|---------|
| -150 to -80 | Random/broken policy |
| -80 to -40 | Learning basics |
| -40 to 0 | Competent play |
| > 0 | Winning more than losing |

## 4. Curriculum Phases

Automatically advances based on training progress:

| Phase | Progress | Focus | Reward Weight |
|-------|----------|-------|---------------|
| EARLY | 0–20% | Damage & KOs | High `w_fainted` |
| MID | 20–50% | Strategy & Setup | High `w_matchup` |
| LATE | 50%+ | Winning Games | High `victory_bonus` |

## 5. Checkpoints

Saved to `checkpoints/<experiment_name>/`:
- `model_N.zip` - Periodic saves (every ~50k steps)
- `model_final.zip` - End of training
- `frozen_iter_N.zip` - Self-play snapshots
- `win_tracker.json` - Per-agent win rate history

## 6. Resuming Training

```bash
python scripts/train.py --resume checkpoints/my_run/model_50.zip --timesteps 20000000
```

## 7. Watching the Bot

```bash
# Launch dashboard
streamlit run scripts/dashboard.py

# In browser: Load model, click "Watch Self-Play" or "Play Against Bot"
```

## 8. Troubleshooting

| Issue | Solution |
|-------|----------|
| "Already challenging" loop | Restart Showdown server |
| "Team was rejected" | Check `gen9randombattle.json` for invalid Pokemon |
| Low FPS (< 50) | Reduce `--n-envs` |
| OOM errors | Reduce `--n-envs` or `--n-steps` |
