# Training Guide

## Quick Start

```bash
# Terminal 1: Start Showdown server
./start_server.sh

# Terminal 2: Run training
python scripts/train.py --n-envs 8 --total-timesteps 1000000
```

## Command Line Options

| Flag | Default | Description |
|------|---------|-------------|
| `--total-timesteps` | 1M | Total environment steps |
| `--n-envs` | 8 | Parallel environments |
| `--reward-shaping` | potential | Reward type (potential only) |
| `--resume` | None | Resume from checkpoint |
| `--dry-run` | False | Quick 1000-step test |

## Training the Win Predictor

Before using potential-based shaping, train the win predictor:

```bash
# Scrape replays (if needed)
python scripts/scrape_replays.py --target-count 10000 --min-rating 1300

# Train predictor
python scripts/train_predictor.py

# Monitor progress
python scripts/monitor_training.py
```

## Monitoring

### Tensorboard
```bash
tensorboard --logdir logs/
```

| Metric | Meaning |
|--------|---------|
| `rollout/ep_rew_mean` | Average episode reward |
| `rollout/ep_len_mean` | Average battle length |
| `curriculum/winrate_*` | Win rate per opponent |

## Checkpoints

Saved to `checkpoints/<experiment>/`:
- `model_N.zip` - Periodic saves
- `model_final.zip` - End of training
- `frozen_iter_N.zip` - Self-play snapshots

## Resuming Training

```bash
python scripts/train.py --resume checkpoints/my_run/model_50.zip
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Already challenging" | Restart Showdown server |
| Server won't start | Check Node.js: `node --version` |
| OOM errors | Reduce `--n-envs` |
