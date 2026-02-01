# Setup Guide

## Prerequisites

- **Python 3.10+**
- **Node.js 18+** (for Pokemon Showdown server)
- **macOS / Linux** (Windows not tested)

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/saketatreya/showdownRL.git
cd showdownRL
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Clone Pokemon Showdown

The Showdown server is **not included** in this repo (too large). Clone it separately:

```bash
git clone https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown
npm install
cd ..
```

---

## Running the Server

Start the local Showdown server:

```bash
# Option 1: Using the helper script
./start_server.sh

# Option 2: Manual
cd pokemon-showdown
node pokemon-showdown start --no-security
```

The server runs on `localhost:8000` by default.

---

## Training

```bash
# 4 parallel environments, 10M timesteps (default)
python scripts/train.py --n-envs 4

# Quick test run
python scripts/train.py --dry-run

# Resume from checkpoint
python scripts/train.py --resume checkpoints/selfplay/model_50.zip
```

## Monitoring

```bash
tensorboard --logdir logs/
```

| Metric | Meaning |
|--------|---------|
| `rollout/ep_rew_mean` | Average episode reward (+1/-1) |
| `rollout/ep_len_mean` | Average battle length (turns) |

## Checkpoints

Saved to `checkpoints/<experiment>/`:
- `model_N.zip` - Periodic saves (used as self-play opponents)
- `model_final.zip` - End of training

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Already challenging" | Restart Showdown server |
| Server won't start | Check Node.js: `node --version` |
| OOM errors | Reduce `--n-envs` (try 2) |
| Connection refused | Ensure server is running: `curl http://localhost:8000` |
