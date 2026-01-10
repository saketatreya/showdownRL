# Pokemon Showdown RL Agent 🧠⚔️

![Status](https://img.shields.io/badge/Status-Research-blue.svg) ![Python](https://img.shields.io/badge/Python-3.10+-blue.svg) ![Framework](https://img.shields.io/badge/Framework-SB3_Contrib-orange.svg)

A **Reinforcement Learning agent** for **Pokemon Showdown Random Battles** (Generation 9). Uses LSTM-based memory, Bayesian belief tracking, and potential-based reward shaping from expert replays.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+

### Installation

```bash
# Clone repository
git clone https://github.com/saketatreya/showdownRL.git
cd showdownRL

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Clone Pokemon Showdown (external dependency)
git clone https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown && npm install && cd ..
```

### Start Training

```bash
# Terminal 1: Start Showdown server
./start_server.sh

# Terminal 2: Run training
python scripts/train.py --n-envs 8 --total-timesteps 1000000
```

### Monitor Progress

```bash
# Live training dashboard
python scripts/monitor_training.py
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       Training Loop (train.py)                  │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────────────┐   │
│  │ RecurrentPPO│   │ SubprocVecEnv│  │ CurriculumWrapper   │   │
│  │ (LSTM 256)  │ → │ (8 workers) │ → │ (Opponent Sampling) │   │
│  └─────────────┘   └─────────────┘   └─────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Environment (Gen9RLEnvironment)               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │ poke-env     │  │ BeliefTracker│  │ RewardEvaluator      │   │
│  │ (Showdown)   │  │ (POMDP)      │  │ (Hybrid/Potential)   │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Observation Space (1149 dims)

| Component | Dims | Description |
|-----------|------|-------------|
| Active Pokemon | 156 | HP, status, stats, moves, type |
| Team State | 468 | 6 × 78 features per Pokemon |
| Opponent Beliefs | 468 | Bayesian estimates of hidden info |
| Field State | 57 | Weather, terrain, hazards, screens |

---

## 🎯 Reward Shaping

### Option 1: Hybrid Rewards (Default)
Dense rewards based on battle momentum:
```python
python scripts/train.py --reward-shaping hybrid
```

### Option 2: Potential-Based Rewards
Uses a **Win Predictor** trained on 100k expert replays:
```python
# First, train the win predictor (if not already done)
python scripts/train_predictor.py

# Then train with potential-based shaping
python scripts/train.py --reward-shaping potential
```

The potential-based approach uses:
- R' = R + γΦ(s') - Φ(s)
- Where Φ(s) = P(win | state) from the neural network

---

## 📂 Project Structure

```
├── src/
│   ├── environment.py      # Gymnasium environment wrapper
│   ├── rewards.py          # Reward evaluators (Hybrid, Potential)
│   ├── encoders.py         # Observation space encoders
│   ├── belief_tracker.py   # Bayesian inference for hidden info
│   ├── win_predictor.py    # Neural network for win probability
│   ├── replay_parser.py    # Parse Showdown replays for training
│   ├── curriculum_agents/  # Scripted opponent bots
│   └── teams/              # Team builders
├── scripts/
│   ├── train.py            # Main RL training script
│   ├── train_predictor.py  # Win predictor training
│   ├── scrape_replays.py   # Download replays from Showdown
│   ├── play_against_bot.py # Interactive play
│   └── monitor_*.py        # Training monitors
├── docs/                   # Documentation
└── requirements.txt
```

---

## 🔬 Research Features

### Curriculum Learning
Progressive opponent unlocking based on training progress:
- 0%+: Random, MaxDamage, Heuristic
- 10%+: TypePunisher
- 20%+: Self-play (frozen checkpoints)
- 50%+: PassivityExploiter

### Win Predictor
Supervised learning on 100k high-Elo replays:
- 744-dimensional feature vector
- MLP architecture (512 → 256 → 128 → 1)
- ~57% validation accuracy on early/mid-game states

---

## 📝 License

MIT License

## 🙏 Acknowledgments

- [poke-env](https://github.com/hsahovic/poke-env) - Pokemon Showdown Python interface
- [Pokemon Showdown](https://pokemonshowdown.com/) - Battle simulator
- [Stable Baselines 3](https://stable-baselines3.readthedocs.io/) - RL algorithms
