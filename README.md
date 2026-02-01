# Pokemon Showdown RL Agent 🧠⚔️

A **Reinforcement Learning agent** for **Pokemon Showdown Random Battles** (Generation 9), implementing the methodology from [Wang (2024)](wang-jett-meng-eecs-2024-thesis.pdf). Uses RecurrentPPO with LSTM memory, Bayesian belief tracking for hidden information, and pure self-play training with sparse terminal rewards.

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
python scripts/train.py --n-envs 4
```

### Monitor Progress

```bash
tensorboard --logdir logs/
```

---

## 🏗️ Architecture

See [docs/architecture.md](docs/architecture.md) for the full breakdown.

```
Training Pipeline
 ├─ RecurrentPPO (LSTM 256) ──→ SubprocVecEnv (4-8 workers)
 │                                  ├─ 70% Self-Play (frozen checkpoints)
 │                                  └─ 30% RandomPlayer baseline
 └─ Checkpoint Pool ──→ Historical snapshots for Fictitious Self-Play

Environment (Gen9RLEnvironment)
 ├─ poke-env (Showdown WebSocket)
 ├─ BeliefTracker (Bayesian POMDP inference)
 ├─ ObservationBuilder (1163-dim state vector)
 └─ Sparse Reward: +1 (win) / -1 (loss) / 0 (other)
```

### Key Design Decisions

- **Sparse Rewards**: No intermediate reward shaping. The agent must learn strategy purely from win/loss signals.
- **Self-Play**: 70% of training environments fight frozen historical checkpoints (Fictitious Self-Play).
- **Wang LR Schedule**: Learning rate decays as `lr / (8x + 1)^1.5` over training.
- **POMDP Handling**: Bayesian belief tracker infers opponent items, moves, and abilities from observations.

---

## 📂 Project Structure

```
├── src/
│   ├── environment.py         # Gymnasium environment wrapper
│   ├── encoders.py            # Observation space encoders (1163 dims)
│   ├── embeddings.py          # ObservationBuilder
│   ├── belief_tracker.py      # Bayesian inference for hidden info
│   ├── damage_calc.py         # Damage projection engine
│   ├── shadow_battle.py       # Lightweight battle state proxy
│   ├── actions.py             # Action space handler (0-25 mapping)
│   ├── trained_player.py      # Inference wrapper for trained models
│   ├── callbacks.py           # TensorBoard + ELO callbacks
│   ├── elo_tracker.py         # ELO rating system
│   ├── config.py              # Training hyperparameters
│   ├── utils.py               # Shared utilities
│   └── teams/
│       ├── random_battle_teambuilder.py  # Random team generation
│       └── team_validator.py             # Team format validation
├── scripts/
│   ├── train.py               # Main training script (PPO + self-play)
│   └── play_against_bot.py    # Interactive play with debug visualization
├── docs/                      # Documentation
├── gen9randombattle.json       # Pokemon role/moveset data dictionary
└── requirements.txt
```

---

## 📝 License

MIT License

## 🙏 Acknowledgments

- [Wang (2024)](wang-jett-meng-eecs-2024-thesis.pdf) - *Winning at Pokémon Random Battles Using Reinforcement Learning* (MIT M.Eng Thesis)
- [poke-env](https://github.com/hsahovic/poke-env) - Pokemon Showdown Python interface
- [Pokemon Showdown](https://pokemonshowdown.com/) - Battle simulator
- [Stable Baselines 3](https://stable-baselines3.readthedocs.io/) - RL algorithms
