# System Architecture

## High-Level Diagram

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
│  │ poke-env     │  │ BeliefTracker│  │ PotentialReward      │   │
│  │ (Showdown)   │  │ (POMDP)      │  │ (Win Predictor)      │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### `Gen9RLEnvironment` (`src/environment.py`)
Custom Gymnasium environment bridging Python and Pokemon Showdown.
- **Parallelism**: `SubprocVecEnv` for multi-worker training
- **CurriculumWrapper**: Middleware for opponent swapping
- **Reward Shaping**: Potential-based shaping from win predictor

### `BeliefTracker` (`src/belief_tracker.py`)
Bayesian inference for hidden information.
- **Input**: Observed moves, damage, item/ability activations
- **Output**: Probability distributions for opponent sets

### `PotentialBasedRewardEvaluator` (`src/rewards.py`)
Reward shaping using win probability predictor.
- **Formula**: R' = R_terminal + γΦ(s') - Φ(s)
- **Φ(s)**: Win probability from neural network
- See [Reward System](rewards.md) for details

### `WinPredictor` (`src/win_predictor.py`)
Neural network for win probability estimation.
- **Architecture**: MLP (744 → 512 → 256 → 128 → 1)
- **Training**: 100k expert replays (first 60% of games)

### `ObservationBuilder` (`src/embeddings.py`)
Encodes battle state into neural network input.
- **Encoders**: Active, Team, Moves, Matchup, Field, Belief
- **Output**: 1149-dim float vector

### `OpponentPoolManager` (`src/curriculum_agents/opponent_pool.py`)
Opponent selection with progressive unlocking.
- **Tiers**: Random → Tactical → Strategic → Self-Play
- **Agents**: 10 heuristic bots + frozen checkpoints

## Wrapper Stack

```
Gen9RLEnvironment (core battle logic)
    └─ SingleAgentWrapper (poke-env)
        └─ Monitor (logging)
            └─ CurriculumWrapper (opponent management)
```

## Key Data Flows

| Flow | Path |
|------|------|
| **Action** | Policy → ActionHandler → WebSocket → Showdown |
| **Observation** | Showdown → Battle → Encoders → 1149 floats → Policy |
| **Reward** | Win Predictor → Potential Difference → scalar |
| **Curriculum** | CurriculumCallback → OpponentPool → new agent |
