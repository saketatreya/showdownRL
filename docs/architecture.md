# System Architecture: Streamlined Pure Reinforcement Learning (Wang Methodology)

This document provides a comprehensive overview of the refactored ShowdownBot architecture, which implements a Pure Reinforcement Learning approach based on Wang's 2024 thesis. All curriculum learning, heuristic reward shaping, and hardcoded logic have been removed in favor of pure self-play driven by sparse terminal rewards.

---

## High-Level Architecture Diagram

```mermaid
graph TD
    subgraph "Training Pipeline (train.py)"
        Trainer[Trainer (RecurrentPPO)] -->|Push checkpts| CkptPool[Checkpoint Directory]
        Trainer -->|Spawn 4-8 Parallel Envs| VecEnv(SubprocVecEnv)
        
        VecEnv --> Env1[Gen9RLEnvironment 1]
        VecEnv --> Env2[Gen9RLEnvironment X]
        
        CkptPool -->|Sample 70% of time| SPPlayer[TrainedPlayer (Past Checkpoint)]
        SPPlayer -.->|Vs| Env1
    end

    subgraph "Environment Details (Gen9RLEnvironment)"
        B[Poke-Env Battle API] -->|Feeds data to| BT[BeliefTracker]
        B -->|Feeds data to| OB[ObservationBuilder]
        BT -->|Prior distributions| OB
        OB -->|Outputs 1163 Floats| LSTM[Maskable PPO LSTM Policy]
        LSTM -->|Select Action 0-25| AO[SinglesEnv action_to_order]
        AO -->|Execute| B
        
        B -->|Terminal Condition| R[Sparse Reward: +1 / -1 / 0]
        R --> LSTM
    end
```

## Core Components Breakdown

### 1. `Gen9RLEnvironment` (`src/environment.py`)
This is the central interface between the standard Gymnasium framework and the `poke-env` event loop. It handles resetting Pokémon Showdown battles, synchronizing steps, and interpreting rewards.
- **Sparse Rewards**: Intercepts the end of a battle. If the agent wins, it returns `+1.0`. If it loses, `-1.0`. For all other non-terminal turns, it strictly returns `0.0`. This forces the RL model to figure out strategies (like pivoting, setting hazards, or setting up stats) entirely autonomously without human-designed positive reinforcement loops ("reward hacking").

### 2. `ObservationBuilder` (`src/embeddings.py`) & `Encoders` (`src/encoders.py`)
Because neural networks only understand arrays of numbers, these components map the rich Pokémon Showdown JSON state into a `1163`-dimensional continuous state vector used by the LSTM.
- **Sub-Encoders**:
  - `ActivePokemonEncoder`: Encodes HP, typing, base stats, boosts, status, tera state/type, and item/ability flags.
  - `TeamEncoder`: Encodes HP, typing, status presence, and fainted/active/revealed flags for up to 6 mons.
  - `MovesEncoder`: Encodes the agent's 4 available moves (type, BP, accuracy, PP, priority, category, effectiveness, etc.).
  - `FieldEncoder`: Encodes active field elements such as Spikes, Stealth Rocks, Weather (Sun/Rain), and Terrains.
  - `DamageEncoder`: Ties into `damage_calc.py` to bake in theoretical minimum/maximum percentage damage rolls.

### 3. `BeliefTracker` (`src/belief_tracker.py`)
Gen 9 Random Battles contain *hidden information* (POMDP). You do not know your opponent's items, precise EVs/stats, or full move-sets until they reveal them.
- Uses Bayesian approximation to track probability distributions over the enemy's potential role out of the generated subsets from the `gen9randombattle.json` data dictionary.
- Whenever an opponent uses a move or triggers an item, it prunes impossible sets and updates the active probability matrix, which is then serialized and passed to the `BeliefEncoder`.

### 4. `TrainedPlayer` (`src/trained_player.py`)
A generalized wrapper that inherits from `poke_env`'s basic `Player` class but replaces human logic with standard `model.predict()` calls on the loaded `sb3_contrib` RecurrentPPO checkpoints.
- **Functionality**: Serves both as the opponent during Fictitious Self-Play (loading historically trained, frozen checkpoints) and as the primary deployment wrapper when evaluating your final bot against real players or random bot baselines.
- **Role in MCTS**: Given that full Monte Carlo Tree Search within Python requires an impossibly fast pure-Python Showdown forward-model simulator (which we lack, since we rely on the NodeJS backend), the `TrainedPlayer` effectively functions as the actor-critic inference evaluator (the ultimate goal of the MCTS described in Wang's thesis).

### 5. `ActionHandler` (`src/actions.py`)
A straightforward bidirectional mapping utility.
- Translates the flat discrete action space integers (0 through 25) into concrete `BattleOrder` directives via `poke_env.environment.SinglesEnv.action_to_order`.

### 6. Training Script (`scripts/train.py`)
The execution orchestrator.
- Uses `SubprocVecEnv` to spawn 4-8 parallel asynchronous environments connected to the local NodeJS Showdown server.
- Initiates Fictitious Self-Play: 70% of the parallel environments will fight against uniquely sampled past versions of the model itself from the `checkpoints` directory, while 30% fight baseline heuristic bots (such as `RandomPlayer`) to prevent catastrophic forgetting.
- Dictates the core Wang curriculum learning-rate decay function: `lr / (8x + 1)^1.5` over time.

---

## Cleanups Taken to Achieve This

The following systems existed in prior iterations but were explicitly deleted to achieve the current streamlined RL pipeline:
- **`src/curriculum_agents/`** (`OpponentPoolManager`, `WinRateTracker`): We no longer rotate heuristic opponents based on complex phase conditions. We just use random uncurated past checkpoints (Self-Play).
- **Dense Rewards (`src/rewards.py`)**: The model no longer receives fractional scalar points for matching types or maintaining high HP.
- **Thematic Teambuilding (`src/teams/`)**: Teams are generated natively and purely randomly by the server's `gen9randombattle` engine.
- **Unused Scripts**: Removed `scripts/tune_hyperparams.py` because it relied on the deleted curriculum components and didn't fit into the new simplified paradigm.

*(Note: Your IDE may still show `docs/rewards.md` as open, but it—along with `docs/curriculum_agents.md`—has been fully deleted from the filesystem as they are no longer relevant to the codebase.)*
