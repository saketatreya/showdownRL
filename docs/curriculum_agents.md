# Curriculum Agents 👨‍🏫

The agent learns through a progression of increasingly sophisticated opponents, managed by the `OpponentPoolManager` using an AlphaStar-style 3-tier system.

## 1. Tier Structure

### Tutorial Tier (Easy)
Foundation mechanics—teaches basic damage output and type awareness.

| Agent | Behavior | Lesson |
|-------|----------|--------|
| **MaxDamageBot** | Always picks highest base power move | Respect raw damage output |
| **TypePunisher** | Prioritizes super-effective moves | Understand type effectiveness |

### Tactical Tier (Mid)
Introduces positioning, trading, and field control.

| Agent | Behavior | Lesson |
|-------|----------|--------|
| **RevengeKiller** | Switches to faster mon to KO after ally faints | Momentum and speed tiers |
| **SacrificeTrader** | Sacks weak mons to get free switch-ins | Sacrifice for positioning |
| **HazardStacker** | Leads with hazards, pivots to spread damage | Entry hazard pressure |

### Strategic Tier (Hard)
Advanced concepts—setup, priority, and adaptability.

| Agent | Behavior | Lesson |
|-------|----------|--------|
| **SetupSweeper** | Uses stat-boost moves then sweeps | Threat of setup |
| **PrioritySniper** | Finishes low-HP threats with priority moves | Priority brackets |
| **PivotSpammer** | U-turn/Volt Switch for momentum | Pivot chains |
| **GreedPunisher** | Exploits predictable switches | Punishing greedy plays |
| **PassivityExploiter** | Stalls and capitalizes on passive play | Time pressure |

### Self-Play Tier
Past versions of the training agent (frozen checkpoints).

| Component | Description |
|-----------|-------------|
| **FrozenCheckpointPool** | Maintains last 10 snapshots of the model |
| **Lesson** | Prevents strategy cycling, forces adaptation |

## 2. Dynamic Sampling

Opponent selection uses a weighted system based on:
1. **Training Progress** (0.0 → 1.0)
2. **Win Rate per Tier** (mastery detection)

### Weight Calculation
```
Tutorial weight = max(0, 0.4 - 0.5*progress - 0.2*tutorial_wr)
Tactical weight = max(0, 0.35 - 0.2*abs(progress-0.3) - 0.2*tactical_wr)
Strategic weight = max(0, 0.15 + 0.3*progress - 0.2*strategic_wr)
Self-Play weight = 0.1 + 0.4*progress
```

### Example Distribution

| Progress | Tutorial | Tactical | Strategic | Self-Play |
|----------|----------|----------|-----------|-----------|
| 0% (Early) | 42% | 37% | 11% | 10% |
| 50% (Mid) | 20% | 30% | 25% | 25% |
| 90% (Late) | 5% | 15% | 30% | 50% |

## 3. Team Assignment

Each agent type uses themed teams:

| Agent | Team Source |
|-------|-------------|
| `hazard_stacker` | `src/teams/hazard_teams.py` (curated pool) |
| `setup_sweeper` | `src/teams/setup_teams.py` (curated pool) |
| `priority_sniper` | `src/teams/priority_teams.py` (curated pool) |
| `pivot_spammer` | `src/teams/pivot_teams.py` (curated pool) |
| `revenge_killer` | `src/teams/revenge_teams.py` (curated pool) |
| Others | `RandomBattleTeambuilder` (fully random) |

**Per-Game Rotation**: All teams rotate every game via `Teambuilder.yield_team()`.
