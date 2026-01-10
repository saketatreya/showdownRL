# Reward System 🍬

## 1. Core Mechanism: Delta-Based Shaping

Rewards are based on *changes* in state, not absolute values. This prevents reward farming.

$$R_{total} = \sum_{c} w_c \times \text{clamp}(\Delta V_c, -1, 1) + \text{bonuses} - \text{penalties}$$

---

## 2. Reward Components

### A. Fainted Differential
$$\Delta_{fainted} = (\text{OppFainted}_t - \text{OppFainted}_{t-1}) - (\text{AllyFainted}_t - \text{AllyFainted}_{t-1})$$
- **+1.0** per opponent KO
- **-1.0** per ally KO

### B. HP Balance
$$\Delta_{hp} = \sum_{i} \frac{\text{OppHP}_i^{t-1} - \text{OppHP}_i^t}{\text{MaxHP}_i} - \sum_{j} \frac{\text{AllyHP}_j^{t-1} - \text{AllyHP}_j^t}{\text{MaxHP}_j}$$
- Rewards dealing damage, penalizes taking damage

### C. Type Matchup Advantage
$$V_{matchup} = \text{mean}_{active} \left( \log_2(\text{TypeMultiplier}) \right)$$

| Multiplier | Score | Meaning |
|------------|-------|---------|
| 4.0× | +2.0 | Double super-effective |
| 2.0× | +1.0 | Super-effective |
| 1.0× | 0.0 | Neutral |
| 0.5× | -1.0 | Resisted |
| 0.0× | -3.0 | Immune (capped) |

### D. Stat Boosts
$$\Delta_{boosts} = \sum_{s \in \text{Stats}} (\text{Stage}_s^t - \text{Stage}_s^{t-1})$$
- Capped at `boost_cap` (default: 3)
- Scaled by `boost_multiplier` (default: 0.02)

### E. Entry Hazards
- **Stealth Rock**: +1.0 when set on opponent side
- **Spikes**: +0.5 per layer
- **Toxic Spikes**: +0.3 per layer

---

## 3. Phase-Based Weights

Weights change with training progress (`src/config.py: REWARD_CURRICULUM`):

| Component | Early (0–20%) | Mid (20–50%) | Late (50%+) |
|-----------|---------------|--------------|-------------|
| **Goal** | **Damage** | **Strategy** | **Winning** |
| `w_fainted` | 4.0 | 3.0 | 2.5 |
| `w_hp` | 1.5 | 1.0 | 0.5 |
| `w_matchup` | 0.4 | 0.5 | 0.4 |
| `w_boosts` | 0.1 | 0.3 | 0.15 |
| `w_hazards` | 0.05 | 0.1 | 0.05 |
| `w_status` | 0.2 | 0.3 | 0.2 |
| `victory_bonus` | 15.0 | 18.0 | 25.0 |
| `defeat_penalty` | -12.0 | -14.0 | -20.0 |
| `switch_tax` | 0.3 | 0.25 | 0.25 |
| `step_cost` | 0.005 | 0.01 | 0.02 |

---

## 4. Penalties & Bonuses

### Switch Tax
- **Cost**: `-switch_tax` per switch action
- **Purpose**: Prevents switch-spam; switching must gain matchup value > tax

### Attack Bonus
- **Bonus**: `+attack_bonus` when using an attacking move
- **Purpose**: Encourages damage dealing over passive play

### Move Fail Penalty
- **Cost**: `-move_fail_penalty` when move fails (miss, immune, etc.)
- **Purpose**: Discourages reckless move choices

### Step Cost
- **Cost**: `-step_cost` per turn
- **Purpose**: Encourages efficient wins; prevents stalling

### Momentum Penalty
- **Cost**: `-momentum_penalty` per turn without dealing damage
- **Grace Period**: `momentum_grace_turns` before penalty applies
- **Purpose**: Prevents setup farming

---

## 5. Terminal Rewards

| Outcome | Reward |
|---------|--------|
| Victory | `+victory_bonus` |
| Defeat | `+defeat_penalty` (negative) |

---

## 6. Implementation

The reward is calculated by `HybridRewardEvaluator` in `src/rewards/hybrid.py`:
1. Tracks previous battle state
2. Computes deltas for each component
3. Applies phase-appropriate weights
4. Adds bonuses/penalties based on action type
5. Returns scalar reward