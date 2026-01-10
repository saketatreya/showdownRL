# Reward System

## Potential-Based Reward Shaping

The agent uses **potential-based reward shaping** derived from a win probability predictor trained on 100k expert replays.

### Formula
$$R' = R_{terminal} + \gamma \Phi(s') - \Phi(s)$$

Where:
- $\Phi(s)$ = Predicted win probability from state $s$
- $\gamma$ = Discount factor (0.99)
- $R_{terminal}$ = +1 for win, -1 for loss

### Win Predictor

**Architecture**: MLP (744 → 512 → 256 → 128 → 1)

**Training Data**:
- 100k Gen9 Random Battle replays (1300+ Elo)
- ~3M early/mid-game states (first 60% of each game)
- Symmetric augmentation (both perspectives)

**Features** (744 dimensions):
- Active Pokemon: HP, status, types, boosts, moves (×2 players)
- Team: HP, status per Pokemon (×6 per side)
- Field: Weather, terrain, hazards, screens
- Meta: Turn number, game phase

---

## Why Potential-Based?

Traditional dense rewards (damage dealt, KOs) can mislead the agent:
- Rewards dealing damage even in losing positions
- Doesn't capture long-term strategic value

Potential-based shaping:
- Preserves optimal policy (Ng et al., 1999)
- Rewards *improving* win probability, not just damage
- Learns strategic concepts: type matchups, hazards, momentum

---

## Usage

```bash
# Train with potential-based rewards
python scripts/train.py --reward-shaping potential --n-envs 8
```

Requires `models/win_predictor.pt` (train with `scripts/train_predictor.py`).