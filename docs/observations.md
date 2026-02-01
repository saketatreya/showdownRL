# Observation Space 👁️

**Shape**: `(1163,)` Float32 Vector  
**Range**: `[-1.0, 4.0]` (normalized with some overflow headroom)

## Encoder Breakdown

| Encoder | Size | Description |
|---------|------|-------------|
| `ActivePokemonEncoder` × 2 | 194 | Our active + opponent active |
| `TeamEncoder` × 2 | 492 | Our bench + opponent bench (6 mons each) |
| `MovesEncoder` | 148 | Our 4 available moves (37 features × 4) |
| `OpponentMovesEncoder` | 40 | Known opponent moves |
| `MatchupEncoder` | 84 | Switch target effectiveness (21 × 4) |
| `DamageEncoder` | 25 | Damage prediction matrix (5 × 5) |
| `FieldEncoder` | 35 | Weather, terrain, hazards, rooms |
| `BeliefEncoder` | 96 | Probability distributions from BeliefTracker |
| `MetaEncoder` | 23 | Turn count, game phase, win probability |
| `ActionMaskEncoder` | 26 | Valid action mask |
| **Total** | **1163** | |

---

## 1. Active Pokemon (97 floats each)

| Indices | Feature | Format |
|---------|---------|--------|
| 0 | HP % | 0.0–1.0 |
| 1–18 | Type 1 | One-hot (18 types) |
| 19–36 | Type 2 | One-hot (18 types) |
| 37–42 | Stat Boosts | (stage+6)/12 for Atk/Def/SpA/SpD/Spe/Eva |
| 43–49 | Status | One-hot: None/Brn/Frz/Par/Psn/Slp/Tox |
| 50–56 | Volatiles | Binary: Confused/Taunted/Encore/Sub/Protect/Leech/Curse |
| 57 | Terastallized | Binary |
| 58–75 | Tera Type | One-hot (18 types) |
| 76–85 | Item Flags | Common item indicators |
| 86 | Fainted | Binary |
| 87–96 | Ability Flags | Common ability indicators |

---

## 2. Team Bench (246 floats per side)

Per-Pokemon encoding (41 floats × 6 slots):
- Species embedding (normalized ID)
- HP % (known for allies, estimated for enemies)
- Status condition
- Fainted flag
- Revealed moves count
- Type indicators

---

## 3. Moves (148 floats)

Per-Move encoding (37 floats × 4 moves):

| Feature | Description |
|---------|-------------|
| Base Power | Normalized (0–200 → 0–1) |
| Accuracy | Normalized (0–100 → 0–1) |
| PP Remaining | Fraction of max PP |
| Type | One-hot (18 types) |
| Category | One-hot: Physical/Special/Status |
| Priority | Normalized (-7 to +5 → 0–1) |
| STAB | Binary |
| Effectiveness | Log2 multiplier vs opponent |
| Expected Damage % | From damage calculator |
| Contact | Binary |
| Recoil | Fraction |

---

## 4. Field State (35 floats)

| Component | Encoding |
|-----------|----------|
| Weather | One-hot: Sun/Rain/Sand/Snow/None |
| Terrain | One-hot: Electric/Psychic/Misty/Grassy/None |
| Our Hazards | SR (0/1), Spikes (0–3), T-Spikes (0–2), Webs (0/1) |
| Opp Hazards | Same as above |
| Screens | Light Screen/Reflect/Aurora Veil (per side) |
| Rooms | Trick Room/Wonder Room/Magic Room |
| Turn Count | current_turn / 100 |

---

## 5. Belief State (96 floats)

Probability distributions from `BeliefTracker`:
- Move type probabilities per opponent mon (16 features × 6)
- Item probability (Choice Scarf, Leftovers, etc.)
- Ability probability
- Speed tier estimation

---

## 6. Meta Context (23 floats)

| Feature | Description |
|---------|-------------|
| Turn number | Normalized |
| Phase indicator | Early/Mid/Late (one-hot) |
| Momentum | Turns since damage dealt |
| Win probability | Estimated from HP totals |
| Opponent pattern | Detected playstyle signals |
