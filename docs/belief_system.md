# Belief System 🔮

Random Battles are a **POMDP** (Partially Observable Markov Decision Process). The opponent's moves, item, ability, and tera type are hidden until revealed.

The `BeliefTracker` (`src/belief_tracker.py`) uses **Bayesian inference** to convert hidden information into probability distributions.

## 1. Architecture

```
BeliefTracker
    └─ PokemonBelief (per opponent mon)
        ├─ role_probs: Dict[str, float]    # P(role | observations)
        ├─ observed_moves: Set[str]        # Confirmed moves
        ├─ confirmed_item: Optional[str]   # Revealed item
        ├─ confirmed_ability: Optional[str] # Revealed ability
        └─ observed_tera: Optional[str]    # Revealed tera type
```

## 2. Prior Distribution

Loaded from `gen9randombattle.json`:
- Each species has defined **roles** (e.g., Weavile: "Fast Physical Sweeper")
- Each role specifies likely moves, items, abilities, tera types
- Prior probabilities based on role usage rates

**Example**:
```
Weavile:
  roles:
    Fast Physical Sweeper: 80% usage
      moves: [Knock Off, Ice Spinner, Low Kick, Ice Shard]
      items: [Choice Band, Life Orb]
    Swords Dance: 20% usage
      moves: [Swords Dance, Triple Axel, ...]
```

## 3. Update Mechanisms

### A. Deductive Updates (Certainty)
Events that confirm facts with P=1.0:

| Observation | Update |
|-------------|--------|
| Move used | `observed_moves.add(move)`, eliminate incompatible roles |
| Item consumed | `confirmed_item = item` |
| Ability triggered | `confirmed_ability = ability` |
| Terastallized | `observed_tera = tera_type` |

### B. Role Probability Updates
When a move is observed:
1. Filter roles that can have this move
2. Renormalize probabilities over remaining roles
3. Roles without the move get P=0

**Example**:
- Prior: P(Sweeper)=0.8, P(Swords Dance)=0.2
- Observe: `Swords Dance` used
- Posterior: P(Sweeper)=0.0, P(Swords Dance)=1.0

### C. Speed Inference
When turn order reveals speed information:

| Event | Inference |
|-------|-----------|
| Slower mon moves first | P(Scarf) increases, P(Trick Room) considered |
| Outsped unexpectedly | P(Priority move) if damage, else P(Scarf)=1.0 |

**Example**:
- My Dragapult (421 Speed) vs opponent Gholdengo (max 293)
- Gholdengo moves first → Must have Choice Scarf
- Update: `P(Item=Choice Scarf) = 1.0`

### D. Move Lock Inference
If opponent uses same move twice with Choice item:
- `P(locked_into_move) = 1.0`
- Safe to switch into immunity

## 4. Embedding Output

`PokemonBelief.to_embedding(size=10)` returns:
| Index | Feature |
|-------|---------|
| 0–3 | Top 4 role probabilities |
| 4–7 | Top 4 unrevealed move type probabilities |
| 8 | Item revealed (0/1) |
| 9 | Moves revealed / 4 |

## 5. Integration with Policy

The `BeliefEncoder` feeds embeddings into the observation vector:
- 96 floats total (16 features × 6 opponent mons)
- Allows policy to reason about hidden information
- Enables informed switching against predicted movesets
