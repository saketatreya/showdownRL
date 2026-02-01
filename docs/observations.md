# Observation Space

**Shape**: `(1163,)` float32 vector  
**Range**: `[-1.0, 4.0]` (normalized features with some overflow headroom)

## Concatenation Order

The observation vector is built by `ObservationBuilder.embed_battle` (`src/embeddings.py`) by concatenating encoder outputs in this order:

1. `ActivePokemonEncoder` (self) — 97
2. `ActivePokemonEncoder` (opponent) — 97
3. `TeamEncoder` (self) — 246
4. `TeamEncoder` (opponent) — 246
5. `MovesEncoder` — 148
6. `OpponentMovesEncoder` — 84
7. `MatchupEncoder` — 25
8. `DamageEncoder` — 40
9. `FieldEncoder` — 35
10. `BeliefEncoder` — 96
11. `ActionMaskEncoder` — 26
12. `MetaEncoder` — 23

## Encoder Breakdown

| Encoder | Size | Description |
|---------|------|-------------|
| `ActivePokemonEncoder` × 2 | 194 | Active mon features (self + opponent) |
| `TeamEncoder` × 2 | 492 | Team features (self + opponent, 6 slots each) |
| `MovesEncoder` | 148 | Our 4 move features (37 × 4) |
| `OpponentMovesEncoder` | 84 | Opponent revealed move features (21 × 4) |
| `MatchupEncoder` | 25 | Switch candidate matchup heuristics (5 × 5) |
| `DamageEncoder` | 40 | Damage + threat heuristics |
| `FieldEncoder` | 35 | Weather, terrain, hazards, screens, rooms |
| `BeliefEncoder` | 96 | Belief features (16 × 6 opponent mons) |
| `ActionMaskEncoder` | 26 | Valid action mask aligned to poke-env |
| `MetaEncoder` | 23 | Turn/game context heuristics |
| **Total** | **1163** | |

## Action Mask (26 actions)

The action mask is produced by `ActionMaskEncoder` (`src/encoders.py`) and is aligned with poke-env's Gen9 Singles action mapping (`poke_env.environment.SinglesEnv.action_to_order`):

- `0..5`: switch (team slots)
- `6..9`: move (move slots)
- `10..13`: move + mega (unused in Gen9 formats but still part of the action space)
- `14..17`: move + z-move (unused in Gen9 formats but still part of the action space)
- `18..21`: move + dynamax (unused in Gen9 formats but still part of the action space)
- `22..25`: move + terastallize

Each mask entry is `1.0` if poke-env accepts the action as legal for the current `battle.valid_orders`, else `0.0`.

## Notes

- Treat `src/encoders.py` and `src/embeddings.py` as the source of truth for exact feature definitions.
- The training policy (`src/models/maskable_lstm_policy.py`) uses the embedded action mask to ensure the agent never samples an illegal action.
