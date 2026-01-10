"""
Observation Builder for Gen 9 Random Battle RL bot.
Coordinates feature extraction using modular encoders from src/encoders.py.
"""

import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional
from gymnasium.spaces import Box
from poke_env.battle import AbstractBattle

from .belief_tracker import BeliefTracker
from .damage_calc import BeliefDamageCalculator
from .encoders import (
    ActivePokemonEncoder, TeamEncoder, MovesEncoder, OpponentMovesEncoder,
    MatchupEncoder, DamageEncoder, FieldEncoder, BeliefEncoder,
    ActionMaskEncoder, MetaEncoder
)

logger = logging.getLogger(__name__)

class ObservationBuilder:
    """
    Constructs the observation vector for the RL agent.
    Delegates to specialized Encoder classes.
    """
    
    # Consistent Constants for external reference (Full Block Sizes)
    ACTIVE_DIM = ActivePokemonEncoder.SIZE
    TEAM_DIM = TeamEncoder.SIZE
    MOVES_DIM = MovesEncoder.SIZE
    OPP_MOVES_DIM = OpponentMovesEncoder.SIZE
    MATCHUP_DIM = MatchupEncoder.SIZE
    DAMAGE_DIM = DamageEncoder.SIZE
    FIELD_DIM = FieldEncoder.SIZE
    BELIEF_DIM = BeliefEncoder.SIZE
    MASK_DIM = ActionMaskEncoder.SIZE
    META_DIM = MetaEncoder.SIZE
    
    # Auxiliary constants
    TEAM_POKEMON_DIM = TeamEncoder.POKEMON_DIM
    
    def __init__(self, pokemon_data: Dict[str, Any], belief_tracker: BeliefTracker):
        """
        Initialize the observation builder.
        
        Args:
            pokemon_data: Dictionary from gen9randombattle.json
            belief_tracker: BeliefTracker instance for opponent predictions
        """
        self.pokemon_data = pokemon_data
        self.belief_tracker = belief_tracker
        
        # Initialize damage calculator
        self.dmg_calc = BeliefDamageCalculator(self.pokemon_data, self.belief_tracker)
        
        # Initialize Encoders
        self.active_encoder = ActivePokemonEncoder()
        self.team_encoder = TeamEncoder()
        self.moves_encoder = MovesEncoder()
        self.opp_moves_encoder = OpponentMovesEncoder()
        self.matchup_encoder = MatchupEncoder()
        # DamageEncoder needs calculator and belief tracker
        self.damage_encoder = DamageEncoder(self.dmg_calc, self.belief_tracker)
        self.field_encoder = FieldEncoder()
        self.belief_encoder = BeliefEncoder(self.belief_tracker)
        self.mask_encoder = ActionMaskEncoder()
        self.meta_encoder = MetaEncoder()
        
        # Calculate total size
        self.observation_size = self._calculate_observation_size()
        
    def _calculate_observation_size(self) -> int:
        """Calculate total observation vector size sum(encoders)."""
        size = 0
        size += self.ACTIVE_DIM * 2     # Our active + opp active
        size += self.TEAM_DIM * 2       # Our team + opp team
        size += self.MOVES_DIM          # Our moves
        size += self.OPP_MOVES_DIM      # Opp known moves
        size += self.MATCHUP_DIM        # Switch matchups
        size += self.DAMAGE_DIM         # Damage matrix
        size += self.FIELD_DIM          # Field state
        size += self.BELIEF_DIM         # Belief state
        size += self.MASK_DIM           # Action mask
        size += self.META_DIM           # Meta context
        return size

    def get_observation_space_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get low and high bounds for observation space."""
        low = np.zeros(self.observation_size, dtype=np.float32)
        high = np.ones(self.observation_size, dtype=np.float32)
        
        # Some values can be negative (e.g. unrevealed -1) or >1 (damage %)
        low[:] = -1.0
        high[:] = 4.0 # Generous upper bound
        
        return low, high
    
    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        """
        Convert battle state to observation vector.
        """
        obs_parts = []
        
        # Helper to strict validate parts
        def add_part(part: np.ndarray, expected_size: int, name: str):
            if part.shape != (expected_size,):
                raise ValueError(f"Encoder {name} size mismatch! Expected {expected_size}, Got {part.shape}")
            if np.isnan(part).any():
                logger.error(f"Encoder {name} produced NaNs!")
                part = np.nan_to_num(part) # Soft fix for NaNs to prevent crash, but log error
            obs_parts.append(part)

        # 1. Active Pokemon
        add_part(self.active_encoder.encode(battle.active_pokemon, is_opponent=False), self.ACTIVE_DIM, "Active(Self)")
        add_part(self.active_encoder.encode(battle.opponent_active_pokemon, is_opponent=True), self.ACTIVE_DIM, "Active(Opp)")
        
        # 2. Teams
        add_part(self.team_encoder.encode(battle.team, battle.active_pokemon, is_opponent=False), self.TEAM_DIM, "Team(Self)")
        add_part(self.team_encoder.encode(battle.opponent_team, battle.opponent_active_pokemon, is_opponent=True), self.TEAM_DIM, "Team(Opp)")
        
        # 3. Moves
        add_part(self.moves_encoder.encode(battle), self.MOVES_DIM, "Moves")
        
        # 4. Opponent Moves
        add_part(self.opp_moves_encoder.encode(battle), self.OPP_MOVES_DIM, "OppMoves")
        
        # 5. Matchups
        add_part(self.matchup_encoder.encode(battle), self.MATCHUP_DIM, "Matchups")
        
        # 6. Damage
        add_part(self.damage_encoder.encode(battle), self.DAMAGE_DIM, "Damage")
        
        # 7. Field
        add_part(self.field_encoder.encode(battle), self.FIELD_DIM, "Field")
        
        # 8. Beliefs
        add_part(self.belief_encoder.encode(battle), self.BELIEF_DIM, "Beliefs")
        
        # 9. Action Mask
        add_part(self.mask_encoder.encode(battle), self.MASK_DIM, "Mask")
        
        # 10. Meta
        add_part(self.meta_encoder.encode(battle), self.META_DIM, "Meta")
        
        # Concatenate
        observation = np.concatenate(obs_parts)
        
        # Final Strict Check
        if len(observation) != self.observation_size:
            raise ValueError(
                f"CRITICAL OBS SIZE MISMATCH: Expected {self.observation_size}, Got {len(observation)}. "
                f"Encoder constants are out of sync with actual output!"
            )
                
        return observation.astype(np.float32)
