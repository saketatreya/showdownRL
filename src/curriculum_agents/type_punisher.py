"""
TypePunisher: Switches to resist agent's most common move type.
Teaches the agent to mix move types, not spam one.
"""

from typing import Optional
from collections import Counter

from poke_env.battle import AbstractBattle, Pokemon

from .base import BaseCurriculumPlayer


class TypePunisher(BaseCurriculumPlayer):
    """
    Tracks opponent's move types and switches to resist.
    
    BEHAVIOR: Switch to resist most common attack type, then attack.
    LESSON: Agent learns to vary move types.
    EXPLOITABLE BY: Mixing move types, predicting switches.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._move_type_counts: Counter = Counter()
        self._counted_moves: set = set()
    
    def choose_move(self, battle: AbstractBattle):
        # Track what types the opponent (our target) is using
        self._track_opponent_moves(battle)
        
        # Find most common attack type used against us
        most_common_type = self._get_most_common_type()
        
        # If we have data, try to switch to a resist
        if most_common_type and battle.available_switches:
            resist_switch = self._find_resist(battle, most_common_type)
            if resist_switch:
                # Only switch if we're NOT already resisting (multiplier > 0.5 means neutral/weak)
                active = battle.active_pokemon
                if active and self._type_resists(active, most_common_type) > 0.5:
                    return self.create_order(resist_switch)
        
        # Default: click hardest hit
        return self._click_best_move(battle)
    
    def _track_opponent_moves(self, battle: AbstractBattle):
        """Track types of moves opponent uses."""
        opp = battle.opponent_active_pokemon
        if not opp or not opp.moves:
            return
        
        # Track all revealed move types
        for move_id, move in opp.moves.items():
            if move.type and move_id not in self._counted_moves:
                move_type = move.type.name.lower()
                self._move_type_counts[move_type] += 1
                self._counted_moves.add(move_id)
    
    def _get_most_common_type(self) -> Optional[str]:
        """Return most common move type."""
        if not self._move_type_counts:
            return None
        return self._move_type_counts.most_common(1)[0][0]
    
    def _find_resist(self, battle: AbstractBattle, atk_type: str) -> Optional[Pokemon]:
        """Find a switch that resists the given type."""
        for switch in battle.available_switches:
            mult = self._type_resists(switch, atk_type)
            if mult <= 0.5:  # Resist or immune
                return switch
        return None
    
    def _battle_finished_callback(self, battle):
        """Reset tracking for new battle."""
        super()._battle_finished_callback(battle)
        self._move_type_counts.clear()
        self._counted_moves.clear()
