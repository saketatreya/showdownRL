"""
PivotSpammer: Uses U-turn/Volt Switch every turn.
Teaches the agent to predict pivots.
"""

from poke_env.battle import AbstractBattle

from .base import BaseCurriculumPlayer


class PivotSpammer(BaseCurriculumPlayer):
    """
    Uses pivot moves (U-turn, Volt Switch) whenever available.
    
    BEHAVIOR:
    - If we have a pivot move: Use it
    - Otherwise: Click strongest move
    LESSON: Agent learns to predict pivots.
    EXPLOITABLE BY: Predicting the pivot, hard reads.
    """
    
    PIVOT_MOVES = {'uturn', 'voltswitch', 'flipturn', 'partingshot', 'teleport'}
    
    def choose_move(self, battle: AbstractBattle):
        # Check for pivot move
        pivot_move = self._get_pivot_move(battle)
        if pivot_move:
            return self.create_order(pivot_move)
        
        # No pivot: click strongest
        return self._click_best_move(battle)
    
    def _get_pivot_move(self, battle: AbstractBattle):
        """Get a pivot move if available and not immune."""
        opp = battle.opponent_active_pokemon
        
        for move in battle.available_moves:
            if move.id.lower() in self.PIVOT_MOVES:
                # Check immunity for damaging pivots
                if move.base_power and move.base_power > 0:
                    eff = self._get_effectiveness(move, opp)
                    if eff > 0:
                        return move
                else:
                    # Non-damaging pivot (Teleport)
                    return move
        return None
