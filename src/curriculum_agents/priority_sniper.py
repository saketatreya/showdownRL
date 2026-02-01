"""
PrioritySniper: Uses priority to finish low-HP targets.
Teaches the agent about priority brackets and HP thresholds.
"""

from poke_env.battle import AbstractBattle

from .base import BaseCurriculumPlayer


class PrioritySniper(BaseCurriculumPlayer):
    """
    Uses priority moves to finish low-HP targets.
    
    BEHAVIOR:
    - If opponent < 30% HP and we have priority: Use priority
    - If opponent < 30% HP and we're faster: Click strongest
    - Otherwise: Click strongest move
    LESSON: Agent learns priority brackets.
    EXPLOITABLE BY: Not letting mons get low, bulky pivots.
    """
    
    SNIPE_THRESHOLD = 0.30
    
    def choose_move(self, battle: AbstractBattle):
        opp = battle.opponent_active_pokemon
        
        if not opp:
            return self._click_best_move(battle)
        
        opp_hp = opp.current_hp_fraction
        
        # Snipe mode: opponent is low
        if opp_hp < self.SNIPE_THRESHOLD:
            # Priority first
            priority_move = self._get_best_priority(battle, opp)
            if priority_move:
                return self.create_order(priority_move)
        
        # Default: strongest move
        return self._click_best_move(battle)
    
    def _get_best_priority(self, battle: AbstractBattle, opp):
        """Get strongest non-immune priority move."""
        priority_moves = []
        for move in battle.available_moves:
            try:
                priority = getattr(move, 'priority', 0)
                if priority > 0:
                    eff = self._get_effectiveness(move, opp)
                    if eff > 0:  # Not immune
                        priority_moves.append((move, (move.base_power or 0) * eff))
            except (KeyError, AttributeError):
                continue
        
        if not priority_moves:
            return None
        
        return max(priority_moves, key=lambda x: x[1])[0]
