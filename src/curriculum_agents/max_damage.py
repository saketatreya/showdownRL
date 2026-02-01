"""
MaxDamageBot: Always clicks the hardest hitting move.
Teaches the agent that doing damage is important.
"""

from poke_env.battle import AbstractBattle

from .base import BaseCurriculumPlayer


class MaxDamageBot(BaseCurriculumPlayer):
    """
    Always clicks move with highest (Base Power × Type Effectiveness).
    
    BEHAVIOR: Pure aggression, no switching, no setup.
    LESSON: Agent learns that damage output matters.
    EXPLOITABLE BY: Switching to resists, using faster mons.
    """
    
    def choose_move(self, battle: AbstractBattle):
        # Get all damaging moves with scores
        opp = battle.opponent_active_pokemon
        
        best_move = None
        best_score = -1
        
        for move in battle.available_moves:
            power = move.base_power or 0
            if power == 0:
                continue  # Skip status moves
            
            eff = self._get_effectiveness(move, opp)
            score = power * eff
            
            if score > best_score:
                best_score = score
                best_move = move
        
        # Use best damaging move
        if best_move:
            return self.create_order(best_move)
        
        # No damaging moves - use first available
        if battle.available_moves:
            return self.create_order(battle.available_moves[0])
        
        # Forced switch
        if battle.available_switches:
            return self.create_order(battle.available_switches[0])
        
        return self.choose_random_move(battle)
