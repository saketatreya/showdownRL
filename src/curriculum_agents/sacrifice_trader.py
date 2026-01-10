"""
SacrificeTrader: Trades 1-for-1 happily.
Teaches the agent to preserve Pokemon, not just maximize damage.
"""

from poke_env.battle import AbstractBattle

from .base import BaseCurriculumPlayer


class SacrificeTrader(BaseCurriculumPlayer):
    """
    Happily trades 1-for-1. Never switches.
    
    BEHAVIOR:
    - Always click strongest move
    - Never switch (even if dying)
    - Pure aggression
    LESSON: Agent learns to preserve mons, not just trade.
    EXPLOITABLE BY: Switching to tank, not taking bad trades.
    """
    
    def choose_move(self, battle: AbstractBattle):
        # Always attack, never switch
        best = self._get_best_move_by_effectiveness(battle)
        if best:
            return self.create_order(best)
        
        # If no damaging moves, use whatever
        if battle.available_moves:
            return self.create_order(battle.available_moves[0])
        
        # Forced to switch (fainted)
        if battle.available_switches:
            return self.create_order(battle.available_switches[0])
        
        return self.choose_random_move(battle)
