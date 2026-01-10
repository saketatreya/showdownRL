"""
GreedPunisher: Pure aggression, punishes setup.
Teaches the agent that setup has opportunity cost.
"""

from poke_env.battle import AbstractBattle

from .base import BaseCurriculumPlayer


class GreedPunisher(BaseCurriculumPlayer):
    """
    Pure aggression. Never uses status/setup.
    
    BEHAVIOR:
    - Always click strongest damaging move
    - Never use status, setup, or protect
    - Trade aggressively
    LESSON: Agent learns that setup has cost.
    EXPLOITABLE BY: Setting up on good opportunities.
    """
    
    def choose_move(self, battle: AbstractBattle):
        active = battle.active_pokemon
        opp = battle.opponent_active_pokemon
        
        # Only use damaging moves
        damaging = [m for m in battle.available_moves 
                    if m.base_power and m.base_power > 0]
        
        if damaging and active and opp:
            best_move = None
            best_score = -1
            
            for move in damaging:
                # 1. Base Power and Effectiveness
                power = move.base_power or 0
                eff = self._get_effectiveness(move, opp)
                
                # 2. STAB (Same Type Attack Bonus)
                stab = 1.5 if move.type in (active.type_1, active.type_2) else 1.0
                
                # 3. Stat Category (Physical vs Special) - simplified check
                # Note: move.category is likely an Enum in poke-env
                cat_str = str(move.category).upper()
                
                if 'PHYSICAL' in cat_str:
                    atk_stat = active.base_stats.get('atk', 100)
                    def_stat = opp.base_stats.get('def', 100)
                elif 'SPECIAL' in cat_str:
                    atk_stat = active.base_stats.get('spa', 100)
                    def_stat = opp.base_stats.get('spd', 100)
                else:
                    atk_stat = 100
                    def_stat = 100
                
                # Simple damage heuristic: (Power * STAB * Eff * Atk / Def)
                # We assume 100 as baseline for stats if missing
                damage_score = (power * stab * eff * atk_stat / def_stat)
                
                if damage_score > best_score:
                    best_score = damage_score
                    best_move = move
                    
            if best_move:
                return self.create_order(best_move)
        
        # Fallback: MaxDamageBot logic (if something fails above) or random
        if damaging:
             return self.create_order(max(damaging, key=lambda m: m.base_power or 0))
        
        # No damaging moves
        if battle.available_moves:
            return self.create_order(battle.available_moves[0])
            
        # Forced switch
        if battle.available_switches:
            return self.create_order(battle.available_switches[0])
        
        return self.choose_random_move(battle)
