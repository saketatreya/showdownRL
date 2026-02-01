"""
RevengeKiller: Finishes low-HP mons ruthlessly.
Teaches the agent to respect speed tiers and priority.
"""

from poke_env.battle import AbstractBattle

from .base import BaseCurriculumPlayer


class RevengeKiller(BaseCurriculumPlayer):
    """
    Finishes low-HP pokemon ruthlessly.
    
    BEHAVIOR: 
    - If opponent < 40% HP and we're faster: Attack
    - If opponent < 25% HP and we have priority: Use priority
    - Otherwise: Click hardest hit
    LESSON: Agent learns to respect speed tiers and not leave mons low.
    EXPLOITABLE BY: Switching out low-HP mons, playing bulky.
    """
    
    FINISH_THRESHOLD = 0.40
    PRIORITY_THRESHOLD = 0.25
    
    def choose_move(self, battle: AbstractBattle):
        opp = battle.opponent_active_pokemon
        active = battle.active_pokemon
        
        if not opp or not active:
            return self._click_best_move(battle)
        
        opp_hp = opp.current_hp_fraction
        
        # Check for priority finish
        if opp_hp < self.PRIORITY_THRESHOLD:
            priority_move = self._get_best_priority(battle)
            if priority_move:
                return self.create_order(priority_move)
        
        # Check for speed-based finish
        if opp_hp < self.FINISH_THRESHOLD:
            if self._is_faster(active, opp, battle):
                best = self._get_best_move_by_effectiveness(battle)
                if best:
                    return self.create_order(best)
        
        # Default: click hardest hit
        return self._click_best_move(battle)
    
    def _get_best_priority(self, battle: AbstractBattle):
        """Get strongest priority move."""
        priority_moves = []
        for m in battle.available_moves:
            try:
                if getattr(m, 'priority', 0) > 0:
                    priority_moves.append(m)
            except (KeyError, AttributeError):
                continue
        
        if not priority_moves:
            return None
        
        opp = battle.opponent_active_pokemon
        return max(priority_moves, key=lambda m: 
            (m.base_power or 0) * self._get_effectiveness(m, opp))
    
    def _is_faster(self, our_mon, their_mon, battle) -> bool:
        """Speed comparison accounting for boosts."""
        try:
            # Base stats
            our_speed = our_mon.stats.get('spe', 100) if our_mon.stats else 100
            their_speed = their_mon.stats.get('spe', 100) if their_mon.stats else 100
            
            # Helper for boost multiplier
            def get_mult(stage):
                stage = max(-6, min(6, stage))
                if stage >= 0:
                    return (2 + stage) / 2
                else:
                    return 2 / (2 - stage)

            # Apply boosts
            our_boost = our_mon.boosts.get('spe', 0)
            their_boost = their_mon.boosts.get('spe', 0)
            
            our_speed *= get_mult(our_boost)
            their_speed *= get_mult(their_boost)
            
            # Paralysis check
            if our_mon.status and str(our_mon.status).lower() == 'par':
                our_speed *= 0.5
            if their_mon.status and str(their_mon.status).lower() == 'par':
                their_speed *= 0.5

            return our_speed >= their_speed
        except:
            return True
