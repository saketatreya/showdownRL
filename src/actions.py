"""
Action handling logic for the Gen 9 Random Battle RL bot.
Handles fallback logic and canonical action-to-order mapping.
"""

import logging
import numpy as np
from typing import Optional, Any
from poke_env.battle import AbstractBattle
from poke_env.player.battle_order import BattleOrder, DefaultBattleOrder, SingleBattleOrder

logger = logging.getLogger(__name__)

class ActionHandler:
    """
    Handles action processing and fallback logic.
    Provides canonical mapping from discrete action index to BattleOrder.
    """
    
    def action_to_order(self, action: int, battle: AbstractBattle) -> BattleOrder:
        """
        Convert discrete action index to BattleOrder.
        
        Action space (26 actions):
        - 0-3: Use move 1-4
        - 4-8: Switch to pokemon 1-5
        - 9-12: Use move 1-4 with terastallize
        - 13-17: Switch to pokemon 1-5 (reserved, same as 4-8)
        - 18-25: Reserved (fallback)
        """
        # Input validation with debug logging
        if not isinstance(action, (int, np.integer)):
            logger.debug(f"Action type converted: {type(action).__name__} -> int")
            action = int(action)
        
        if not 0 <= action < 26:
            logger.debug(f"Action clamped: {action} -> [0, 25]")
            action = max(0, min(25, action))
        
        available_moves = battle.available_moves
        available_switches = battle.available_switches
        can_tera = battle.can_tera
        n_moves = len(available_moves)
        n_switches = len(available_switches)
        
        # CRITICAL: If no moves/switches/trapped/recharging -> Default
        if n_moves == 0 and not battle.force_switch:
            return DefaultBattleOrder()
        
        # 1. Moves 0-3
        if action < 4:
            if n_moves > 0:
                move_idx = action % n_moves
                return SingleBattleOrder(available_moves[move_idx])
            elif n_switches > 0:
                return SingleBattleOrder(available_switches[0])
                
        # 2. Switches 4-8
        elif action < 9:
            idx = (action - 4)
            if n_switches > 0:
                switch_idx = idx % n_switches
                return SingleBattleOrder(available_switches[switch_idx])
            elif n_moves > 0:
                return SingleBattleOrder(available_moves[0])
                
        # 3. Tera Moves 9-12
        elif action < 13:
            idx = (action - 9)
            if n_moves > 0 and can_tera:
                 move_idx = idx % n_moves
                 return SingleBattleOrder(available_moves[move_idx], terastallize=True)
            elif n_moves > 0:
                 move_idx = idx % n_moves
                 return SingleBattleOrder(available_moves[move_idx])
            elif n_switches > 0:
                 return SingleBattleOrder(available_switches[0])
                 
        # 4. Reserved/Fallback 13-25
        else:
            return self.get_fallback_order(action, battle)
            
        # Ultimate fallback
        return self.get_fallback_order(action, battle)

    def get_fallback_order(self, original_action: int, battle: AbstractBattle) -> BattleOrder:
        """
        Get a valid fallback order when the target action is invalid/impossible.
        """
        if not battle.available_moves and not battle.available_switches:
             return DefaultBattleOrder()
             
        # Prioritize moves (safest)
        if battle.available_moves:
             return SingleBattleOrder(battle.available_moves[0])
             
        # Then switches
        if battle.available_switches:
             return SingleBattleOrder(battle.available_switches[0])
             
        return DefaultBattleOrder()
