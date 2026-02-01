"""
Action handling logic for the Gen 9 Random Battle RL bot.
Handles fallback logic and canonical action-to-order mapping.
"""

import logging
import numpy as np
from typing import Optional, Any
from poke_env.battle import AbstractBattle
from poke_env.environment import SinglesEnv
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
        
        Action space (26 actions) matches poke-env SinglesEnv for Gen9:
        - 0-5: switch
        - 6-9: move
        - 10-13: move + mega (unused in Gen9 formats)
        - 14-17: move + z-move (unused in Gen9 formats)
        - 18-21: move + dynamax (unused in Gen9 formats)
        - 22-25: move + terastallize
        """
        # Input validation with debug logging
        if not isinstance(action, (int, np.integer)):
            logger.debug(f"Action type converted: {type(action).__name__} -> int")
            action = int(action)
        
        if not 0 <= action < 26:
            logger.debug(f"Action clamped: {action} -> [0, 25]")
            action = max(0, min(25, action))

        try:
            return SinglesEnv.action_to_order(np.int64(action), battle, fake=False, strict=True)
        except Exception:
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
