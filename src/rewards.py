"""
Sparse reward evaluation for self-play training.

Uses only terminal rewards (+1 win, -1 loss, 0 otherwise).
The agent learns V(s) through PPO's value function head.
"""

from typing import Dict, Any
from poke_env.battle import AbstractBattle


class SparseRewardEvaluator:
    """
    Simple sparse reward evaluator.
    
    R = +1 for win, -1 for loss, 0 otherwise.
    """
    
    def __init__(self, victory_bonus: float = 1.0, defeat_penalty: float = -1.0):
        """
        Args:
            victory_bonus: Reward for winning
            defeat_penalty: Reward for losing
        """
        self.victory_bonus = victory_bonus
        self.defeat_penalty = defeat_penalty
    
    def reset(self):
        """Reset for new episode (no state to track)."""
        pass
    
    def set_progress(self, progress: float):
        """No-op for compatibility."""
        pass
    
    def calc_reward(self, battle: AbstractBattle, info: Dict[str, Any] = None) -> float:
        """
        Calculate sparse reward.
        
        Args:
            battle: Current battle state
            info: Optional info dict (unused)
            
        Returns:
            +1 for win, -1 for loss, 0 otherwise
        """
        if battle.won:
            return self.victory_bonus
        elif battle.lost:
            return self.defeat_penalty
        else:
            return 0.0
    
    def get_reward_components(self) -> Dict[str, float]:
        """Return empty dict - no components to track."""
        return {}
