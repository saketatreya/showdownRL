"""
Curriculum agents for training.
Provides deterministic, interpretable opponents for curriculum learning.
"""

from .base import BaseCurriculumPlayer

# Tier 1: Fundamentals
from .max_damage import MaxDamageBot
from .type_punisher import TypePunisher

# Tier 2: Intermediate
from .revenge_killer import RevengeKiller
from .sacrifice_trader import SacrificeTrader
from .hazard_stacker import HazardStacker

# Tier 3: Advanced
from .setup_sweeper import SetupSweeper
from .priority_sniper import PrioritySniper
from .pivot_spammer import PivotSpammer

# Exploiters
from .greed_punisher import GreedPunisher
from .passivity_exploiter import PassivityExploiter

# Pool Manager
from .opponent_pool import OpponentPoolManager, FrozenCheckpointPool

__all__ = [
    'BaseCurriculumPlayer',
    # Tier 1
    'MaxDamageBot',
    'TypePunisher',
    # Tier 2
    'RevengeKiller',
    'SacrificeTrader',
    'HazardStacker',
    # Tier 3
    'SetupSweeper',
    'PrioritySniper',
    'PivotSpammer',
    # Exploiters
    'GreedPunisher',
    'PassivityExploiter',
    # Pool
    'OpponentPoolManager',
    'FrozenCheckpointPool',
]
