"""
Opponent Pool Manager v3: Simplified Time-Based Curriculum.

Three modes:
1. Baseline: 3 fixed opponents (handled in train.py)
2. Curriculum: Time-based unlock (no performance gates)
   - 0-30%: Random only
   - 30-60%: Random + MaxDamage
   - 60%+: Random + MaxDamage + Heuristic
3. Self-play: Handled in train.py (3 fixed + 30% self-play)
"""

import random
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

from poke_env.player import Player, RandomPlayer, MaxBasePowerPlayer, SimpleHeuristicsPlayer
from poke_env import AccountConfiguration

from ..teams.random_battle_teambuilder import RandomBattleTeambuilder


class FrozenCheckpointPool:
    """Maintains rolling window of frozen checkpoints for Self-Play."""
    
    WINDOW_SIZE = 10
    
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self._checkpoints: List[Path] = []
        self._refresh()
    
    def _refresh(self):
        """Refresh checkpoint list from disk."""
        if not self.checkpoint_dir.exists():
            self._checkpoints = []
            return
        
        all_checkpoints = list(self.checkpoint_dir.glob("model_*.zip"))
        
        def get_iter(p):
            try:
                return int(p.stem.split("_")[1])
            except:
                return 0
        
        all_checkpoints = sorted(all_checkpoints, key=get_iter)
        self._checkpoints = all_checkpoints[-self.WINDOW_SIZE:]
    
    def sample(self) -> Optional[Path]:
        """Sample uniformly from checkpoint pool."""
        self._refresh()
        if not self._checkpoints:
            return None
        return random.choice(self._checkpoints)
    
    def add_snapshot(self, model, iteration: int):
        """Save a new snapshot to the pool."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        save_path = self.checkpoint_dir / f"model_{iteration}.zip"
        model.save(str(save_path))
        self._refresh()
    
    @property
    def is_empty(self) -> bool:
        return len(self._checkpoints) == 0


class OpponentPoolManager:
    """
    Time-Based Curriculum v3.
    
    Simply unlocks opponents based on training progress (no performance gates).
    
    - 0-30%: RandomPlayer only
    - 30-60%: RandomPlayer + MaxDamage  
    - 60%+: RandomPlayer + MaxDamage + Heuristic
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        battle_format: str = "gen9randombattle",
        pokemon_data: Optional[Dict] = None,
    ):
        self.battle_format = battle_format
        self.checkpoint_pool = FrozenCheckpointPool(checkpoint_dir)
        self._rng = random.Random()
        self.pokemon_data = pokemon_data
        self.random_teambuilder = RandomBattleTeambuilder(pokemon_data=pokemon_data)
        
        # Track current phase for logging
        self._last_phase = None

    def sample_opponent(
        self, 
        progress: float,
        win_tracker: Any,
        pokemon_data: Optional[Dict] = None,
    ) -> Tuple[Player, str, str]:
        """
        Sample opponent based on progress (time-based unlock).
        """
        import uuid
        
        # Determine available opponents based on progress
        if progress < 0.30:
            available = ['random_player']
            phase = "Phase 1 (0-30%)"
        elif progress < 0.60:
            available = ['random_player', 'max_damage']
            phase = "Phase 2 (30-60%)"
        else:
            available = ['random_player', 'max_damage', 'heuristic']
            phase = "Phase 3 (60%+)"
        
        # Log phase transitions
        if phase != self._last_phase:
            print(f"\n🎯 [Curriculum] Entering {phase} - {len(available)} opponents available")
            self._last_phase = phase
        
        # Sample uniformly from available opponents
        agent_type = self._rng.choice(available)
        
        # Create the opponent
        rand_suffix = uuid.uuid4().hex[:6]
        
        if agent_type == 'random_player':
            config = AccountConfiguration(f"Rand{rand_suffix}", None)
            opponent = RandomPlayer(
                battle_format=self.battle_format,
                account_configuration=config,
                team=self.random_teambuilder.yield_team()
            )
        elif agent_type == 'max_damage':
            config = AccountConfiguration(f"MaxD{rand_suffix}", None)
            opponent = MaxBasePowerPlayer(
                battle_format=self.battle_format,
                account_configuration=config,
                team=self.random_teambuilder.yield_team()
            )
        else:  # heuristic
            config = AccountConfiguration(f"Heur{rand_suffix}", None)
            opponent = SimpleHeuristicsPlayer(
                battle_format=self.battle_format,
                account_configuration=config,
                team=self.random_teambuilder.yield_team()
            )
        
        win_tracker.set_current_opponent(agent_type)
        return opponent, phase, agent_type

    def create_self_play_opponent(self, pokemon_data: Optional[Dict] = None) -> Optional[Player]:
        """Create a frozen self-play opponent if checkpoints available."""
        checkpoint_path = self.checkpoint_pool.sample()
        
        if not checkpoint_path:
            return None
        
        try:
            from sb3_contrib import RecurrentPPO
            from src.trained_player import TrainedPlayer
            import uuid
            
            model = RecurrentPPO.load(str(checkpoint_path))
            rand_suffix = uuid.uuid4().hex[:6]
            config = AccountConfiguration(f"Frz{rand_suffix}", None)
            
            return TrainedPlayer(
                model=model,
                pokemon_data=pokemon_data,
                battle_format=self.battle_format,
                account_configuration=config,
                deterministic=True,
            )
        except Exception as e:
            print(f"[SelfPlay] Failed to load checkpoint: {e}")
            return None

    def get_stats(self) -> Dict[str, Any]:
        """Return pool statistics."""
        return {
            'frozen_count': len(self.checkpoint_pool._checkpoints),
            'current_phase': self._last_phase,
        }
