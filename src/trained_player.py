"""
Player wrapper for trained RL models.
Allows a trained RecurrentPPO model to play as a poke_env Player.
"""

import numpy as np
from typing import Optional, Dict, Any

from poke_env.player import Player
from poke_env.battle import AbstractBattle
from poke_env.player.battle_order import BattleOrder
from poke_env.environment import SinglesEnv

from .belief_tracker import BeliefTracker
from .embeddings import ObservationBuilder
from .utils import load_pokemon_data


class TrainedPlayer(Player):
    """
    A Player that uses a trained SB3 RecurrentPPO model to make decisions.
    
    Used for:
    - Evaluation against baselines
    - Self-play as opponent
    - Playing against humans
    """
    
    def __init__(
        self,
        model,
        pokemon_data: Optional[Dict[str, Any]] = None,
        pokemon_data_path: str = "gen9randombattle.json",
        deterministic: bool = True,
        **kwargs
    ):
        """
        Initialize the trained player.
        
        Args:
            model: Trained SB3 model (RecurrentPPO)
            pokemon_data: Pre-loaded Pokemon data
            pokemon_data_path: Path to load Pokemon data from
            deterministic: If True, use greedy action selection
            **kwargs: Additional Player arguments
        """
        super().__init__(**kwargs)
        
        self.model = model
        self.deterministic = deterministic
        
        # Load Pokemon data
        if pokemon_data is None:
            self.pokemon_data = load_pokemon_data(pokemon_data_path)
        else:
            self.pokemon_data = pokemon_data
        
        # Initialize belief tracker and observation builder
        self.belief_tracker = BeliefTracker(self.pokemon_data)
        self.obs_builder = ObservationBuilder(self.pokemon_data, self.belief_tracker)
        
        # LSTM hidden state (for RecurrentPPO)
        self._lstm_states = None
        self._episode_start = True
        self._last_battle_tag = None
    
    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        """
        Choose a move using the trained model.
        
        Args:
            battle: Current battle state
            
        Returns:
            BattleOrder to execute
        """
        # If this Player instance is used purely as a policy (e.g. as the opponent in
        # poke-env's SingleAgentWrapper), it will not receive PSClient battle callbacks.
        # Detect new battles and reset episode state manually to avoid cross-battle leakage.
        battle_tag = getattr(battle, "battle_tag", None)
        if battle_tag is not None and battle_tag != self._last_battle_tag:
            try:
                self.belief_tracker.reset()
            except Exception:
                pass
            self._lstm_states = None
            self._episode_start = True
            self._last_battle_tag = battle_tag

        # Update beliefs
        self._update_beliefs(battle)
        
        # Get observation
        obs = self.obs_builder.embed_battle(battle)
        
        # Get action from model
        try:
            action, self._lstm_states = self.model.predict(
                obs,
                state=self._lstm_states,
                episode_start=np.array([self._episode_start]),
                deterministic=self.deterministic,
            )
        except Exception:
            # Never crash callers (training/eval) because a frozen opponent checkpoint is incompatible.
            self._lstm_states = None
            self._episode_start = True
            return Player.choose_random_singles_move(battle)
        self._episode_start = False
        
        # Convert action to int (handles 0-dim numpy arrays from predict)
        action = int(action.item()) if hasattr(action, 'item') else int(action)
        
        # Convert action to BattleOrder
        return self._action_to_order(battle, action)
    
    def _update_beliefs(self, battle: AbstractBattle):
        """Update belief tracker based on battle events."""
        for pokemon in battle.opponent_team.values():
            species = pokemon.species
            
            if pokemon.moves:
                for move_id in pokemon.moves:
                    self.belief_tracker.update(species, observed_move=move_id)
            
            if pokemon.item:
                self.belief_tracker.update(species, observed_item=pokemon.item)
            
            if pokemon.ability:
                self.belief_tracker.update(species, observed_ability=pokemon.ability)
    
    def _action_to_order(self, battle: AbstractBattle, action: int) -> BattleOrder:
        """
        Convert an action index to a BattleOrder using poke-env's native mapping.
        This keeps TrainedPlayer compatible with the environment's 26-action space.
        """
        # Ensure action is in-range for Discrete(26)
        if action < 0 or action >= 26:
            action = max(0, min(25, int(action)))

        try:
            return SinglesEnv.action_to_order(np.int64(action), battle, fake=False, strict=True)
        except Exception:
            # Never crash training/eval because an opponent produced an illegal action.
            return Player.choose_random_singles_move(battle)
    
    def _battle_finished_callback(self, battle: AbstractBattle):
        """Called when battle finishes. Reset state."""
        super()._battle_finished_callback(battle)
        self.belief_tracker.reset()
        self._lstm_states = None
        self._episode_start = True
        self._last_battle_tag = None


def load_trained_player(
    checkpoint_path: str,
    pokemon_data_path: str = "gen9randombattle.json",
    battle_format: str = "gen9randombattle",
    **kwargs
) -> TrainedPlayer:
    """
    Load a trained player from a checkpoint.
    
    Args:
        checkpoint_path: Path to saved model (.zip)
        pokemon_data_path: Path to Pokemon data JSON
        battle_format: Battle format string
        **kwargs: Additional Player arguments
        
    Returns:
        TrainedPlayer ready to battle
    """
    from sb3_contrib import RecurrentPPO
    
    model = RecurrentPPO.load(checkpoint_path)
    pokemon_data = load_pokemon_data(pokemon_data_path)
    
    return TrainedPlayer(
        model=model,
        pokemon_data=pokemon_data,
        battle_format=battle_format,
        **kwargs
    )
