"""
Player wrapper for trained RL models.
Allows a trained RecurrentPPO model to play as a poke_env Player.
"""

import numpy as np
from typing import Optional, Dict, Any

from poke_env.player import Player
from poke_env.battle import AbstractBattle
from poke_env.player.battle_order import BattleOrder

from .belief_tracker import BeliefTracker
from .embeddings import ObservationBuilder
from .utils import load_pokemon_data
from .actions import ActionHandler


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
        self.action_handler = ActionHandler()
        
        # LSTM hidden state (for RecurrentPPO)
        self._lstm_states = None
        self._episode_start = True
    
    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        """
        Choose a move using the trained model.
        
        Args:
            battle: Current battle state
            
        Returns:
            BattleOrder to execute
        """
        # Update beliefs
        self._update_beliefs(battle)
        
        # Get observation
        obs = self.obs_builder.embed_battle(battle)
        
        # Get action from model
        action, self._lstm_states = self.model.predict(
            obs,
            state=self._lstm_states,
            episode_start=np.array([self._episode_start]),
            deterministic=self.deterministic
        )
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
        """Convert action index to BattleOrder using centralized handler."""
        return self.action_handler.action_to_order(action, battle)
    
    def _battle_finished_callback(self, battle: AbstractBattle):
        """Called when battle finishes. Reset state."""
        super()._battle_finished_callback(battle)
        self.belief_tracker.reset()
        self._lstm_states = None
        self._episode_start = True


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
