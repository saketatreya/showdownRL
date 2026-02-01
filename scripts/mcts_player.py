"""
Monte Carlo Tree Search (MCTS) Inference Player.
Implements a 1-Step Lookahead Search combining PPO Policy Priors with 
Belief-driven heuristic Q-value estimation to select optimal moves at inference time.
"""

import numpy as np
import torch
import math
import asyncio
from typing import Optional, Dict, Any, List, Tuple

from poke_env.player import Player
from poke_env.battle import AbstractBattle
from poke_env.battle.move import Move
from poke_env.battle.pokemon import Pokemon
from poke_env.player.battle_order import BattleOrder, SingleBattleOrder

from src.trained_player import TrainedPlayer
from src.utils import load_pokemon_data
from src.damage_calc import BeliefDamageCalculator
from poke_env.environment import SinglesEnv

class MCTSPlayer(TrainedPlayer):
    """
    Player that uses MCTS logic at inference time.
    Because Pokemon Showdown battles cannot be perfectly branched/forked (Live WebSocket),
    we implement a Depth-1 MCTS (or Expectimax) that leverages:
    1. PPO Actor (Policy Prior probabilities)
    2. BeliefDamageCalculator (Immediate Reward / Transition Engine)
    3. PPO Critic (Value State estimation)
    """
    
    def __init__(
        self,
        model,
        pokemon_data: Optional[Dict[str, Any]] = None,
        pokemon_data_path: str = "gen9randombattle.json",
        c_puct: float = 1.0,
        **kwargs
    ):
        super().__init__(model, pokemon_data, pokemon_data_path, deterministic=True, **kwargs)
        self.c_puct = c_puct
        self.dmg_calc = BeliefDamageCalculator(self.pokemon_data, self.belief_tracker)
        
    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        """
        Choose the best move using shallow MCTS (PUCT algorithm).
        """
        # 1. Update Beliefs
        self._update_beliefs(battle)
        
        # 2. Extract Observation
        obs = self.obs_builder.embed_battle(battle)
        obs_tensor = torch.tensor(obs).unsqueeze(0) # Add batch dim
        
        # 3. Get Policy Priors (P) and State Value (V) from PPO
        with torch.no_grad():
            features = self.model.policy.features_extractor(obs_tensor)
            # Need to handle Recurrent features if lstm is used.
            # Simplifying: take the base policy outputs without recurrent state updates for the simulation
            if hasattr(self.model.policy, "mlp_extractor"):
                latent_pi, latent_vf = self.model.policy.mlp_extractor(features)
                action_logits = self.model.policy.action_net(latent_pi)
                value = self.model.policy.value_net(latent_vf).item()
            else:
                 # Standard extraction for recurrent policies
                 policy_outputs = self.model.policy.get_distribution(obs_tensor)
                 action_logits = policy_outputs.distribution.logits
                 value = 0.0 # Approximation if critic inaccessible directly
                 
            # Convert logits to probabilities via Softmax
            action_probs = torch.softmax(action_logits, dim=-1).squeeze().numpy()
            
        # 4. Filter for Valid Actions
        valid_actions = self._get_valid_actions(battle)
        if not valid_actions:
            return Player.choose_random_singles_move(battle)
            
        # 5. Evaluate Q-Values (Depth 1 Simulation via Damage Calc)
        q_values = {}
        for action_idx in valid_actions:
            order = self._action_to_order(battle, action_idx)
            q_values[action_idx] = self._simulate_action_value(battle, order, value)
            
        # 6. Apply PUCT Formula to select best action
        # PUCT(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(total_visits) / (1 + visit_count)
        # For D=1 without iterative rollouts, total_visits=1, visit_count=0
        best_puct = -float('inf')
        best_action = valid_actions[0]
        
        for action_idx in valid_actions:
            q = q_values[action_idx]
            p = action_probs[action_idx]
            
            puct_score = q + self.c_puct * p
            
            if puct_score > best_puct:
                best_puct = puct_score
                best_action = action_idx
                
        # Update LSTM state for the CHOSEN action (step forward ground truth)
        _, self._lstm_states = self.model.predict(
            obs,
            state=self._lstm_states,
            episode_start=np.array([self._episode_start]),
            deterministic=True
        )
        self._episode_start = False
        
        return self._action_to_order(battle, best_action)
        
    def _get_valid_actions(self, battle: AbstractBattle) -> List[int]:
        """Return valid action indices for poke-env's 26-action Gen9 mapping."""
        valid: List[int] = []
        for action in range(26):
            try:
                SinglesEnv.action_to_order(np.int64(action), battle, fake=False, strict=True)
            except Exception:
                continue
            valid.append(action)
        return valid

    def _simulate_action_value(self, battle: AbstractBattle, order: BattleOrder, base_state_value: float) -> float:
        """
        Estimate the Q-value of taking an action using our forward model (BeliefDamageCalculator).
        Returns a scalar heuristic.
        """
        q_val = base_state_value
        
        # If it's a switch, heuristic is mostly based on type matchups (MatchupEncoder covers this in base_state_value)
        # But we can add a tiny penalty for giving up a free turn, unless Forced.
        if isinstance(order, SingleBattleOrder) and isinstance(order.order, Pokemon):
            if not battle.force_switch:
                q_val -= 0.05
            return q_val
            
        # If it's a move, calculate expected damage
        if isinstance(order, SingleBattleOrder) and isinstance(order.order, Move):
            move = order.order
            
            # Status moves rely heavily on the PPO Prior, we assign 0 heuristic damage
            if move.base_power == 0:
                # E.g., Swords Dance might be great, base_state_value usually captures setup viability
                return q_val
                
            try:
                dmg_result = self.dmg_calc.calculate_move_damage(battle, move, is_our_move=True)
                
                # Convert damage % to Q-value scalar (-1 to 1 theoretical range)
                expected_pct = (dmg_result.min_percent + dmg_result.max_percent) / 2.0
                
                # Cap damage contribution
                dmg_heuristic = min(expected_pct, 1.0) * 0.4 # Max 0.4 added Q
                
                # Bonus for OHKO
                if dmg_result.is_ohko:
                    dmg_heuristic += 0.2
                    
                q_val += dmg_heuristic
            except Exception:
                pass # Fallback to base value
                
        return q_val
