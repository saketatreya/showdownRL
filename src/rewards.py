"""
Reward evaluation logic for the Gen 9 Random Battle RL bot.
Decoupled from the main environment class for modularity.
"""

from typing import Dict, Optional, Any
from copy import deepcopy
import logging
import os

import numpy as np
from poke_env.battle import AbstractBattle
from poke_env.battle.side_condition import SideCondition

from .utils import get_type_effectiveness
from .config import REWARD_CURRICULUM

class DeltaRewardEvaluator:
    """
    Calculates rewards using a delta-based state value function.
    
    R_t = V(S_t) - V(S_{t-1})
    
    where V(S) is a weighted sum of state components (HP, Fainted, Matchups, etc.)
    """
    
    def __init__(self, reward_config: Dict[str, float]):
        """
        Args:
            reward_config: Dictionary of weights from config.py
        """
        self.reward_config = reward_config.copy()
        
        self._prev_state_value: Optional[Dict[str, float]] = None
        self._last_reward_components = {
            'hp': 0.0,
            'fainted': 0.0,
            'matchup': 0.0
        }
        # Momentum tracking to prevent boost farming
        self._turns_without_damage: int = 0
        self._prev_opponent_hp: float = 1.0
        self._current_phase: str = "early" # Track current phase for logging
        self._last_active_pokemon_name: Optional[str] = None
        
        # Sparsification State
        self.sparse_mode: bool = False
        self.sparse_blend: float = 0.0
    
    def set_progress(self, progress: float):
        """
        Update reward weights based on training progress using the curriculum.
        
        Args:
            progress: Training progress from 0.0 to 1.0
        """
        # Determine phase
        new_phase = 'early'
        if progress >= 0.5:
            new_phase = 'late'
        elif progress >= 0.2:
            new_phase = 'mid'
            
        # Only update if phase changed or first run
        if new_phase != self._current_phase:
            logging.info(f"Step {progress:.2%}: Switching reward curriculum to {new_phase.upper()} phase")
            self._current_phase = new_phase
            
        # Apply weights for the current phase
        phase_weights = deepcopy(REWARD_CURRICULUM[new_phase])  # Prevent shared mutation
        
        # Update internal config
        for k, v in phase_weights.items():
            if k in self.reward_config:
                self.reward_config[k] = v
        
        # Dense mode only (sparsity ablation removed)
        self.sparse_mode = False
        self.sparse_blend = 0.0
    
    def reset(self):
        """Reset internal state for a new episode."""
        self._prev_state_value = None
        self._last_reward_components = {
            'hp': 0.0,
            'fainted': 0.0,
            'matchup': 0.0
        }
        self._turns_without_damage = 0
        self._prev_opponent_hp = 1.0
        self._last_active_pokemon_name = None
        
    def get_reward_components(self) -> Dict[str, float]:
        """Return the cumulative components of reward for metrics."""
        return self._last_reward_components
    
    def calc_reward(self, battle: AbstractBattle) -> float:
        """
        Calculate reward for the current step.
        """
        # Calculate current state value
        current_value = self._calculate_state_value(battle)
        
        # Get previous state value (or initialize if first step)
        if self._prev_state_value is None:
            self._prev_state_value = current_value.copy()
            # For first step, delta is effectively 0 (or we could reward initial state? No.)
            # But we must return the terminal value if match ends immediately? Unlikely.
            prev_value = current_value 
        else:
            prev_value = self._prev_state_value
        
        # Calculate delta reward per component to support thresholding
        delta_reward = 0.0
        hp_threshold = self.reward_config.get('hp_threshold', 0.0)
        
        # Iterate over all value components
        for key, val in current_value.items():
            if key == 'terminal': continue # Terminal handled separately
            
            prev_v = prev_value.get(key, 0.0)
            diff = val - prev_v
            
            # Apply HP Threshold (Sparsity S1)
            if key == 'hp' and hp_threshold > 0:
                # If change is statistically insignificant, ignore it (Reward=0)
                # But we still update the state in _prev_state_value (below) so we don't 'accumulate' the error
                # This effectively means small damage is "free/ignored"
                if abs(diff) < hp_threshold:
                    diff = 0.0
            
            delta_reward += diff
            
            # Update metrics
            if key in self._last_reward_components:
                self._last_reward_components[key] += diff
        
        # Terminal reward (always added fully)
        reward = delta_reward
        reward += current_value['terminal'] # Start with terminal
        
        # Continuous bonuses
        continuous_bonus = self._calc_continuous_bonus(battle)
        reward += continuous_bonus
        
        # Apply costs
        reward -= self.reward_config['step_cost']
        reward -= self._calc_switch_tax(battle)
        reward += self._calc_momentum_penalty(battle)
        
        # Store current value for next step
        self._prev_state_value = current_value.copy()
        
        # Update tracking state for next step
        self._update_tracking_state(battle)
        
        return float(reward)
    
    def _update_tracking_state(self, battle: AbstractBattle):
        """Update internal tracking state after reward calculation."""
        # 1. Update active pokemon
        if battle.active_pokemon:
            self._last_active_pokemon_name = battle.active_pokemon.species

        # 2. Update momentum (damage) tracking
        current_opp_hp = sum(p.current_hp_fraction or 0 for p in battle.opponent_team.values())
        
        # FIX: Loophole closed. Required > 5% damage to reset turns_without_damage
        dealt_damage = current_opp_hp < self._prev_opponent_hp - 0.05
        
        if dealt_damage:
            self._turns_without_damage = 0
        else:
            self._turns_without_damage += 1
            
        self._prev_opponent_hp = current_opp_hp

    def _calc_switch_tax(self, battle: AbstractBattle) -> float:
        """
        Switch tax penalty.
        """
        current_active = battle.active_pokemon
        tax = 0.0
        
        # Check if we have a previous active pokemon recorded
        if self._last_active_pokemon_name is not None:
            # If active changed (we switched)
            if current_active and current_active.species != self._last_active_pokemon_name:
                # Iterate to find prev mon by species
                prev_mon = None
                for mon in battle.team.values():
                    if mon.species == self._last_active_pokemon_name:
                        prev_mon = mon
                        break
                
                # Only penalize VOLUNTARY switches (prev mon still alive)
                if prev_mon and not prev_mon.fainted:
                    tax = self.reward_config.get('switch_tax', 0.5)
            
        return tax

    def _calc_momentum_penalty(self, battle: AbstractBattle) -> float:
        """Penalize turns spent not dealing damage to discourage infinite setup."""
        current_opp_hp = sum(p.current_hp_fraction or 0 for p in battle.opponent_team.values())
        
        # Preview the update logic (read-only)
        turns_without_damage = self._turns_without_damage
        
        # FIX: Threshold 0.05 (5%)
        dealt_damage = current_opp_hp < self._prev_opponent_hp - 0.05
        
        if dealt_damage:
            turns_without_damage = 0
        else:
            turns_without_damage += 1
        
        # Apply penalty after grace period
        grace = self.reward_config.get('momentum_grace_turns', 2)
        if turns_without_damage > grace:
            penalty_per_turn = self.reward_config.get('momentum_penalty', 0.1)
            # FIX: Cap scaling to prevent value function collapse on long stalls
            # Max penalty = 0.1 * 10 = -1.0 per turn. Sufficient to deter, but not -400.0.
            scaling = min(10, turns_without_damage - grace)
            return -penalty_per_turn * scaling
        
        return 0.0

    def _calc_simple_matchup(self, our_mon, opp_mon) -> float:
        """Quick matchup score: positive = we have advantage."""
        if not our_mon or not opp_mon:
            return 0.0
        
        # Our offensive potential
        our_types = []
        if our_mon.type_1:
            our_types.append(our_mon.type_1.name.lower())
        if our_mon.type_2:
            our_types.append(our_mon.type_2.name.lower())
        
        # Their defensive profile
        def_type1 = opp_mon.type_1.name.lower() if opp_mon.type_1 else None
        def_type2 = opp_mon.type_2.name.lower() if opp_mon.type_2 else None
        
        our_best_eff = max(
            [get_type_effectiveness(t, def_type1, def_type2) for t in our_types] or [1.0]
        )
        
        # Their offensive potential against us
        their_types = []
        if opp_mon.type_1:
            their_types.append(opp_mon.type_1.name.lower())
        if opp_mon.type_2:
            their_types.append(opp_mon.type_2.name.lower())
        
        our_def_type1 = our_mon.type_1.name.lower() if our_mon.type_1 else None
        our_def_type2 = our_mon.type_2.name.lower() if our_mon.type_2 else None
        
        their_best_eff = max(
            [get_type_effectiveness(t, our_def_type1, our_def_type2) for t in their_types] or [1.0]
        )
        
        # Return normalized score: positive = we win matchup
        return (our_best_eff - their_best_eff) / 2.0
    
    def _calculate_state_value(self, battle: AbstractBattle) -> Dict[str, float]:
        """Calculate the total state value V(S)."""
        values = {}
        values['hp'] = self._calc_hp_value(battle) * self.reward_config['w_hp']
        values['fainted'] = self._calc_fainted_value(battle) * self.reward_config['w_fainted']
        values['matchup'] = self._calc_matchup_value(battle) * self.reward_config['w_matchup']
        values['speed'] = self._calc_speed_value(battle) * self.reward_config['w_speed']
        values['status'] = self._calc_status_value(battle) * self.reward_config['w_status']
        values['boosts'] = self._calc_boosts_value(battle) * self.reward_config['w_boosts']
        # Accuracy removed (double-counting)
        values['hazards_opp'] = self._calc_hazards_value(battle) * self.reward_config['w_hazards_up']
        values['hazards_our'] = self._calc_our_hazards_value(battle) * self.reward_config.get('w_hazards_our', 0.2)
        values['terminal'] = self._calc_terminal_value(battle)
        return values
    
    # ... Helper calculation methods ...
    
    def _calc_continuous_bonus(self, battle: AbstractBattle) -> float:
        """Calculate continuous bonuses (additive, not delta)."""
        bonus = 0.0
        if battle.active_pokemon and battle.active_pokemon.boosts:
            boost_sum = sum(max(0, min(6, v)) for v in battle.active_pokemon.boosts.values())
            capped_boost = min(3, boost_sum)
            bonus += capped_boost * self.reward_config['continuous_boost_bonus']
        return bonus
    
    def _calc_hp_value(self, battle: AbstractBattle) -> float:
        """
        Calculate normalized HP difference.
        Assumes a 6-pokemon team format. Unrevealed pokemon are assumed to be at 100% HP.
        """
        TEAM_SIZE = 6
        
        # Calculate Our HP
        our_known_hp = sum(p.current_hp_fraction or 0 for p in battle.team.values())
        our_unknown_count = max(0, TEAM_SIZE - len(battle.team))
        our_total_hp = our_known_hp + (our_unknown_count * 1.0)
        our_normalized = our_total_hp / TEAM_SIZE
        
        # Calculate Opponent HP
        opp_known_hp = sum(p.current_hp_fraction or 0 for p in battle.opponent_team.values())
        opp_unknown_count = max(0, TEAM_SIZE - len(battle.opponent_team))
        opp_total_hp = opp_known_hp + (opp_unknown_count * 1.0)
        opp_normalized = opp_total_hp / TEAM_SIZE
        
        return our_normalized - opp_normalized
    
    def _calc_fainted_value(self, battle: AbstractBattle) -> float:
        """
        Calculate fainted difference (Negative is bad).
        """
        TEAM_SIZE = 6
        
        our_fainted = sum(1 for p in battle.team.values() if p.fainted)
        their_fainted = sum(1 for p in battle.opponent_team.values() if p.fainted)
        
        # We want to minimize our fainted (negative) and maximize theirs (positive)
        # Normalized to [0, 1] range per side
        return (their_fainted / TEAM_SIZE) - (our_fainted / TEAM_SIZE)
    
    def _calc_matchup_value(self, battle: AbstractBattle) -> float:
        our_pokemon = battle.active_pokemon
        their_pokemon = battle.opponent_active_pokemon
        if our_pokemon is None or their_pokemon is None: return 0.0
        
        their_types = [t.name.lower() for t in their_pokemon.types if t]
        our_types = [t.name.lower() for t in our_pokemon.types if t]
        
        # Use actual available moves for outgoing effectiveness (more accurate)
        max_outgoing = 0.25
        if hasattr(battle, 'available_moves') and battle.available_moves:
            for move in battle.available_moves:
                if move.type:
                    move_type = move.type.name.lower()
                    eff = 1.0
                    for def_type in their_types:
                        eff *= get_type_effectiveness(move_type, def_type)
                    max_outgoing = max(max_outgoing, eff)
        else:
            # Fallback to STAB types
            for atk_type in our_types:
                eff = 1.0
                for def_type in their_types:
                    eff *= get_type_effectiveness(atk_type, def_type)
                max_outgoing = max(max_outgoing, eff)
            
        max_incoming = 0.25
        for atk_type in their_types:
            eff = 1.0
            for def_type in our_types:
                eff *= get_type_effectiveness(atk_type, def_type)
            max_incoming = max(max_incoming, eff)
            
        # CONTINUOUS SCORING: Use log2 for smoother gradients
        # log2(4) = 2, log2(2) = 1, log2(1) = 0, log2(0.5) = -1, log2(0.25) = -2
        import math
        
        # Offense: how well we hit them (log2 scale, clamped to [-2, 2])
        if max_outgoing > 0:
            offense_score = max(-2.0, min(2.0, math.log2(max_outgoing)))
        else:
            offense_score = -2.0
        
        # Defense: how well we resist them (inverse - lower incoming is better)
        if max_incoming > 0:
            # Negate because low incoming is good
            defense_score = max(-2.0, min(2.0, -math.log2(max_incoming)))
        else:
            defense_score = 2.0  # Immune is best defense
        
        return offense_score + defense_score
    
    def _calc_speed_value(self, battle: AbstractBattle) -> float:
        me = battle.active_pokemon
        opp = battle.opponent_active_pokemon
        if not me or not opp: return 0.0
        
        # FIX: Granular boost multipliers
        boost_multipliers = {
            -6: 0.25, -5: 0.29, -4: 0.33, -3: 0.40, -2: 0.50, -1: 0.67,
            0: 1.0,
            1: 1.5, 2: 2.0, 3: 2.5, 4: 3.0, 5: 3.5, 6: 4.0
        }
        
        my_boost = me.boosts.get('spe', 0)
        opp_boost = opp.boosts.get('spe', 0)
        
        my_mult = boost_multipliers.get(my_boost, 1.0)
        opp_mult = boost_multipliers.get(opp_boost, 1.0)
        
        # Status checks (Paralysis = 0.5x speed)
        if me.status and str(me.status).lower() == 'par': 
            my_mult *= 0.5
        if opp.status and str(opp.status).lower() == 'par':
            opp_mult *= 0.5
            
        my_spe = me.base_stats['spe'] * my_mult
        opp_spe = opp.base_stats['spe'] * opp_mult
        
        # Normalized difference [-1, 1]
        max_spe = max(my_spe, opp_spe)
        if max_spe == 0: return 0.0
        
        return (my_spe - opp_spe) / max_spe

    def _calc_hazards_value(self, battle: AbstractBattle) -> float:
        """Calculate hazard value scaled by expected damage to remaining opponent team."""
        side = battle.opponent_side_conditions
        
        # FIX: Check using Enum
        has_sr = SideCondition.STEALTH_ROCK in side
        has_spikes = SideCondition.SPIKES in side
        has_tspikes = SideCondition.TOXIC_SPIKES in side
        has_web = SideCondition.STICKY_WEB in side
        
        if not (has_sr or has_spikes or has_tspikes or has_web):
            return 0.0
        
        score = 0.0
        
        # Calculate expected Stealth Rock damage based on remaining opponent types
        if has_sr:
            for pokemon in battle.opponent_team.values():
                if pokemon.fainted:
                    continue
                sr_mult = self._get_sr_multiplier(pokemon)
                score += sr_mult * 0.125  # 12.5% base damage * type multiplier
        
        # Spikes
        if has_spikes:
            spikes_layers = side.get(SideCondition.SPIKES, 1)
            # poke-env stores count as value for Spikes enum? need to verify
            # Just assume 1 if present for safety if type check fails, but SideCondition mapping usually dict {Enum: int}
            
            for pokemon in battle.opponent_team.values():
                if pokemon.fainted:
                    continue
                # Check if grounded
                is_flying = pokemon.type_1 and pokemon.type_1.name.lower() == 'flying'
                is_flying = is_flying or (pokemon.type_2 and pokemon.type_2.name.lower() == 'flying')
                if not is_flying:
                    score += 0.0625 * spikes_layers  # 6.25% per layer
        
        if has_tspikes: score += 0.3
        if has_web: score += 0.5
        
        return min(3.0, score)
    
    def _get_sr_multiplier(self, pokemon) -> float:
        """Get Stealth Rock damage multiplier based on types."""
        multiplier = 1.0
        rock_type = 'rock'
        
        if pokemon.type_1:
            type1 = pokemon.type_1.name.lower()
            multiplier *= get_type_effectiveness(rock_type, type1)
        if pokemon.type_2:
            type2 = pokemon.type_2.name.lower()
            multiplier *= get_type_effectiveness(rock_type, type2)
        
        return multiplier

    def _calc_our_hazards_value(self, battle: AbstractBattle) -> float:
        """
        Calculate NEGATIVE value for hazards on OUR side.
        """
        side = battle.side_conditions
        score = 0.0
        
        # FIX: Enum usage
        has_sr = SideCondition.STEALTH_ROCK in side
        
        if has_sr:
            for pokemon in battle.team.values():
                if pokemon.fainted:
                    continue
                sr_mult = self._get_sr_multiplier(pokemon)
                score += sr_mult * 0.125
        
        # Spikes
        spikes_count = side.get(SideCondition.SPIKES, 0)
        
        if spikes_count > 0:
            for pokemon in battle.team.values():
                if pokemon.fainted:
                    continue
                is_flying = pokemon.type_1 and pokemon.type_1.name.lower() == 'flying'
                is_flying = is_flying or (pokemon.type_2 and pokemon.type_2.name.lower() == 'flying')
                if not is_flying:
                    score += 0.0625 * spikes_count
        
        if SideCondition.TOXIC_SPIKES in side: score += 0.3
        if SideCondition.STICKY_WEB in side: score += 0.5
        
        # Return NEGATIVE (hazards on our side are BAD)
        return -min(3.0, score)

    def _calc_status_value(self, battle: AbstractBattle) -> float:
        # FIX: Dynamic team size
        our_team_size = max(1, len(battle.team))
        opp_team_size = max(1, len(battle.opponent_team))
        
        our_statused = sum(1 for p in battle.team.values() if p.status)
        their_statused = sum(1 for p in battle.opponent_team.values() if p.status)
        return (their_statused / opp_team_size) - (our_statused / our_team_size)
    
    def _calc_boosts_value(self, battle: AbstractBattle) -> float:
        def sum_boosts(pokemon) -> float:
            if pokemon is None or not pokemon.boosts: return 0.0
            # FIX: Exclude accuracy/evasion from general boost sum if desired?
            # Report said accuracy double counts. We removed _calc_accuracy_value so now it's single counted here.
            # This is correct.
            return sum(max(-2, min(2, v)) for v in pokemon.boosts.values())
        return (sum_boosts(battle.active_pokemon) - sum_boosts(battle.opponent_active_pokemon)) / 14.0
    
    def _calc_terminal_value(self, battle: AbstractBattle) -> float:
        if battle.won: return self.reward_config['victory_bonus']
        elif battle.lost: return self.reward_config['defeat_penalty']
        return 0.0

class HybridRewardEvaluator(DeltaRewardEvaluator):
    """
    Hybrid Evaluator that uses both State Deltas and Action Events.
    Fixes U-turn/Failure blindness by using 'info' signals.
    """
    def calc_reward(self, battle: AbstractBattle, info: Dict[str, Any] = None) -> float:
        # 1. Pure Sparse Check (Optimization)
        if self.sparse_mode:
            self._update_tracking_state(battle) # Keep trackers correctly updated even if not used
            if battle.finished:
                return 1.0 if battle.won else -1.0
            return 0.0
            
        # 2. State Value Delta
        current_state_values = self._calculate_state_value(battle)
        current_total = sum(current_state_values.values())
        
        if self._prev_state_value is None:
            self._prev_state_value = current_state_values
            # Update tracking state for first step
            self._update_tracking_state(battle)
            return 0.0
            
        # Calculate delta from previous totals
        prev_total = sum(self._prev_state_value.values())
        state_delta = current_total - prev_total
        
        # Update metrics (tracking cumulative change)
        self._last_reward_components['hp'] += (current_state_values['hp'] - self._prev_state_value.get('hp', 0))
        self._last_reward_components['fainted'] += (current_state_values['fainted'] - self._prev_state_value.get('fainted', 0))
        self._last_reward_components['matchup'] += (current_state_values['matchup'] - self._prev_state_value.get('matchup', 0))
        
        # Store current state as dict for next step (Fixes Type Inconsistency)
        self._prev_state_value = current_state_values
        
        # 3. Event-Based Rewards
        event_reward = 0.0
        
        if info:
            switch_type = info.get('event_switch_type')
            move_failed = info.get('event_move_failed', False)
            
            # Switch Logic
            if switch_type == 'manual':
                event_reward -= self.reward_config['switch_tax']
            
            # Failed Move Logic
            if move_failed:
                event_reward -= self.reward_config['move_fail_penalty']
                
            # Attack Bonus REMOVED per Deep Dive Analysis
        
        # 4. Momentum & Step Cost
        # FIX: Calculate momentum penalty BEFORE updating tracking state
        momentum_penalty = self._calc_momentum_penalty(battle)
        self._update_tracking_state(battle)
        
        event_reward += momentum_penalty
        event_reward -= self.reward_config['step_cost']
        
        dense_reward = state_delta + event_reward
        
        # 5. Apply Sparsification Blending
        if self.sparse_blend > 0:
            sparse_component = 0.0
            if battle.finished:
                sparse_component = 1.0 if battle.won else -1.0
            
            # Blend: (1 - alpha) * dense + alpha * sparse
            return (1.0 - self.sparse_blend) * dense_reward + self.sparse_blend * sparse_component
            
        return dense_reward


class PotentialBasedRewardEvaluator:
    """
    Reward shaping using learned potential function Φ(s) = P(win|state).
    
    R' = R_terminal + γ * Φ(s') - Φ(s)
    
    This is theoretically sound (Ng et al. 1999) - preserves optimal policy
    while providing dense learning signal based on actual game outcomes.
    """
    
    def __init__(
        self, 
        win_predictor_path: str = "data/win_predictor.pt",
        gamma: float = 0.99,
        terminal_scale: float = 1.0
    ):
        """
        Args:
            win_predictor_path: Path to trained WinPredictor model
            gamma: Discount factor for potential shaping
            terminal_scale: Scale for terminal rewards
        """
        self.gamma = gamma
        self.terminal_scale = terminal_scale
        self._prev_potential: Optional[float] = None
        self._turn_count: int = 0
        
        # Load win predictor
        self.win_predictor = None
        self.win_predictor_path = win_predictor_path
        
        if os.path.exists(win_predictor_path):
            try:
                from .win_predictor import WinPredictor
                self.win_predictor = WinPredictor.load(win_predictor_path)
                logging.info(f"[PotentialBased] Loaded Φ(s) from {win_predictor_path}")
            except Exception as e:
                logging.warning(f"[PotentialBased] Failed to load win predictor: {e}")
                self.win_predictor = None
        else:
            logging.warning(f"[PotentialBased] No win predictor at {win_predictor_path}, using fallback")
    
    def reset(self):
        """Reset internal state for a new episode."""
        self._prev_potential = None
        self._turn_count = 0
    
    def calc_reward(self, battle: AbstractBattle, features: Optional[np.ndarray] = None) -> float:
        """
        Calculate reward using potential-based shaping.
        
        R' = R_terminal + γ * Φ(s') - Φ(s)
        
        Args:
            battle: Current battle state
            features: Optional pre-computed feature vector (128-dim)
            
        Returns:
            Shaped reward
        """
        self._turn_count += 1
        
        # Terminal reward
        if battle.finished:
            terminal_reward = self.terminal_scale * (1.0 if battle.won else -1.0)
            self.reset()
            return terminal_reward
        
        # If no win predictor, fall back to sparse only
        if self.win_predictor is None:
            return 0.0
        
        # Calculate potential Φ(s') = P(win | current_state)
        if features is None:
            # Extract features from battle state
            features = self._extract_features(battle)
        
        current_potential = self.win_predictor.predict(features)
        
        # First step: no shaping (no previous state)
        if self._prev_potential is None:
            self._prev_potential = current_potential
            return 0.0
        
        # Potential-based shaping: γΦ(s') - Φ(s)
        shaping = self.gamma * current_potential - self._prev_potential
        self._prev_potential = current_potential
        
        return shaping
    
    def _extract_features(self, battle: AbstractBattle) -> np.ndarray:
        """
        Extract features from live battle matching ReplayParser output (744 dims).
        
        Layout matches replay_parser.py:
        - [0:97]     P1 active pokemon
        - [97:194]   P2 active pokemon
        - [194:440]  P1 team (6 x 41)
        - [440:686]  P2 team (6 x 41)
        - [686:721]  Field state (35)
        - [721:744]  Meta context (23)
        """
        ACTIVE_DIM = 97
        TEAM_POKEMON_DIM = 41
        TEAM_DIM = TEAM_POKEMON_DIM * 6
        FIELD_DIM = 35
        META_DIM = 23
        FEATURE_DIM = 744
        
        TYPE_TO_IDX = {
            'Normal': 0, 'Fire': 1, 'Water': 2, 'Electric': 3, 'Grass': 4,
            'Ice': 5, 'Fighting': 6, 'Poison': 7, 'Ground': 8, 'Flying': 9,
            'Psychic': 10, 'Bug': 11, 'Rock': 12, 'Ghost': 13, 'Dragon': 14,
            'Dark': 15, 'Steel': 16, 'Fairy': 17
        }
        
        STATUS_TO_IDX = {'brn': 0, 'par': 1, 'slp': 2, 'frz': 3, 'psn': 4, 'tox': 5}
        
        features = np.zeros(FEATURE_DIM, dtype=np.float32)
        offset = 0
        
        # Encode active pokemon (97 dims each)
        def encode_active(pokemon, base_offset):
            if pokemon is None:
                return
            
            # HP
            features[base_offset] = pokemon.current_hp_fraction or 0.0
            
            # Types (18 + 18)
            types_list = list(pokemon.types) if pokemon.types else []
            if len(types_list) > 0 and types_list[0]:
                t_name = types_list[0].name if hasattr(types_list[0], 'name') else str(types_list[0])
                if t_name in TYPE_TO_IDX:
                    features[base_offset + 1 + TYPE_TO_IDX[t_name]] = 1.0
            if len(types_list) > 1 and types_list[1]:
                t_name = types_list[1].name if hasattr(types_list[1], 'name') else str(types_list[1])
                if t_name in TYPE_TO_IDX:
                    features[base_offset + 19 + TYPE_TO_IDX[t_name]] = 1.0
            
            # Level as stat proxy
            features[base_offset + 37] = (pokemon.level or 100) / 100.0
            
            # Boosts (7)
            if pokemon.boosts:
                boost_order = ['atk', 'def', 'spa', 'spd', 'spe', 'accuracy', 'evasion']
                for i, stat in enumerate(boost_order):
                    features[base_offset + 43 + i] = pokemon.boosts.get(stat, 0) / 6.0
            
            # Status (7)
            if pokemon.status:
                status_name = pokemon.status.name if hasattr(pokemon.status, 'name') else str(pokemon.status)
                status_name = status_name.lower()[:3]
                if status_name in STATUS_TO_IDX:
                    features[base_offset + 50 + STATUS_TO_IDX[status_name]] = 1.0
            else:
                features[base_offset + 56] = 1.0  # Healthy
            
            # Fainted
            features[base_offset + 57] = 1.0 if pokemon.fainted else 0.0
            
            # Tera
            features[base_offset + 58] = 1.0 if pokemon.terastallized else 0.0
            if pokemon.tera_type:
                tera_name = pokemon.tera_type.name if hasattr(pokemon.tera_type, 'name') else str(pokemon.tera_type)
                if tera_name in TYPE_TO_IDX:
                    features[base_offset + 59 + TYPE_TO_IDX[tera_name]] = 1.0
        
        # P1 active (our pokemon)
        encode_active(battle.active_pokemon, offset)
        offset += ACTIVE_DIM
        
        # P2 active (opponent)
        encode_active(battle.opponent_active_pokemon, offset)
        offset += ACTIVE_DIM
        
        # Encode team (41 dims per pokemon, 6 pokemon)
        def encode_team(team, base_offset):
            team_list = list(team.values())[:6]
            for i, pokemon in enumerate(team_list):
                poke_offset = base_offset + i * TEAM_POKEMON_DIM
                if pokemon:
                    features[poke_offset] = pokemon.current_hp_fraction or 0.0
                    
                    types_list = list(pokemon.types) if pokemon.types else []
                    if len(types_list) > 0 and types_list[0]:
                        t_name = types_list[0].name if hasattr(types_list[0], 'name') else str(types_list[0])
                        if t_name in TYPE_TO_IDX:
                            features[poke_offset + 1 + TYPE_TO_IDX[t_name]] = 1.0
                    if len(types_list) > 1 and types_list[1]:
                        t_name = types_list[1].name if hasattr(types_list[1], 'name') else str(types_list[1])
                        if t_name in TYPE_TO_IDX:
                            features[poke_offset + 19 + TYPE_TO_IDX[t_name]] = 1.0
                    
                    features[poke_offset + 37] = 1.0 if pokemon.status else 0.0
                    features[poke_offset + 38] = 1.0 if pokemon.fainted else 0.0
                    features[poke_offset + 39] = 1.0 if pokemon == battle.active_pokemon else 0.0
                    features[poke_offset + 40] = 1.0  # Revealed
        
        # P1 team
        encode_team(battle.team, offset)
        offset += TEAM_DIM
        
        # P2 team
        encode_team(battle.opponent_team, offset)
        offset += TEAM_DIM
        
        # Field state (35 dims)
        WEATHER_TO_IDX = {'sunnyday': 0, 'raindance': 1, 'sandstorm': 2, 'snow': 3, 'hail': 3}
        TERRAIN_TO_IDX = {'electricterrain': 0, 'grassyterrain': 1, 'psychicterrain': 2, 'mistyterrain': 3}
        SIDE_COND_TO_IDX = {'stealthrock': 0, 'spikes': 1, 'toxicspikes': 2, 'stickyweb': 3,
                            'reflect': 4, 'lightscreen': 5, 'auroraveil': 6, 'tailwind': 7}
        
        # Weather
        if battle.weather:
            weather_key = list(battle.weather.keys())[0].name.lower() if battle.weather else ''
            if weather_key in WEATHER_TO_IDX:
                features[offset + WEATHER_TO_IDX[weather_key]] = 1.0
        
        # Terrain
        if battle.fields:
            for field in battle.fields:
                field_name = field.name.lower() if hasattr(field, 'name') else str(field).lower()
                if field_name in TERRAIN_TO_IDX:
                    features[offset + 4 + TERRAIN_TO_IDX[field_name]] = 1.0
        
        # Side conditions
        if battle.side_conditions:
            for cond, val in battle.side_conditions.items():
                cond_name = cond.name.lower() if hasattr(cond, 'name') else str(cond).lower()
                if cond_name in SIDE_COND_TO_IDX:
                    features[offset + 14 + SIDE_COND_TO_IDX[cond_name]] = min(val / 3.0, 1.0)
        
        if battle.opponent_side_conditions:
            for cond, val in battle.opponent_side_conditions.items():
                cond_name = cond.name.lower() if hasattr(cond, 'name') else str(cond).lower()
                if cond_name in SIDE_COND_TO_IDX:
                    features[offset + 22 + SIDE_COND_TO_IDX[cond_name]] = min(val / 3.0, 1.0)
        
        offset += FIELD_DIM
        
        # Meta context (23 dims)
        features[offset] = min(battle.turn / 100.0, 1.0)
        
        our_total_hp = sum(p.current_hp_fraction for p in battle.team.values() if p and p.current_hp_fraction)
        opp_total_hp = sum(p.current_hp_fraction for p in battle.opponent_team.values() if p and p.current_hp_fraction)
        features[offset + 1] = our_total_hp / 6.0
        features[offset + 2] = opp_total_hp / 6.0
        features[offset + 3] = (our_total_hp - opp_total_hp) / 6.0
        
        our_fainted = sum(1 for p in battle.team.values() if p and p.fainted)
        opp_fainted = sum(1 for p in battle.opponent_team.values() if p and p.fainted)
        features[offset + 4] = our_fainted / 6.0
        features[offset + 5] = opp_fainted / 6.0
        features[offset + 6] = (opp_fainted - our_fainted) / 6.0
        
        features[offset + 7] = len(battle.team) / 6.0
        features[offset + 8] = len(battle.opponent_team) / 6.0
        
        # Game phase
        if battle.turn <= 5:
            features[offset + 9] = 1.0
        elif battle.turn <= 20:
            features[offset + 10] = 1.0
        else:
            features[offset + 11] = 1.0
        
        return features
    
    def get_potential(self, battle: AbstractBattle) -> float:
        """Get current potential Φ(s) for debugging/logging."""
        if self.win_predictor is None:
            return 0.5
        features = self._extract_features(battle)
        return self.win_predictor.predict(features)
