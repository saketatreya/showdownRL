"""
Custom Gymnasium environment for Gen 9 Random Battle RL training.
Extends poke_env's SinglesEnv with custom observations and rewards.

Reward Design: Delta-based state value function
- Material Layer: HP balance + fainted count
- Positional Layer: Type matchup advantage
- Future Potential Layer: Status + stat boosts
- Terminal Layer: Win/loss bonus + step cost
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import gymnasium
from gymnasium.spaces import Box

from poke_env.environment import SinglesEnv
from poke_env.battle import AbstractBattle

from .belief_tracker import BeliefTracker
from .embeddings import ObservationBuilder
from .utils import load_pokemon_data
from .rewards import DeltaRewardEvaluator, HybridRewardEvaluator
from .actions import ActionHandler
from .config import RewardConfig
from .curriculum_agents.win_tracker import WinRateTracker


class Gen9RLEnvironment(SinglesEnv, gymnasium.Env):
    """
    Custom Gymnasium environment for Gen 9 Random Battle.
    ...
    """
    # Get default reward config from config.py (single source of truth)
    # Extra env-specific keys added here
    _ENV_SPECIFIC_CONFIG = {
        'step_cost': 0.01,
        'switch_tax': 0.10,
        'move_fail_penalty': 0.15,
        'invalid_action_penalty': 2.0,
    }

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    def __init__(
        self,
        # ... args ...
        pokemon_data: Optional[Dict[str, Any]] = None,
        opponent: Optional[Any] = None,
        start_challenging: bool = False,
        pokemon_data_path: str = "gen9randombattle.json",
        reward_config: Optional[Dict[str, float]] = None,
        battle_format: str = "gen9randombattle",
        teambuilder: Optional[Any] = None,
        **kwargs
    ):
        # SinglesEnv does not accept 'teambuilder' in its __init__, so we handle it manually
        # Explicitly pop it just in case it's in kwargs (though listed in args now)
        if 'teambuilder' in kwargs:
            teambuilder = kwargs.pop('teambuilder')
            
        self._opponent = opponent
        self.opponent = opponent # standard poke_env attribute
        self._start_challenging = start_challenging
        # Do not pass opponent to super() via kwargs as SinglesEnv does not accept it.

        super().__init__(battle_format=battle_format, **kwargs)
        
        # Manually set the teambuilder to the internal _team attribute
        # The Player class (parent of SinglesEnv or member) uses self._team to generate teams
        if teambuilder:
            self._team = teambuilder
            
            # DEBUG: Inspect structure
            # print(f"[Gen9RLEnvironment] Attributes: {dir(self)}")
            
            # Check for common Player/Agent attributes
            # Check for common Player/Agent attributes
            if hasattr(self, 'agent1'):
                self.agent1._team = teambuilder
                # print(f"[Gen9RLEnvironment] Assigned teambuilder to self.agent1 ({type(self.agent1)})")
            
            if hasattr(self, 'agent2'):
                # In self-play or some envs, agent2 might also need a team, 
                # but usually agent2 is managed by the wrapper. 
                # We assign it just in case if it accepts it.
                # However, for training agent, agent1 is the key.
                # We default to assignment if it exists.
                # We should NOT assign the same teambuilder instance to agent2 if it has state.
                # Ideally, agent2 (opponent) is managed by `OpponentPool` or updated via `CurriculumWrapper`.
                # But if we must, we should ensure it's not the same stateful iterator.
                # For safety, we usually DO NOT assign `agent2._team` here if we're in self-play mode 
                # where agent2 is actively managed. 
                # However, for simple DummyVecEnv against random stock bot, it might be needed.
                # We'll assign it BUT warn if it's the exact same object.
                if self.agent2._team is not teambuilder:
                    self.agent2._team = teambuilder
                # print(f"[Gen9RLEnvironment] Assigned teambuilder to self.agent2 ({type(self.agent2)})")

            if hasattr(self, 'agent'):
                self.agent._team = teambuilder
                # print(f"[Gen9RLEnvironment] Assigned teambuilder to self.agent ({type(self.agent)})")
            elif hasattr(self, '_player'):  # Some versions use _player
                self._player._team = teambuilder
                # print(f"[Gen9RLEnvironment] Assigned teambuilder to self._player ({type(self._player)})")
            elif hasattr(self, 'player'):
                self.player._team = teambuilder
                # print(f"[Gen9RLEnvironment] Assigned teambuilder to self.player ({type(self.player)})")
            
            if not any(hasattr(self, attr) for attr in ['agent1', 'agent', '_player', 'player']):
                 print(f"[Gen9RLEnvironment] Assigned teambuilder to self (Inheritance mode or agent not found yet)")
                 # Fallback: If agent is created lazily, we might need to hook reset()
        
        # ... (Loading data remains same) ...
        if pokemon_data is None:
            self.pokemon_data = load_pokemon_data(pokemon_data_path)
        else:
            self.pokemon_data = pokemon_data
            
        # Increase challenge timeout for parallel training stability
        self._challenge_timeout = 10  # Reduced to 10s to fail fast on rejected teams (was 120)
        
        self.belief_tracker = BeliefTracker(self.pokemon_data)
        self.obs_builder = ObservationBuilder(self.pokemon_data, self.belief_tracker)
        
        low, high = self.obs_builder.get_observation_space_bounds()
        self.observation_spaces = {
            agent: Box(low=low, high=high, dtype=np.float32)
            for agent in self.possible_agents
        }
        
        default_config = RewardConfig()
        self.reward_config = default_config.model_dump()
        self.reward_config.update(self._ENV_SPECIFIC_CONFIG)
        if reward_config:
            self.reward_config.update(reward_config)
        
        # Initialize Reward Evaluator
        reward_shaping = kwargs.get('reward_shaping', 'hybrid')
        if reward_shaping == 'potential':
            from .rewards import PotentialBasedRewardEvaluator
            self.reward_evaluator = PotentialBasedRewardEvaluator()
        else:
            self.reward_evaluator = HybridRewardEvaluator(self.reward_config)
        
        self.action_handler = ActionHandler()
        self._init_metrics()
        self._current_signals = {} # Cache for signals detected in calc_reward
        
        # Explicitly define Gym spaces to satisfy validation without waiting for agent creation
        # Action space: 26 discrete actions (See ActionHandler)
        from gymnasium.spaces import Discrete
        self._action_space = Discrete(26)
        
        # Observation space: Box
        self._observation_space = Box(low=low, high=high, dtype=np.float32)

    def reset(self, seed=None, options=None):
        """
        Reset environment with robust retry logic for server overloading.
        """
        import asyncio
        import time
        from pathlib import Path
        
        # Increment internal game counter
        if not hasattr(self, '_games_played'):
            self._games_played = 0
        self._games_played += 1
        
        # Determine if we should save replay (1 in 1000)
        save_replay = (self._games_played % 1000 == 0)
        replay_path = "replays" if save_replay else None
        
        # Propagate to internal agents
        if hasattr(self, 'agent1'):
            self.agent1.save_replay_path = replay_path
        if hasattr(self, 'agent2'):
            self.agent2.save_replay_path = replay_path
            
        if save_replay:
             print(f"[Gen9RLEnvironment] Saving replay for game {self._games_played} to {replay_path}")

        max_retries = 5
        base_delay = 10.0 # Increased to 10s to allow server timeouts to clear
        
        for attempt in range(max_retries):
            try:
                # ===============================================================
                pass
                # ===============================================================

                # RESET REWARD EVALUATOR FIRST to prevent delta spikes from previous episode
                if hasattr(self, 'reward_evaluator'):
                    self.reward_evaluator.reset()

                # Standard reset
                obs, info = super().reset(seed=seed, options=options)
                
                # ===============================================================
                # BATTLE START LOGGING (Disabled for production speed)
                # ===============================================================
                # try:
                #     battle = getattr(self, 'current_battle', None) or getattr(self, '_current_battle', None)
                #     if battle:
                #         battle_tag = getattr(battle, 'battle_tag', 'unknown')
                #         format_str = getattr(battle, 'format', 'unknown')
                #         # print(f"\n[BATTLE START] {battle_tag} | Format: {format_str}")
                # except Exception:
                #     pass
                # ===============================================================

                
                return obs, info
                
            except Exception as e:
                # Catch ALL errors during reset (Timeouts, KeyErrors from invalid Pokemon data, etc.)
                if attempt < max_retries - 1:
                    # Aggressive Recovery (Unconditional)
                    print(f"[Env] Reset Error: {e}. Initiating Aggressive Recovery...", flush=True)
                    try:
                        agent = getattr(self, 'agent1', None)
                        if agent:
                            # Try to get client and loop
                            client = getattr(agent, '_ps_client', None)
                            if client:
                                loop = getattr(client, '_loop', None) or getattr(agent, '_loop', None)
                                
                                if loop and loop.is_running():
                                    print(f"[Env] Found Loop. Sending /cancelchallenge...", flush=True)
                                    
                                    async def _send_cancel():
                                        await client.send_message("/cancelchallenge", room="")
                                        return True
                                        
                                    future = asyncio.run_coroutine_threadsafe(_send_cancel(), loop)
                                    try:
                                        future.result(timeout=2.0) # Wait for it to actually send
                                        print(f"[Env] Cancel command sent successfully.", flush=True)
                                    except Exception as fut_err:
                                        print(f"[Env] Cancel command timed out or failed: {fut_err}", flush=True)
                                else:
                                    print(f"[Env] Loop not running. Cannot cancel.", flush=True)
                            else:
                                print(f"[Env] No PS Client found on agent.", flush=True)
                    except Exception as rec_ex:
                        print(f"[Env] Aggressive Recovery Exception: {rec_ex}", flush=True)
                    
                    delay = (base_delay * (attempt + 1)) + np.random.uniform(2.0, 5.0)
                    print(f"[Env] Retrying in {delay:.1f}s... (Check for invalid teams/data)")
                    time.sleep(delay)
                else:
                    print(f"[Env] Critical: Failed to reset after {max_retries} attempts.")
                    raise e
        return None  # Should not reach here

    def _init_metrics(self):
        """Initialize episode metrics to default values."""
        self._episode_metrics = {
            "damaging_moves": 0,
            "total_moves": 0,
            "setup_moves": 0,
            "setup_success": 0,
            "switch_events": 0,
            "switch_matchup_delta": 0.0,
            "move_failures": 0,
            "reward_components": {
                "hp": 0.0, "fainted": 0.0, "matchup": 0.0, "victory": 0.0
            }
        }

    @property
    def observation_space(self):
        """Return the observation space."""
        low, high = self.obs_builder.get_observation_space_bounds()
        return Box(low=low, high=high, dtype=np.float32)

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        """
        Convert battle state to observation vector.
        
        Args:
            battle: Current battle state
            
        Returns:
            Observation vector
        """
        # Update beliefs based on battle events
        self._update_beliefs_from_battle(battle)
        
        # Build observation
        embedding = self.obs_builder.embed_battle(battle)
        
        # Update metrics for this step
        if hasattr(self, '_update_metrics'):
            self._update_metrics(battle)
            
        return embedding
    
    def _update_beliefs_from_battle(self, battle: AbstractBattle):
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


    # ... (metrics methods) ...

    # =========================================================================
    # DELTA-BASED REWARD FUNCTION
    # =========================================================================
    
    def calc_reward(self, battle: AbstractBattle, last_battle: Optional[AbstractBattle] = None) -> float:
        """
        Calculate reward using Hybrid Evaluator.
        Detects events (U-turn, Failure) by comparing battle vs last_battle.
        """
        # Detect signals
        signals = {
            'event_switch_type': None,
            'event_move_failed': False
        }
        
        # 1. Switch Detection
        # We need to know if we SWITCHED explicitly or via U-turn.
        # last_battle has previous active. battle has current.
        if last_battle and battle:
            pre_active = last_battle.active_pokemon
            post_active = battle.active_pokemon
            
            if pre_active and post_active and pre_active.species != post_active.species:
                # Active changed.
                # Was it forced?
                if last_battle.force_switch:
                    signals['event_switch_type'] = 'forced'
                else:
                    # Check action type
                    # We need the last action index. poke-env stores self.last_action (if available?)
                    # If not, we infer: 
                    # If we used a Move (0-3) and switched -> U-turn.
                    # If we used Switch (4+) -> Manual.
                    # Problem: We don't easily have 'last_action' index here unless we stored it.
                    # But wait! 'step' hasn't returned yet. 'self._last_action' might be set?
                    # poke-env doesn't expose _last_action widely.
                    # However, we can use the 'turn' count? No.
                    # Let's trust 'Manual Switch' has distinct signature?
                    # Actually, we can store 'last_action_index' in step() before calling super().step()?
                    # But calc_reward is called INSIDE super().step().
                    # So we need to store it in 'Gen9RLEnvironment' before calling super().step().
                    # We can use 'self._current_action_index' set in step().
                    pass 
            
        # Refined Logic relying on self._current_action_index (set in step)
        if hasattr(self, '_current_action_index'):
            action = self._current_action_index
            # Fix for TypeError: handle dict actions (sometimes passed by wrappers)
            if isinstance(action, dict):
                # Assume action is {'action': int} or similar?
                # Actually, stable_baselines3 passes np.ndarray or int. 
                # If wrapped in Dict space? 
                # Let's inspect, but for now safe cast.
                # If dict, try to get value. Use list(action.values())[0]?
                # Or just skip signal detection if invalid.
                if 'action' in action:
                     action = action['action']
                else: 
                     # invalid dict input, fallback to 0
                     action = 0
            
            # Ensure action is int/float before comparing
            if isinstance(action, (int, float, np.integer, np.floating)):
                is_move = action < 4
                is_switch = action >= 4
                
                if last_battle and battle:
                     pre_active = last_battle.active_pokemon
                     post_active = battle.active_pokemon
                     if pre_active and post_active and pre_active.species != post_active.species:
                         if last_battle.force_switch:
                             signals['event_switch_type'] = 'forced'
                         elif is_switch:
                             signals['event_switch_type'] = 'manual'
                         elif is_move:
                             signals['event_switch_type'] = 'uturn'
        
        # 2. Move Failure
        # Check messages
        if hasattr(battle, '_messages') and battle._messages:
             # Initialize processed pointer if missing
             if not hasattr(self, '_last_message_index'):
                 self._last_message_index = 0
                 
             # Only check NEW messages
             new_messages = battle._messages[self._last_message_index:]
             for msg in new_messages:
                 msg_str = str(msg).lower() if msg else ''
                 if '|-fail|' in msg_str or '|-immune|' in msg_str or '|-miss|' in msg_str:
                     signals['event_move_failed'] = True
                     break
             
             # Update pointer
             self._last_message_index = len(battle._messages)
        
        # Store for step() to pick up
        self._current_signals = signals
        
        # Calculate Reward
        reward = self.reward_evaluator.calc_reward(battle, info=signals)
        
        # Sync metrics
        if hasattr(self, '_episode_metrics'):
            self._episode_metrics['reward_components'] = self.reward_evaluator.get_reward_components().copy()
            if signals['event_switch_type'] == 'manual':
                self._episode_metrics['switch_events'] += 1
            if signals['event_move_failed']:
                self._episode_metrics['move_failures'] += 1
                
        return reward

    def step(self, action):
        """Run one timestep."""
        self._last_action_invalid = False
        
        # Store action for calc_reward to see
        self._current_action_index = action
        
        try:
            obs, reward, terminated, truncated, info = super().step(action)
            
            # Handle both dict (multi-agent) and float (single-agent) rewards
            is_dict_reward = isinstance(reward, dict)
            
            if is_dict_reward:
                # Multi-agent environment - extract our agent's reward
                if not reward:  # Handle empty dict
                    agent_key = self.possible_agents[0]
                else:
                    agent_key = list(reward.keys())[0]
                reward_value = reward.get(agent_key, 0.0)
            else:
                reward_value = reward
                agent_key = None
            
            # Apply penalty for invalid actions
            if hasattr(self, '_last_action_invalid') and self._last_action_invalid:
                 penalty = self.reward_config.get('invalid_action_penalty', 2.0)
                 reward_value -= penalty
            
            # Update reward (dict or float)
            if is_dict_reward:
                reward[agent_key] = reward_value
            else:
                reward = reward_value
            
            # Populate info with signals detected in calc_reward
            info.update(self._current_signals)
            
            # Metrics
            if hasattr(self, 'get_metrics_info'):
                metrics = self.get_metrics_info()
                metrics['custom/switch_this_turn'] = 1.0 if self._current_signals.get('event_switch_type') == 'manual' else 0.0
                metrics['custom/move_failed_this_turn'] = 1.0 if self._current_signals.get('event_move_failed') else 0.0
                info.update(metrics)
                for val in info.values():
                    if isinstance(val, dict): val.update(metrics)
            
            # === COMPREHENSIVE REWARD LOGGING ===
            # Log rewards for verification (first 5 turns per battle, and terminal)
            try:
                battle = getattr(self, 'current_battle', None) or getattr(self, '_current_battle', None)
                if battle:
                    turn = battle.turn
                    if turn <= 5 or terminated:
                        # Get reward components for detailed logging
                        components = self.reward_evaluator.get_reward_components() if hasattr(self, 'reward_evaluator') else {}
                        
                        log_msg = f"\n[REWARD] Turn {turn}: Total={reward_value:+.2f}"
                        
                        # Add key components
                        if components:
                            hp_delta = components.get('hp_delta', 0)
                            faint_delta = components.get('faint_delta', 0)
                            log_msg += f" | HP:{hp_delta:+.2f} Faint:{faint_delta:+.1f}"
                        
                        if terminated:
                            won = battle.won if hasattr(battle, 'won') else None
                            outcome = "WIN ✓" if won else ("LOSS ✗" if won == False else "DRAW")
                            log_msg += f" | {outcome}"
                        
                        print(log_msg)
            except Exception:
                pass  # Logging should never crash step
            # =====================================
            
            return obs, reward, terminated, truncated, info
            
        except (ValueError, IndexError) as e:
            import logging
            logging.getLogger(__name__).error(f"CRITICAL ENV ERROR: {e}")
            dummy_obs = self.embed_battle(self.current_battle) if self.current_battle else np.zeros(self.observation_space.shape)
            return dummy_obs, 0.0, True, False, {"error": str(e)}
            
            # duplicate exception handler removed

    # =========================================================================
    # ACTION HANDLING - Remap invalid actions to valid ones
    # =========================================================================
    
    def action_to_order(self, action, battle, fake=False, strict=False):
        """
        Convert action index to battle order, with penalty tracking.
        """
        try:
            # Try the parent's action_to_order
            return super().action_to_order(action, battle, fake=fake, strict=True)
        except (ValueError, IndexError):
            # Invalid action - remap to a valid one using ActionHandler
            order = self.action_handler.get_fallback_order(action, battle)
            
            # Apply penalty via side-channel (member variable)
            if not fake:
                 self._last_action_invalid = True
                 
            return order
    
    def order_to_action(self, order, battle, fake=False, strict=True):
        """
        Robust order_to_action conversion.
        Captures invalid opponent orders to prevent training crashes.
        """
        try:
             return super().order_to_action(order, battle, fake=fake, strict=strict)
        except (ValueError, IndexError):
             # Log warning if needed, but return a safe default (e.g., 0)
             # This prevents SingleAgentWrapper from crashing when the opponent logic 
             # tries to perform an action deemed invalid by the local client state.
             # The server is the ultimate arbiter.
             return 0

class TrainingEnvManager:
    """
    Manages training environment lifecycle.
    
    Persistent implementation: Creates ONE environment and hot-swaps the opponent.
    This prevents thread churn and segfaults.
    """
    
    def __init__(
        self,
        pokemon_data: Optional[Dict[str, Any]] = None,
        pokemon_data_path: str = "gen9randombattle.json",
        battle_format: str = "gen9randombattle",
        reward_config: Optional[Dict[str, float]] = None,
        opponent = None, # Initial opponent
        opponent_type: str = 'baseline',  # For win rate tracking
    ):
        self.pokemon_data = pokemon_data if pokemon_data else load_pokemon_data(pokemon_data_path)
        self.pokemon_data_path = pokemon_data_path
        self.battle_format = battle_format
        self.reward_config = reward_config
        self._opponent_type = opponent_type
        
        # Initialize the SINGLE persistent environment
        self._init_env(opponent)
    
    def _init_env(self, opponent):
        """Create the persistent environment instance."""
        from poke_env.environment import SingleAgentWrapper
        
        # Create base env
        self._base_env = Gen9RLEnvironment(
            pokemon_data=self.pokemon_data,
            pokemon_data_path=self.pokemon_data_path,
            battle_format=self.battle_format,
            reward_config=self.reward_config,
        )
        
        # Wrap it
        if opponent:
            self._current_wrapper = SingleAgentWrapper(self._base_env, opponent=opponent)
        else:
            # Temporary placeholder if no opponent yet
            self._current_wrapper = SingleAgentWrapper(self._base_env, opponent=opponent)
        
        # Wrap with CurriculumWrapper - pass opponent_type for win rate tracking
        self._current_wrapper = CurriculumWrapper(self._current_wrapper, opponent_type=self._opponent_type)
            
        self._current_wrapper._base_env = self._base_env # Link back
    
    def swap_opponent(self, new_opponent, opponent_type: str = None):
        """
        Hot-swap the opponent without destroying the environment.
        This keeps the background threads / websocket alive.
        
        Args:
            new_opponent: The new opponent player
            opponent_type: Name of opponent type for win rate tracking (e.g., 'max_damage')
        """
        if opponent_type:
            self._opponent_type = opponent_type
            # Update CurriculumWrapper's opponent_type
            self._current_wrapper.set_opponent_type(opponent_type)
        
        if self._current_wrapper:
            # Access underlying SingleAgentWrapper (wrapped by CurriculumWrapper)
            # CurriculumWrapper -> SingleAgentWrapper
            saw = self._current_wrapper.env 
            saw._opponent = new_opponent
            
            # CRITICAL: We might need to reset underlying battle queue or state
            # but poke-env handles most of this on reset().
            # Just ensuring the wrapper points to the new guy is usually enough 
            # IF the environment is reset immediately after.
            
            return self._current_wrapper
            
        # Fallback (shouldn't happen with proper init)
    def close(self):
        """Close the environment."""
        if self._current_wrapper:
            self._current_wrapper.close()

def make_env(
    pokemon_data: Optional[Dict[str, Any]] = None,
    pokemon_data_path: str = "gen9randombattle.json",
    battle_format: str = "gen9randombattle",
    **kwargs
) -> Gen9RLEnvironment:
    """Factory function to create a Gen9RLEnvironment."""
    return Gen9RLEnvironment(
        pokemon_data=pokemon_data,
        pokemon_data_path=pokemon_data_path,
        battle_format=battle_format,
        **kwargs
    )


def make_training_env(
    opponent,
    pokemon_data: Optional[Dict[str, Any]] = None,
    pokemon_data_path: str = "gen9randombattle.json",
    battle_format: str = "gen9randombattle",
    reward_config: Optional[Dict[str, float]] = None,
):
    """
    Create a training-ready environment with opponent.
    """
    from poke_env.environment import SingleAgentWrapper
    env = Gen9RLEnvironment(
        pokemon_data=pokemon_data,
        pokemon_data_path=pokemon_data_path,
        battle_format=battle_format,
        reward_config=reward_config,
    )
    wrapped = SingleAgentWrapper(env, opponent=opponent)
    wrapped._base_env = env
    wrapped._opponent = opponent
    
    # Wrap with CurriculumWrapper
    wrapped = CurriculumWrapper(wrapped)
    
    return wrapped

class CurriculumWrapper(gymnasium.Wrapper):
    """
    Wrapper to explicitly expose set_progress method to SubprocVecEnv's env_method.
    Also tracks opponent type and injects battle results into info dict for win rate tracking.
    Handles dynamic opponent resampling to ensure curriculum progression.
    """
    def __init__(self, env, opponent_type: str = 'unknown', opponent_pool = None, win_tracker_path: str = None, pokemon_data = None):
        super().__init__(env)
        self._opponent_type = opponent_type
        self.opponent_pool = opponent_pool
        self.win_tracker_path = win_tracker_path
        self.pokemon_data = pokemon_data
        
    def set_opponent_type(self, opponent_type: str):
        """Set the current opponent type for win rate tracking."""
        self._opponent_type = opponent_type
        
    def resample_opponent(self, progress: float):
        """
        Resample the opponent using the pool and hot-swap it.
        Called via VecEnv.env_method() from the main process callback.
        """
        if not self.opponent_pool or not self.win_tracker_path:
            return # Cannot resample without pool/tracker
            
        # Create local tracker instance to read stats
        tracker = WinRateTracker(self.win_tracker_path)
        
        # Sample new opponent
        # Note: logging happens inside sample_opponent
        opponent, tier, agent_type = self.opponent_pool.sample_opponent(
            progress=progress,
            win_tracker=tracker,
            pokemon_data=self.pokemon_data
        )
        
        # Update internal tracking
        self._opponent_type = agent_type
        
        # Hot-swap opponent in SingleAgentWrapper
        # Access unwrapped env until we find SingleAgentWrapper or similar
        # Typically: CurriculumWrapper -> Monitor -> SingleAgentWrapper
        
        current = self.env
        swapped = False
        while hasattr(current, 'env'):
            # poke-env uses 'opponent' (no underscore), check both for safety
            if hasattr(current, 'opponent'):
                # print(f"[CurriculumWrapper] HOT-SWAPPING opponent to {agent_type}")
                current.opponent = opponent
                swapped = True
                break
            elif hasattr(current, '_opponent'):
                print(f"[CurriculumWrapper] HOT-SWAPPING opponent (_opponent) to {agent_type}")
                current._opponent = opponent
                swapped = True
                break
            
            if hasattr(current, 'env'):
                current = current.env
            else:
                break
        
        if not swapped:
            print(f"[CurriculumWrapper] WARNING: Could not find opponent attribute to swap!")
        
        # CRITICAL FIX: Also update PokeEnv's internal agent2's team
        # The SingleAgentWrapper.opponent only handles choose_move()
        # But PokeEnv's agent2 is what actually sends team to Showdown server
        try:
            # Find agent2 - it's in the PokeEnv which is wrapped by SingleAgentWrapper
            # Structure: CurriculumWrapper -> Monitor -> SingleAgentWrapper -> Gen9RLEnvironment
            # agent2 is in Gen9RLEnvironment (which extends PokeEnv)
            
            poke_env = None
            current = self.env
            while current is not None:
                if hasattr(current, 'agent2'):
                    poke_env = current
                    break
                current = getattr(current, 'env', None)
            
            if poke_env and hasattr(poke_env, 'agent2'):
                # Get the new team from opponent's teambuilder
                if hasattr(opponent, '_team') and opponent._team:
                    # CRITICAL: Assign the Teambuilder OBJECT, not a string output!
                    # This allows poke-env to call yield_team() for EACH battle.
                    poke_env.agent2._team = opponent._team
                    # Get a preview of what the next team looks like (for logging only)
                    try:
                        preview_team = opponent._team.yield_team() if hasattr(opponent._team, 'yield_team') else str(opponent._team)
                        print(f"[CurriculumWrapper] UPDATED agent2 team: {preview_team[:50]}...")
                    except Exception:
                        print(f"[CurriculumWrapper] UPDATED agent2 team (Teambuilder assigned)")
                elif hasattr(opponent, 'next_team'):
                    # Alternative: use next_team property (likely a static string, acceptable fallback)
                    new_team = opponent.next_team
                    if new_team:
                        poke_env.agent2._team = new_team
                        print(f"[CurriculumWrapper] UPDATED agent2 team (next_team): {new_team[:50]}...")
                    else:
                        print(f"[CurriculumWrapper] WARNING: opponent.next_team returned None")
                else:
                    print(f"[CurriculumWrapper] WARNING: Opponent {type(opponent).__name__} has no _team or next_team")
            else:
                print(f"[CurriculumWrapper] WARNING: Could not find agent2 in environment chain")
        except Exception as e:
            print(f"[CurriculumWrapper] ERROR updating agent2 team: {e}")
            
        return agent_type
        
    def set_progress(self, progress: float):
        """Pass progress to the underlying Gen9RLEnvironment."""
        if hasattr(self.unwrapped, 'set_progress'):
            self.unwrapped.set_progress(progress)
        elif hasattr(self.env, 'set_progress'):
            self.env.set_progress(progress)
            
    def step(self, action):
        """Override step to inject opponent_type and battle_won into info."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Always include opponent type
        info['opponent_type'] = self._opponent_type
        
        # Check for battle completion and add battle_won
        if terminated:
            try:
                # Try to get battle result from underlying env
                raw_env = self.unwrapped
                if hasattr(raw_env, 'current_battle') and raw_env.current_battle:
                    won = raw_env.current_battle.won
                    info['battle_won'] = won if won is not None else False
                else:
                    # Fallback: infer from reward (positive = likely win)
                    info['battle_won'] = reward > 0
            except Exception:
                info['battle_won'] = reward > 0
        
        return obs, reward, terminated, truncated, info
