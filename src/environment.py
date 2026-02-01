"""
Custom Gymnasium environment for Gen 9 Random Battle RL training.
Extends poke_env's SinglesEnv with custom observations and rewards.

Reward Design: Sparse terminal reward (Wang 2024)
- +1.0 on win, -1.0 on loss, 0.0 otherwise
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
from .actions import ActionHandler


class Gen9RLEnvironment(SinglesEnv, gymnasium.Env):
    """
    Custom Gymnasium environment for Gen 9 Random Battle.
    ...
    """


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
        
        self.action_handler = ActionHandler()
        
        # Explicitly define Gym spaces to satisfy validation without waiting for agent creation
        # Action space: 26 discrete actions (poke-env SinglesEnv for Gen9)
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

        # New battle => reset any cross-episode state
        try:
            self.belief_tracker.reset()
        except Exception:
            pass
        
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

    def get_additional_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Extend poke-env info dicts with battle outcome and metadata.
        This is critical for callbacks (e.g. EloCallback) in SubprocVecEnv, where
        direct environment introspection from the main process is not possible.
        """
        agent1 = self.agent1.username
        agent2 = self.agent2.username

        info: Dict[str, Dict[str, Any]] = {agent1: {}, agent2: {}}

        def add_battle_info(agent: str, battle: Optional[AbstractBattle]):
            if not battle:
                return
            info[agent]["battle_tag"] = getattr(battle, "battle_tag", None)
            info[agent]["turn"] = getattr(battle, "turn", 0)
            finished = bool(getattr(battle, "finished", False))
            info[agent]["battle_finished"] = finished

            if finished:
                won = getattr(battle, "won", None)
                if won is True:
                    info[agent]["battle_won"] = True
                    info[agent]["result"] = "win"
                elif won is False:
                    info[agent]["battle_won"] = False
                    info[agent]["result"] = "loss"
                else:
                    info[agent]["battle_won"] = False
                    info[agent]["result"] = "draw"

        add_battle_info(agent1, getattr(self, "battle1", None))
        add_battle_info(agent2, getattr(self, "battle2", None))

        # Side-channel: allow wrappers/scripts to attach an external opponent identifier.
        # (Used by EloCallback to track rating vs sampled self-play checkpoints.)
        if hasattr(self, "external_opponent_id"):
            info[agent1]["opponent_id"] = getattr(self, "external_opponent_id")

        return info


    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        """
        Convert battle state to observation vector.
        
        Args:
            battle: Current battle state
            
        Returns:
            Observation vector
        """
        # Update beliefs only from agent1's perspective. poke-env calls embed_battle for
        # both battle1 and battle2 each step; updating from both perspectives would
        # contaminate the belief state (especially in mirror-species matchups).
        try:
            if battle is not None and battle.player_username == self.agent1.username:
                self._update_beliefs_from_battle(battle)
        except Exception:
            # If battle/agent objects are not fully initialized, fail open.
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

            # Record revealed tera type once terastallization happens.
            try:
                if getattr(pokemon, "is_terastallized", False) and getattr(pokemon, "tera_type", None):
                    self.belief_tracker.update(species, observed_tera=pokemon.tera_type.name.lower())
            except Exception:
                pass


    # ... (metrics methods) ...

    # =========================================================================
    # TERMINAL REWARD FUNCTION
    # =========================================================================
    
    def calc_reward(self, battle: AbstractBattle, last_battle: Optional[AbstractBattle] = None) -> float:
        """
        Calculates terminal reward (+1 for win, -1 for loss, 0 otherwise)
        Matches Wang (2024) MIT Thesis method.
        """
        if getattr(battle, 'won', None) is True:
            return 1.0
        elif getattr(battle, 'won', None) is False:
            return -1.0
        return 0.0

    def step(self, actions):
        """
        Run one timestep (poke-env ParallelEnv API).

        This environment is wrapped by poke-env's `SingleAgentWrapper`, so `actions` is
        expected to be a dict mapping usernames -> discrete action indices.
        """
        self._last_action_invalid = False
        self._current_action_index = actions

        try:
            return super().step(actions)
        except (ValueError, IndexError) as e:
            import logging
            logging.getLogger(__name__).error(f"CRITICAL ENV ERROR: {e}")

            try:
                obs_dim = int(self._observation_space.shape[0])
            except Exception:
                obs_dim = int(getattr(self.obs_builder, "observation_size", 0))

            zero_obs = np.zeros((obs_dim,), dtype=np.float32)
            agent1 = getattr(self.agent1, "username", self.possible_agents[0])
            agent2 = getattr(self.agent2, "username", self.possible_agents[1])

            observations = {agent1: zero_obs, agent2: zero_obs}
            rewards = {agent1: 0.0, agent2: 0.0}
            terminated = {agent1: True, agent2: True}
            truncated = {agent1: False, agent2: False}
            infos = {agent1: {"error": str(e)}, agent2: {"error": str(e)}}

            return observations, rewards, terminated, truncated, infos

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
