from stable_baselines3.common.callbacks import BaseCallback
from typing import Optional
from src.elo_tracker import EloTracker
import numpy as np

class WinRateCallback(BaseCallback):
    """
    Callback to record battle results for win rate tracking.
    Used for gated curriculum progression.
    """
    
    def __init__(self, win_tracker: 'WinRateTracker', verbose: int = 0):
        super().__init__(verbose)
        self.win_tracker = win_tracker
        
    def _on_step(self) -> bool:
        """Check for battle completion and record results."""
        dones = self.locals.get("terminated", self.locals.get("dones", []))
        infos = self.locals.get("infos", [])
        
        for i, done in enumerate(dones):
            if done and i < len(infos):
                info = infos[i]
                
                # Get opponent type from info dict (injected by CurriculumWrapper)
                agent_type = info.get('opponent_type', 'unknown')
                if agent_type in ['unknown', 'frozen', 'random', 'max_bp', 'baseline']:
                    continue
                
                # Get battle result from info dict
                won = info.get('battle_won', False)
                
                # Record result
                self.win_tracker.record_result(agent_type, won)
        
        return True


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Log custom metrics from info dict
        infos = self.locals.get("infos", [])
        if infos:
            for key, value in infos[0].items():
                if key.startswith("custom/") or key.startswith("reward/"):
                    self.logger.record(key, value)
        return True

class EloCallback(BaseCallback):
    """
    Callback to update ELO ratings after each battle.
    """
    
    def __init__(
        self, 
        elo_tracker: EloTracker,
        current_checkpoint: str,
        opponent_checkpoint: str,
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.elo_tracker = elo_tracker
        self.current_checkpoint = current_checkpoint
        self.opponent_checkpoint = opponent_checkpoint
        
    def _on_step(self) -> bool:
        """
        Called after each step. Checks for battle completion.
        """
        # SB3 v2 uses 'terminated' and 'truncated', v1 uses 'dones'
        dones = self.locals.get("terminated", self.locals.get("dones", []))
        infos = self.locals.get("infos", [])
        
        for i, done in enumerate(dones):
            if done:
                # Get win status from info or environment
                won = False
                info = infos[i]
                
                # Try to get result from info
                if "battle_won" in info:
                    won = info["battle_won"]
                elif "result" in info:
                    won = info["result"] == "win"
                else:
                    # Fallback: Deep inspection of environment
                    try:
                        # Assuming DummyVecEnv -> Monitor -> SingleAgentWrapper -> Gen9RLEnvironment
                        # This path might vary depending on wrappers
                        raw_env = self.training_env.envs[i].unwrapped
                        if hasattr(raw_env, "last_battle_won"):
                             won = raw_env.last_battle_won
                        elif hasattr(raw_env, "current_battle") and raw_env.current_battle:
                             won = raw_env.current_battle.won
                    except Exception:
                        pass
                
                # Update ELO
                self.elo_tracker.update_ratings(
                    player_a=self.current_checkpoint,
                    player_b=self.opponent_checkpoint,
                    winner_a=won
                )
                
                if self.verbose > 0:
                    print(f"[ELO] Game Finished. Winner: {'Agent' if won else 'Opponent'}")
                    
        return True

class OpponentSamplingCallback(BaseCallback):
    """
    Callback to periodically resample opponents for curriculum progression.
    Ensures that workers rotate their opponents based on current progress.
    Also logs the sampling decisions for observability.
    """
    def __init__(self, check_freq: int = 2048, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.last_update_step = 0
        
    def _on_step(self) -> bool:
        return True
        
    def _on_rollout_end(self) -> None:
        """
        Called after each rollout. Perfect time to swap opponents for the next batch.
        """
        import os
        
        # Skip in RIGOROUS_MODE - we use fixed baseline opponents
        if os.environ.get("RIGOROUS_MODE", "0") == "1":
            return
        
        # Calculate current progress - use _total_timesteps if set by training loop
        # NOTE: model.total_timesteps resets each learn() call, not global progress
        # We inject _global_total_steps ourselves from train.py
        global_total = getattr(self.model, '_global_total_steps', 10_000_000)
        progress = self.model.num_timesteps / global_total
             
        # Call resample_opponent on all workers
        # This returns a list of new agent types from each worker
        try:
             results = self.training_env.env_method("resample_opponent", progress=progress)
             
             # Log the updates clearly as requested by user
             unique_agents, counts = np.unique(results, return_counts=True)
             
             print(f"\n[Curriculum] Opponent Rotation (Step {self.num_timesteps}, Progress {progress:.1%}):")
             for agent, count in zip(unique_agents, counts):
                 if agent: # Ignore None results
                     print(f"  - {agent}: {count} workers")
                     
        except Exception as e:
             if self.verbose > 0:
                 print(f"[Curriculum] Warning: Failed to resample opponents: {e}")

class SmartMetricsCallback(BaseCallback):
    """
    Callback to log detailed metrics:
    - Current distribution of opponents
    - Overall winrate against each
    - Winrate in the last 20 games against each
    """
    def __init__(self, win_tracker: 'WinRateTracker', verbose=0):
        super().__init__(verbose)
        self.win_tracker = win_tracker
        self.tracker_stats = {} # Cache for fast checking

    def _on_step(self) -> bool:
        return True
        
    def _on_rollout_end(self) -> None:
        """
        Log detailed stats at the end of rollouts.
        """
        if self.verbose > 1 and self.win_tracker:
            print("\n" + "="*80)
            print(f" {f'OPPONENT MASTERY REPORT':^76} ")
            print("="*80)
            print(f" {'Opponent Type':<25} | {'Last 20 WR':<10} | {'Total WR':<10} | {'Games':<10}")
            print("-" * 80)
            
            # Get stats from tracker
            # Note: accessing private dict _stats for read-only reporting
            if hasattr(self.win_tracker, '_stats'):
                stats = self.win_tracker._stats
                
                # Sort by games played (descending)
                # Note: stats is a dict of {agent_name: AgentStats_object}
                sorted_agents = sorted(stats.items(), key=lambda x: x[1].total_games, reverse=True)
                
                for agent, data in sorted_agents:
                    # data is AgentStats object
                    wins = data.total_wins
                    games = data.total_games
                    total_wr = data.total_win_rate
                    
                    recent_wr = data.recent_win_rate
                    recent_games = data.recent_games
                    
                    if recent_games > 0:
                        recent_str = f"{recent_wr:5.1%}"
                    else:
                        recent_str = "N/A"
                    
                    print(f" {agent:<25} | {recent_str:<10} | {total_wr:6.1%}    | {games:<10}")
            
            print("="*80 + "\n")
