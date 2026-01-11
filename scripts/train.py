"""
Training script for Gen 9 Random Battle RL bot.
Uses RecurrentPPO with self-play training and AlphaStar-style curriculum.
Supports parallel environments via SubprocVecEnv for faster training.
"""

import argparse
import json
import random
import sys
import time
import os
from pathlib import Path
from typing import Optional, Tuple, Callable

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from poke_env.player import RandomPlayer, MaxBasePowerPlayer, SimpleHeuristicsPlayer
from poke_env import AccountConfiguration

from src.environment import Gen9RLEnvironment, make_training_env, TrainingEnvManager
from src.trained_player import TrainedPlayer
from src.utils import load_pokemon_data
from src.elo_tracker import EloTracker
from src.callbacks import EloCallback, TensorboardCallback, SmartMetricsCallback, WinRateCallback, OpponentSamplingCallback
from src.curriculum_agents.win_tracker import WinRateTracker
from src.config import TrainingConfig, REWARD_CURRICULUM
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

# NOTE: The actual main() function is defined below.
# This file uses a modular structure with helper functions defined first.


def parse_args():
    parser = argparse.ArgumentParser(description="Train Gen 9 Random Battle RL bot")
    
    # Defaults from Wang-Jett-Meng unless overridden
    default_config = TrainingConfig()
    
    parser.add_argument("--timesteps", type=int, default=10_000_000,
                        help="Total timesteps to train")
    parser.add_argument("--iteration-steps", type=int, default=32768,
                        help="Steps per training iteration (Aligned to 8 envs * 2048 steps * 2 = 32768)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--log-dir", type=str, default="logs",
                        help="Directory for tensorboard logs")
    parser.add_argument("--pokemon-data", type=str, default="gen9randombattle.json",
                        help="Path to Pokemon data JSON")
    parser.add_argument("--battle-format", type=str, default="gen9curriculumbattle",
                        help="Battle format (use gen9curriculumbattle for curriculum training)")
    parser.add_argument("--lr", type=float, default=default_config.learning_rate,
                        help="Initial Learning rate")
    parser.add_argument("--n-steps", type=int, default=default_config.n_steps,
                        help="Steps per rollout")
    parser.add_argument("--batch-size", type=int, default=default_config.batch_size,
                        help="Batch size for training")
    parser.add_argument("--n-epochs", type=int, default=default_config.n_epochs,
                        help="Epochs per update")
    parser.add_argument("--gamma", type=float, default=0.9999,
                        help="Discount factor (0.9999 recommended for sparse rewards)")
    parser.add_argument("--experiment-name", type=str, default="selfplay",
                        help="Name of the experiment (affects log/checkpoint dirs)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint path")
    parser.add_argument("--dry-run", action="store_true",
                        help="Quick test run with minimal steps")
    # === NEW: Parallel environment options ===
    parser.add_argument("--n-envs", type=int, default=8,
                        help="Number of parallel environments")
    parser.add_argument("--no-subproc", action="store_true",
                        help="Use DummyVecEnv instead of SubprocVecEnv (for debugging)")
    
    return parser.parse_args()


def get_lr_schedule(initial_lr: float, schedule_type: str = "linear") -> Callable[[float], float]:
    """
    Returns a learning rate schedule function.
    
    IMPORTANT: SB3's `progress_remaining` is computed PER learn() CALL, not globally.
    When you call learn(20k) repeatedly, it resets each time.
    We work around this by tracking global steps externally.
    
    Args:
        initial_lr: The starting learning rate.
        schedule_type: 'linear', 'constant', or 'wang'.
    """
    # Track global progress externally - this gets updated by train.py
    # Default to 0 so we start at full LR
    global_state = {'steps_done': 0, 'total_steps': 10_000_000}
    
    if schedule_type == "linear":
        def func(progress_remaining: float) -> float:
            # Ignore SB3's progress_remaining, use our global tracker
            x = global_state['steps_done'] / global_state['total_steps']
            return initial_lr * (1.0 - x)
        func._global_state = global_state  # Expose for external updates
        return func
    elif schedule_type == "wang":
        def func(progress_remaining: float) -> float:
            # Ignore SB3's progress_remaining, use our global tracker
            x = global_state['steps_done'] / global_state['total_steps']
            # Formula: lr / (8x + 1)^1.5
            return initial_lr / ((8 * x + 1) ** 1.5)
        func._global_state = global_state  # Expose for external updates
        return func
    else:
        def func(progress_remaining: float) -> float:
            return initial_lr
        return func


def make_parallel_env(
    rank: int,
    pokemon_data: dict,
    battle_format: str,
    checkpoint_dir: str,
    current_progress: float = 0.0,
    log_dir: Optional[Path] = None,
) -> Callable:
    """
    Factory function for creating independent environment instances.
    Instantiates a local OpponentPoolManager to sample curriculum opponents.
    """
    def _init() -> Gen9RLEnvironment:
        # Create unique identity for this worker
        timestamp = int(time.time() * 1000) % 100000
        worker_id = f"W{rank}_{timestamp}"
        
        # Import locally to avoid pickling issues
        from poke_env.environment import SingleAgentWrapper
        from poke_env import AccountConfiguration
        from src.curriculum_agents.opponent_pool import OpponentPoolManager
        from src.curriculum_agents.win_tracker import WinRateTracker
        from src.environment import CurriculumWrapper
        from stable_baselines3.common.monitor import Monitor
        from src.teams.random_battle_teambuilder import RandomBattleTeambuilder
        
        # Create local Pool Manager with pokemon_data for random teams
        pool = OpponentPoolManager(
            checkpoint_dir=checkpoint_dir,
            battle_format=battle_format,
            pokemon_data=pokemon_data,
        )
        
        # STAGGERED STARTUP FIX
        # Prevent thundering herd problem on local Pokemon Showdown server
        # Sleep proportional to rank to ensure connections happen sequentially
        if rank > 0:
            delay = rank * 2.0  # 2 seconds per rank
            print(f"Worker {rank}: Staggering startup by {delay}s...")
            time.sleep(delay)
        
        # Create random team builder for training agent
        training_teambuilder = RandomBattleTeambuilder(pokemon_data=pokemon_data)
        
        # Create local WinRateTracker (read/write to shared file)
        # Note: Race conditions on write are possible but acceptable for stats
        win_tracker_path = Path(checkpoint_dir) / "win_tracker.json"
        tracker = WinRateTracker(str(win_tracker_path))
        
        # =====================================================
        # TRAINING MODE SELECTION (v12 Ablation)
        # =====================================================
        training_mode = os.environ.get("TRAINING_MODE", "baseline")
        rigorous_mode = os.environ.get("RIGOROUS_MODE", "0") == "1"
        
        if training_mode == "selfplay" and not pool.checkpoint_pool.is_empty and random.random() < 0.30:
            # Self-play mode: 30% chance of playing against frozen checkpoint
            from poke_env.player import RandomPlayer
            
            sp_opponent = pool.create_self_play_opponent(pokemon_data)
            if sp_opponent:
                opponent = sp_opponent
                agent_type = "frozen"
                tier = "SELFPLAY"
                print(f"Worker {rank}: [SELFPLAY] Using frozen checkpoint")
            else:
                # Fallback to random if self-play fails
                opp_id = f"Opp{rank}x{timestamp}"
                opponent = RandomPlayer(
                    battle_format=battle_format,
                    account_configuration=AccountConfiguration(f"Rand{opp_id}", None),
                    team=training_teambuilder.yield_team()
                )
                agent_type = "random_player"
                tier = "BASELINE"
        elif rigorous_mode or training_mode in ["baseline", "selfplay"]:
            # Baseline/Selfplay: Use only poke-env baseline opponents
            from poke_env.player import RandomPlayer, MaxBasePowerPlayer, SimpleHeuristicsPlayer
            
            opponent_choice = rank % 3
            opp_id = f"Opp{rank}x{timestamp}"
            
            if opponent_choice == 0:
                opponent = RandomPlayer(
                    battle_format=battle_format,
                    account_configuration=AccountConfiguration(f"Rand{opp_id}", None),
                    team=training_teambuilder.yield_team()
                )
                agent_type = "random_player"
            elif opponent_choice == 1:
                opponent = MaxBasePowerPlayer(
                    battle_format=battle_format,
                    account_configuration=AccountConfiguration(f"MaxD{opp_id}", None),
                    team=training_teambuilder.yield_team()
                )
                agent_type = "max_damage"
            else:
                opponent = SimpleHeuristicsPlayer(
                    battle_format=battle_format,
                    account_configuration=AccountConfiguration(f"Heur{opp_id}", None),
                    team=training_teambuilder.yield_team()
                )
                agent_type = "heuristic"
            
            tier = "BASELINE"
            print(f"Worker {rank}: [{training_mode.upper()}] Using {agent_type}")
        else:
            # Curriculum mode: Time-based opponent unlocking
            opponent, tier, agent_type = pool.sample_opponent(
                progress=current_progress,
                win_tracker=tracker,
                pokemon_data=pokemon_data
            )
        
        # Create environment with unique account
        # In RIGOROUS_MODE with gen9randombattle, don't use teambuilder (auto-generated)
        env = Gen9RLEnvironment(
            pokemon_data=pokemon_data,
            battle_format=battle_format,
            account_configuration1=AccountConfiguration(worker_id, None),
            teambuilder=training_teambuilder,
        )
        
        # Skip team syncing in rigorous mode (random battles auto-generate)
        if not rigorous_mode and opponent and hasattr(opponent, '_team'):
            if hasattr(env, 'agent2'):
                env.agent2._team = opponent._team
                print(f"Worker {rank}: Synced agent2 team with {agent_type}")
            else:
                 print(f"Worker {rank}: Warning - Could not find agent2 to sync team")
        
        # Wrap with opponent
        wrapped = SingleAgentWrapper(env, opponent=opponent)
        
        # Add Monitor for logging
        # We assume rank 0 logs to main, others to subdirs?
        # Actually SB3 creates monitor files.
        if log_dir:
            wrapped = Monitor(wrapped, str(log_dir / f"worker_{rank}"))
        else:
            wrapped = Monitor(wrapped)
        
        # Wrap with CurriculumWrapper - pass opponent_type for win rate tracking
        # Also pass pool and tracker path for dynamic resampling
        wrapped = CurriculumWrapper(
            wrapped, 
            opponent_type=agent_type,
            opponent_pool=pool,
            win_tracker_path=str(win_tracker_path),
            pokemon_data=pokemon_data
        )
        
        return wrapped
    
    return _init


def main():
    args = parse_args()
    
    # Set seed
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
    
    # Dry run adjustments
    if args.dry_run:
        args.timesteps = 1000
        args.iteration_steps = 100
        args.n_steps = 32
        args.batch_size = 16
        args.n_envs = 1  # Force single env for dry run
        args.no_subproc = True # Force sequential for debugging
    
    # Create directories with experiment name
    checkpoint_dir = Path(args.checkpoint_dir) / args.experiment_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir) / args.experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Pokemon data
    print(f"Loading Pokemon data from {args.pokemon_data}...")
    pokemon_data = load_pokemon_data(args.pokemon_data)
    print(f"Loaded data for {len(pokemon_data)} Pokemon")
    
    # === CREATE VECTORIZED ENVIRONMENT ===
    n_envs = args.n_envs
    print(f"\nCreating {n_envs} parallel environment(s)...")
    
    # We use parallel envs by default now for efficiency
    # But if n_envs=1 and no-subproc, use Dummy
    
    # Initialize Win Rate Tracker (Main Process)
    win_tracker_path = checkpoint_dir / "win_tracker.json"
    win_tracker = WinRateTracker(str(win_tracker_path))
    
    env_fns = [
        make_parallel_env(
            rank=i,
            pokemon_data=pokemon_data,
            battle_format=args.battle_format,
            checkpoint_dir=str(checkpoint_dir),
            current_progress=0.0,
            log_dir=log_dir,
        )
        for i in range(n_envs)
    ]
    
    if n_envs == 1 and args.no_subproc:
         print("Using DummyVecEnv (sequential)")
         train_env = DummyVecEnv(env_fns)
    elif args.no_subproc:
         print("Using DummyVecEnv (sequential, multiple)")
         train_env = DummyVecEnv(env_fns)
    else:
         print("Using SubprocVecEnv (parallel)")
         train_env = SubprocVecEnv(env_fns, start_method='spawn')
         
    use_parallel = True
    
    print(f"Observation space: {train_env.observation_space.shape}")
    print(f"Action space: {train_env.action_space}")
    
    # Initialize ELO tracker for self-play tracking
    elo_tracker = EloTracker(str(checkpoint_dir))
    
    # Initialize Pool Manager (Main Process - for checkpointing)
    from src.curriculum_agents.opponent_pool import OpponentPoolManager
    pool_manager = OpponentPoolManager(
        checkpoint_dir=str(checkpoint_dir),
        battle_format=args.battle_format,
        pokemon_data=pokemon_data
    )
    
    print(f"Win Rate Tracker: {win_tracker.get_summary()}")
    
    # Create or load model
    from sb3_contrib import RecurrentPPO
    
    # Prepare LR schedule - Use Wang's Inverse Decay by default now
    lr_schedule = get_lr_schedule(args.lr, "wang")
    
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        model = RecurrentPPO.load(args.resume, env=train_env)
        # Hack to update learning rate on resumed model if different
        pass 
    else:
        print("Creating new RecurrentPPO model...")

        # Prepare Model kwargs
        policy_kwargs = {
            "lstm_hidden_size": 256,
            "n_lstm_layers": 1,
            "net_arch": dict(pi=[64, 64], vf=[64, 64])
        }
        
        # ABLATION OVERRIDE: Allow changing net arch via env var
        ablation_policy = os.environ.get("ABLATION_POLICY_KWARGS")
        if ablation_policy:
            try:
                override_kwargs = json.loads(ablation_policy)
                # If explicit list is passed (e.g., [256, 256]), map it to dict expected by SB3
                if "net_arch" in override_kwargs:
                    if isinstance(override_kwargs["net_arch"], list):
                         override_kwargs["net_arch"] = dict(pi=override_kwargs["net_arch"], vf=override_kwargs["net_arch"])
                
                policy_kwargs.update(override_kwargs)
                print(f"!!! ABLATION OVERRIDE: Using Custom Policy Kwargs: {policy_kwargs} !!!")
            except json.JSONDecodeError:
                print(f"Error parsing ABLATION_POLICY_KWARGS: {ablation_policy}")

        model = RecurrentPPO(
            "MlpLstmPolicy",
            train_env,
            learning_rate=lr_schedule,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=0.754,  # Wang's
            clip_range=0.0829, # Wang's
            ent_coef=0.0588,   # Wang's
            vf_coef=0.4375,    # Wang's
            max_grad_norm=0.543, # Wang's
            verbose=1,
            tensorboard_log=str(log_dir),
            policy_kwargs=policy_kwargs
        )
    
    # Training loop
    effective_fps = 60 * n_envs  # Estimated
    eta_hours = args.timesteps / effective_fps / 3600
    print(f"\nStarting training for {args.timesteps:,} total timesteps...")
    print(f"Estimated time: ~{eta_hours:.1f} hours with {n_envs} env(s)")
    
    
    # Initialize progress from model if resuming
    total_timesteps = 0
    iteration = 0
    
    if args.resume:
        total_timesteps = model.num_timesteps
        iteration = total_timesteps // args.iteration_steps
        print(f"RESUMING TRAINING from Step {total_timesteps} (Iteration {iteration})")
    
    # Callbacks
    win_rate_callback = WinRateCallback(win_tracker)
    sampling_callback = OpponentSamplingCallback(verbose=1)
    # Rebuilding callback list - removing smart metrics print since we do it here
    current_callback = CallbackList([TensorboardCallback(), win_rate_callback, sampling_callback])
    
    # Inject global total for progress calculation (model.total_timesteps resets per learn() call)
    model._global_total_steps = args.timesteps
    
    while total_timesteps < args.timesteps:
        iteration += 1
        steps_this_iter = min(args.iteration_steps, args.timesteps - total_timesteps)
        
        # === CURRICULUM UPDATE ===
        current_progress = min(1.0, total_timesteps / args.timesteps)
        
        # Update LR schedule global state (if using our custom schedule)
        if hasattr(lr_schedule, '_global_state'):
            lr_schedule._global_state['steps_done'] = total_timesteps
        
        # Broadcast progress to workers
        try:
            train_env.env_method("set_progress", current_progress)
        except Exception:
            pass # Use pass instead of warning to avoid log spam if unsupported
        # =========================

        # === CURRICULUM STATUS DISPLAY ===
        try:
            # 1. Active Agents Distribution (What workers are actually fighting)
            # Query envs for '_opponent_type' via get_attr (works on wrappers)
            # Query envs for '_opponent_type' via get_wrapper_attr (works on wrappers)
            try:
                active_opponents = train_env.env_method("get_wrapper_attr", "_opponent_type")
            except AttributeError:
                active_opponents = None
            # Fallback if get_wrapper_attr is not supported in this env version
            if not active_opponents or active_opponents[0] is None:
                 active_opponents = train_env.get_attr("_opponent_type")
                 
            # Filter out Nones
            active_opponents = [o for o in active_opponents if o]
            
            if active_opponents:
                from collections import Counter
                counts = Counter(active_opponents)
                total_workers = len(active_opponents)
                
                print(f"\n--- Curriculum Status ({total_workers} Active Workers) ---")
                
                # Active Agents Table
                print(f"{'Active Agent':<25} | {'Count':<6} | {'Ratio':<6}")
                print("-" * 45)
                sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
                for agent, count in sorted_counts:
                    pct = count / total_workers
                    print(f"{agent:<25} | {count:<6} | {pct:<6.1%}")
                print("-" * 45)
            
            # 2. Target Tier Distribution (Probabilities)
            # Using our new pool_manager method
            if pool_manager:
                tier_weights = pool_manager.get_tier_distribution(current_progress, win_tracker)
                print(f"\nTarget Sampling Ratios:")
                for tier, weight in tier_weights.items():
                    if weight > 0.01: # Only show relevant tiers
                        print(f"  - {tier.upper():<10}: {weight:.1%}")

        except Exception as e:
            # print(f"Warning: Could not fetch curriculum status: {e}")
            pass
        # =================================

        # Display Stats
        print(f"\n==========================================================")
        print(f"=== Iteration {iteration} (Step {total_timesteps}/{args.timesteps}) ===")
        print(f"==========================================================")
        
        # Calculate Phase
        phase = win_tracker.get_current_phase(current_progress)
        phase_color = "\033[92m" if phase == 'early' else ("\033[93m" if phase == 'mid' else "\033[91m")
        reset_color = "\033[0m"
        
        print(f"Phase:    {phase_color}{phase.upper()}{reset_color}")
        
        # Calculate current LR
        if callable(model.learning_rate):
            current_lr = model.learning_rate(1.0 - current_progress)
        else:
            current_lr = float(model.learning_rate)
        print(f"Progress: {current_progress:.2%} (LR: {current_lr:.2e})")

        # Display Reward Config for current phase
        print(f"\n--- Reward Config ({phase.upper()}) ---")
        param_str = []
        # Get config from dictionary
        current_rewards = REWARD_CURRICULUM.get(phase, REWARD_CURRICULUM['early'])
        
        # Select key params to display
        key_params = ['w_fainted', 'w_hp', 'w_matchup', 'w_hazards', 'victory_bonus', 'switch_tax']
        for k in key_params:
            if k in current_rewards:
                param_str.append(f"{k}: {current_rewards[k]}")
        
        print(", ".join(param_str))
        
        # Reload win tracker to get latest stats from workers
        win_tracker._load()
        
        # Print Win Rates Table
        print(f"\n--- Opponent Mastery (Window: {win_tracker.WINDOW_SIZE}) ---")
        stats_sorted = sorted(win_tracker.stats.items(), key=lambda x: x[0])
        
        # Updated Headers for Total WR coverage
        print(f"{'Opponent':<20} | {'Recent WR':<10} | {'Total WR':<10} | {'Games (Cov)':<12} | {'Status'}")
        print("-" * 75)
        
        for agent, stat in stats_sorted:
            # Skip if no games
            if stat.total_games == 0: continue
            
            # Determine status based on phase requirements
            marker = " "
            if stat.recent_games >= win_tracker.WINDOW_SIZE:
                if stat.recent_win_rate >= win_tracker.WIN_RATE_THRESHOLD:
                    marker = "✅"
                elif stat.recent_win_rate < 0.2:
                    marker = "⚠️"
            
            # Format: Agent | 60.0% | 45.0% | 20/500 | ✅
            print(f"{agent:<20} | {stat.recent_win_rate:<10.1%} | {stat.total_win_rate:<10.1%} | {stat.recent_games}/{stat.total_games:<4} | {marker}")
        print("-" * 75 + "\n")
        
        print(f"Training for {steps_this_iter} steps...")
        
        # Train
        model.learn(
            total_timesteps=steps_this_iter,
            reset_num_timesteps=False,
            progress_bar=True,
            callback=current_callback
        )
        
        # CRITICAL FIX: Sync with actual model steps to prevent drift
        # SB3 rounds up to nearest n_steps * n_envs
        total_timesteps = model.num_timesteps
        
        # Reload win tracker to get latest stats from workers (updated during learn)
        win_tracker._load()
        print(f"Win Rates: {win_tracker.get_summary()}")
        
        # Log Win Rates to Tensorboard
        # Check if logger exists (it should after learn)
        if hasattr(model, 'logger'):
            for agent_name, stats in win_tracker.stats.items():
                if stats.total_games > 0:
                    model.logger.record(f"curriculum/winrate_{agent_name}", stats.recent_win_rate)
                    model.logger.record(f"curriculum/games_{agent_name}", stats.total_games)
            model.logger.dump(step=total_timesteps)
        
        # Save checkpoint
        checkpoint_path = checkpoint_dir / f"model_{iteration}.zip"
        model.save(str(checkpoint_path))
        print(f"Saved checkpoint: {checkpoint_path}")
        
        # Notify pool manager of new snapshot (for self-play)
        pool_manager.checkpoint_pool.add_snapshot(model, iteration)
        
        # === OPPONENT SWAPPING ===
        # Handled by OpponentSamplingCallback via resample_opponent()
        # No need to rebuild environments here.
    
    # Save final model
    final_path = checkpoint_dir / "model_final.zip"
    model.save(str(final_path))
    pool_manager.checkpoint_pool.add_snapshot(model, iteration + 1)
    
    # Reload stats one last time
    win_tracker._load()
    print(f"Final Win Rates: {win_tracker.get_summary()}")
    print(f"\nTraining complete! Final model saved to: {final_path}")
    
    train_env.close()


if __name__ == "__main__":
    main()
