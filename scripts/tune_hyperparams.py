#!/usr/bin/env python3
"""
Hyperparameter tuning script using Optuna for Bayesian optimization.

Tunes key PPO hyperparameters by running short training trials and
measuring win rate against SimpleHeuristicsPlayer.

Usage:
    python scripts/tune_hyperparams.py --n-trials 50 --timesteps-per-trial 100000
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import optuna
    from optuna.trial import Trial
except ImportError:
    print("Optuna not installed. Run: pip install optuna")
    sys.exit(1)

from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from src.environment import Gen9RLEnvironment
from src.utils import load_pokemon_data
from src.curriculum_agents.win_tracker import WinRateTracker
from src.teams.random_battle_teambuilder import RandomBattleTeambuilder


def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning with Optuna")
    
    parser.add_argument("--n-trials", type=int, default=30,
                        help="Number of Optuna trials to run")
    parser.add_argument("--timesteps-per-trial", type=int, default=100_000,
                        help="Training timesteps per trial (shorter = faster but noisier)")
    parser.add_argument("--eval-episodes", type=int, default=50,
                        help="Number of games to evaluate each trial")
    parser.add_argument("--study-name", type=str, default="pokemon_ppo_study",
                        help="Name of the Optuna study")
    parser.add_argument("--storage", type=str, default=None,
                        help="Optuna storage URL (e.g., sqlite:///optuna.db). Default: in-memory")
    parser.add_argument("--pokemon-data", type=str, default="gen9randombattle.json",
                        help="Path to Pokemon data JSON")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    
    return parser.parse_args()


def create_env(pokemon_data: Dict, battle_format: str = "gen9randombattle") -> DummyVecEnv:
    """Create a single training environment for tuning."""
    timestamp = int(time.time() * 1000) % 100000
    worker_id = f"Tuner_{timestamp}"
    
    # gen9randombattle auto-generates teams, don't pass custom teams
    opponent = SimpleHeuristicsPlayer(
        battle_format=battle_format,
        account_configuration=AccountConfiguration(f"TuneOpp_{timestamp}", None),
    )
    
    env = Gen9RLEnvironment(
        pokemon_data=pokemon_data,
        battle_format=battle_format,
        account_configuration1=AccountConfiguration(worker_id, None),
    )
    
    from poke_env.environment import SingleAgentWrapper
    wrapped = SingleAgentWrapper(env, opponent=opponent)
    wrapped = Monitor(wrapped)
    
    return DummyVecEnv([lambda: wrapped])


def evaluate_model(model, pokemon_data: Dict, n_episodes: int = 50) -> float:
    """
    Evaluate model win rate against SimpleHeuristicsPlayer.
    
    Returns:
        Win rate as float between 0 and 1
    """
    from sb3_contrib import RecurrentPPO
    from poke_env import AccountConfiguration
    from poke_env.environment import SingleAgentWrapper
    
    timestamp = int(time.time() * 1000) % 100000
    
    # gen9randombattle auto-generates teams
    opponent = SimpleHeuristicsPlayer(
        battle_format="gen9randombattle",
        account_configuration=AccountConfiguration(f"EvalOpp_{timestamp}", None),
    )
    
    env = Gen9RLEnvironment(
        pokemon_data=pokemon_data,
        battle_format="gen9randombattle",
        account_configuration1=AccountConfiguration(f"EvalAgent_{timestamp}", None),
    )
    
    wrapped = SingleAgentWrapper(env, opponent=opponent)
    
    wins = 0
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)
    
    for ep in range(n_episodes):
        obs, _ = wrapped.reset()
        done = False
        
        while not done:
            action, lstm_states = model.predict(
                obs, 
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=True
            )
            obs, reward, terminated, truncated, info = wrapped.step(action)
            done = terminated or truncated
            episode_starts = np.array([done])
        
        # Check if we won
        if hasattr(wrapped, 'current_battle') and wrapped.current_battle:
            if wrapped.current_battle.won:
                wins += 1
        elif reward > 0:
            wins += 1
    
    wrapped.close()
    return wins / n_episodes


def objective(trial: Trial, args, pokemon_data: Dict) -> float:
    """
    Optuna objective function.
    
    Samples hyperparameters, trains for a short period, and returns win rate.
    """
    from sb3_contrib import RecurrentPPO
    
    # Sample hyperparameters
    hyperparams = {
        # Discount factor - critical for sparse rewards
        "gamma": trial.suggest_float("gamma", 0.99, 0.9999, log=True),
        
        # GAE lambda - bias/variance tradeoff
        "gae_lambda": trial.suggest_float("gae_lambda", 0.5, 0.99),
        
        # PPO clipping
        "clip_range": trial.suggest_float("clip_range", 0.05, 0.3),
        
        # Entropy coefficient - exploration
        "ent_coef": trial.suggest_float("ent_coef", 0.001, 0.1, log=True),
        
        # Value function coefficient
        "vf_coef": trial.suggest_float("vf_coef", 0.1, 0.9),
        
        # Learning rate
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        
        # Max gradient norm
        "max_grad_norm": trial.suggest_float("max_grad_norm", 0.3, 1.0),
        
        # Batch size (power of 2)
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512, 1024]),
        
        # Number of epochs
        "n_epochs": trial.suggest_int("n_epochs", 3, 10),
        
        # Steps per rollout
        "n_steps": trial.suggest_categorical("n_steps", [512, 1024, 2048]),
    }
    
    # Network architecture
    net_arch_choice = trial.suggest_categorical("net_arch", ["small", "medium", "large"])
    if net_arch_choice == "small":
        net_arch = dict(pi=[128, 128], vf=[128, 128])
    elif net_arch_choice == "medium":
        net_arch = dict(pi=[256, 256], vf=[256, 256])
    else:
        net_arch = dict(pi=[512, 256], vf=[512, 256])
    
    lstm_hidden_size = trial.suggest_categorical("lstm_hidden_size", [128, 256, 512])
    
    print(f"\n=== Trial {trial.number} ===")
    print(f"Hyperparameters: {json.dumps(hyperparams, indent=2)}")
    print(f"Network: {net_arch_choice}, LSTM: {lstm_hidden_size}")
    
    try:
        # Create environment
        env = create_env(pokemon_data)
        
        # Create model with sampled hyperparameters
        model = RecurrentPPO(
            "MlpLstmPolicy",
            env,
            learning_rate=hyperparams["learning_rate"],
            n_steps=hyperparams["n_steps"],
            batch_size=hyperparams["batch_size"],
            n_epochs=hyperparams["n_epochs"],
            gamma=hyperparams["gamma"],
            gae_lambda=hyperparams["gae_lambda"],
            clip_range=hyperparams["clip_range"],
            ent_coef=hyperparams["ent_coef"],
            vf_coef=hyperparams["vf_coef"],
            max_grad_norm=hyperparams["max_grad_norm"],
            verbose=0,
            policy_kwargs={
                "net_arch": net_arch,
                "lstm_hidden_size": lstm_hidden_size,
            }
        )
        
        # Train
        print(f"Training for {args.timesteps_per_trial:,} timesteps...")
        model.learn(total_timesteps=args.timesteps_per_trial, progress_bar=True)
        
        # Evaluate
        print(f"Evaluating over {args.eval_episodes} games...")
        win_rate = evaluate_model(model, pokemon_data, args.eval_episodes)
        
        print(f"Trial {trial.number} Win Rate: {win_rate:.2%}")
        
        env.close()
        return win_rate
        
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        return 0.0  # Return worst score on failure


def main():
    args = parse_args()
    
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
    
    print("=" * 60)
    print("Pokemon RL Hyperparameter Tuning with Optuna")
    print("=" * 60)
    print(f"Trials: {args.n_trials}")
    print(f"Timesteps per trial: {args.timesteps_per_trial:,}")
    print(f"Eval episodes: {args.eval_episodes}")
    print()
    
    # Load Pokemon data
    pokemon_data = load_pokemon_data(args.pokemon_data)
    print(f"Loaded data for {len(pokemon_data)} Pokemon")
    
    # Create or load study
    if args.storage:
        study = optuna.create_study(
            study_name=args.study_name,
            direction="maximize",
            storage=args.storage,
            load_if_exists=True
        )
        print(f"Using persistent storage: {args.storage}")
    else:
        study = optuna.create_study(
            study_name=args.study_name,
            direction="maximize"
        )
        print("Using in-memory storage (results not persisted)")
    
    # Run optimization
    print(f"\nStarting optimization...")
    study.optimize(
        lambda trial: objective(trial, args, pokemon_data),
        n_trials=args.n_trials,
        show_progress_bar=True
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    
    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best win rate: {study.best_value:.2%}")
    print(f"\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save results
    results_path = Path("checkpoints") / "optuna_results.json"
    results_path.parent.mkdir(exist_ok=True)
    
    results = {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "n_trials": len(study.trials),
        "all_trials": [
            {"number": t.number, "value": t.value, "params": t.params}
            for t in study.trials if t.value is not None
        ]
    }
    
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    
    # Print command to use best params
    print("\n" + "=" * 60)
    print("To train with best hyperparameters, update train.py or use:")
    print("=" * 60)
    best = study.best_params
    print(f"""
python scripts/train.py \\
    --gamma {best.get('gamma', 0.9999):.6f} \\
    --lr {best.get('learning_rate', 1e-4):.2e} \\
    --n-steps {best.get('n_steps', 2048)} \\
    --batch-size {best.get('batch_size', 512)}
""")


if __name__ == "__main__":
    main()
