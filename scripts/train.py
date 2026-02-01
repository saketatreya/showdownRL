"""
Training script for Gen 9 Random Battle RL bot.
Uses RecurrentPPO with pure self-play vs historical checkpoints (Wang 2024 methodology).
"""

import argparse
import gc
import json
import random
import time
import os
import csv
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List, Tuple
import traceback

import numpy as np

from poke_env.player import RandomPlayer
from poke_env import AccountConfiguration

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment import Gen9RLEnvironment
from src.trained_player import TrainedPlayer
from src.utils import load_pokemon_data
from src.elo_tracker import EloTracker
from src.callbacks import EloCallback, TensorboardCallback, StdoutWinRateCallback
from src.config import TrainingConfig
from src.teams.random_battle_teambuilder import RandomBattleTeambuilder
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from sb3_contrib import RecurrentPPO
from src.models.maskable_lstm_policy import MaskableRecurrentActorCriticPolicy

def hard_close_vec_env(env) -> None:
    """
    Close a VecEnv and aggressively release file descriptors.

    Why: stable-baselines3's SubprocVecEnv closes the worker processes but does not
    explicitly close the parent-side Pipe connections. When we recreate SubprocVecEnv
    repeatedly (fictitious self-play resampling), those connections can accumulate and
    eventually hit the OS open-files limit (Errno 24).
    """
    if env is None:
        return
    try:
        env.close()
    finally:
        # SubprocVecEnv keeps parent pipe ends open; close them explicitly.
        for attr in ("remotes", "work_remotes"):
            conns = getattr(env, attr, None)
            if not conns:
                continue
            for conn in conns:
                try:
                    conn.close()
                except Exception:
                    pass
        # Encourage timely cleanup of multiprocessing Connection objects.
        gc.collect()

def _extract_model_index(path: Path) -> int:
    # checkpoint filename pattern: model_<N>.zip
    stem = path.stem
    try:
        return int(stem.split("_", 1)[1])
    except Exception:
        return -1

def _list_checkpoints(checkpoint_dir: Path) -> List[Path]:
    checkpoints = sorted(checkpoint_dir.glob("model_*.zip"), key=_extract_model_index)
    return [p for p in checkpoints if _extract_model_index(p) >= 0]

def _select_evenly_spaced(items: List[str], k: int) -> List[str]:
    if k <= 0 or not items:
        return []
    if len(items) <= k:
        return list(items)
    if k == 1:
        return [items[-1]]
    idxs = [round(i * (len(items) - 1) / (k - 1)) for i in range(k)]
    # Deduplicate while preserving order
    out: List[str] = []
    for i in idxs:
        name = items[int(i)]
        if name not in out:
            out.append(name)
    return out

def _load_or_create_eval_set(checkpoint_dir: Path, n_checkpoints: int) -> List[str]:
    """
    Fixed eval set of historical checkpoints for stable benchmarking over time.
    Stored on disk so resumes keep the same opponents.
    """
    path = checkpoint_dir / "eval_set.json"
    if path.exists():
        try:
            with open(path, "r") as f:
                data = json.load(f)
            checkpoints = data.get("checkpoints", [])
            if isinstance(checkpoints, list):
                return [str(x) for x in checkpoints]
        except Exception:
            pass

    existing = [p.name for p in _list_checkpoints(checkpoint_dir)]
    chosen = _select_evenly_spaced(existing, n_checkpoints)
    try:
        with open(path, "w") as f:
            json.dump(
                {
                    "created_at_unix": time.time(),
                    "checkpoints": chosen,
                },
                f,
                indent=2,
            )
    except Exception:
        pass
    return chosen

def _build_opponent_player(
    *,
    opponent_id: str,
    battle_format: str,
    pokemon_data: dict,
    checkpoint_path: Optional[Path] = None,
) -> Tuple[str, Any]:
    """
    Build an opponent policy used only for `choose_move()` inside SingleAgentWrapper.
    IMPORTANT: `start_listening=False` avoids creating extra websocket clients.
    """
    is_random_format = "random" in battle_format.lower()

    if checkpoint_path is None:
        opponent = RandomPlayer(
            battle_format=battle_format,
            account_configuration=AccountConfiguration(opponent_id, None),
            start_listening=False,
            team=None if is_random_format else RandomBattleTeambuilder(pokemon_data=pokemon_data).yield_team(),
        )
        return "RandomPlayer", opponent

    model = RecurrentPPO.load(str(checkpoint_path), device="cpu")
    # Safety: ensure checkpoint matches current env spaces (prevents runtime crashes)
    expected_obs_dim = 1163
    expected_action_dim = 26
    if getattr(model.observation_space, "shape", None) != (expected_obs_dim,):
        raise ValueError(
            f"Incompatible checkpoint obs space {model.observation_space}; expected shape ({expected_obs_dim},)"
        )
    if getattr(model.action_space, "n", None) != expected_action_dim:
        raise ValueError(f"Incompatible checkpoint action space {model.action_space}; expected Discrete({expected_action_dim})")

    opponent = TrainedPlayer(
        model=model,
        battle_format=battle_format,
        account_configuration=AccountConfiguration(opponent_id, None),
        start_listening=False,
        team=None if is_random_format else RandomBattleTeambuilder(pokemon_data=pokemon_data).yield_team(),
    )
    return checkpoint_path.name, opponent

def _fixed_eval(
    *,
    model: RecurrentPPO,
    pokemon_data: dict,
    battle_format: str,
    checkpoint_dir: Path,
    checkpoint_names: List[str],
    n_eval_episodes: int,
    deterministic: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate current model vs a fixed opponent set and return per-opponent metrics.
    Writes `fixed_eval_history.csv` and records scalars to the SB3 logger.
    """
    from poke_env.environment import SingleAgentWrapper

    # Create one eval env and reuse it, swapping opponents in-place.
    eval_env_id = f"Eval{int(time.time() * 1000) % 100000}"
    is_random_format = "random" in battle_format.lower()
    base_env = Gen9RLEnvironment(
        pokemon_data=pokemon_data,
        battle_format=battle_format,
        account_configuration1=AccountConfiguration(eval_env_id, None),
        teambuilder=None if is_random_format else RandomBattleTeambuilder(pokemon_data=pokemon_data),
    )

    # Start with RandomPlayer; will swap later.
    _, rand_opp = _build_opponent_player(
        opponent_id=f"{eval_env_id}_Rand",
        battle_format=battle_format,
        pokemon_data=pokemon_data,
        checkpoint_path=None,
    )
    base_env.external_opponent_id = "RandomPlayer"
    wrapped = Monitor(SingleAgentWrapper(base_env, opponent=rand_opp))

    results: Dict[str, Dict[str, float]] = {}

    def eval_against(opponent_label: str, opponent_player) -> Dict[str, float]:
        # Swap opponent
        try:
            wrapped.env.opponent = opponent_player
        except Exception:
            pass
        try:
            wrapped.env.env.external_opponent_id = opponent_label
        except Exception:
            pass

        episode_rewards, _episode_lengths = evaluate_policy(
            model,
            wrapped,
            n_eval_episodes=int(n_eval_episodes),
            deterministic=bool(deterministic),
            return_episode_rewards=True,
            warn=False,
        )
        wins = sum(1 for r in episode_rewards if r > 0.5)
        losses = sum(1 for r in episode_rewards if r < -0.5)
        draws = int(len(episode_rewards) - wins - losses)
        n = max(1, len(episode_rewards))
        win_rate = wins / n
        draw_rate = draws / n
        avg_return = float(np.mean(episode_rewards)) if episode_rewards else 0.0
        return {
            "episodes": float(n),
            "wins": float(wins),
            "losses": float(losses),
            "draws": float(draws),
            "win_rate": float(win_rate),
            "draw_rate": float(draw_rate),
            "avg_return": float(avg_return),
        }

    # Always evaluate vs RandomPlayer
    results["RandomPlayer"] = eval_against("RandomPlayer", rand_opp)

    # Fixed checkpoints
    for name in checkpoint_names:
        path = checkpoint_dir / name
        if not path.exists():
            continue
        try:
            opp_label, opp = _build_opponent_player(
                opponent_id=f"{eval_env_id}_{name}",
                battle_format=battle_format,
                pokemon_data=pokemon_data,
                checkpoint_path=path,
            )
            results[opp_label] = eval_against(opp_label, opp)
        except Exception as e:
            print(f"[Eval] Skipping {name}: {e}")

    # Persist to CSV for easy plotting
    csv_path = checkpoint_dir / "fixed_eval_history.csv"
    write_header = not csv_path.exists()
    try:
        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(
                    [
                        "timesteps",
                        "iteration",
                        "opponent_id",
                        "episodes",
                        "wins",
                        "losses",
                        "draws",
                        "win_rate",
                        "draw_rate",
                        "avg_return",
                        "timestamp_unix",
                    ]
                )
            for opp_id, m in results.items():
                w.writerow(
                    [
                        int(getattr(model, "num_timesteps", 0)),
                        int(getattr(model, "_iteration", -1)),
                        opp_id,
                        int(m.get("episodes", 0)),
                        int(m.get("wins", 0)),
                        int(m.get("losses", 0)),
                        int(m.get("draws", 0)),
                        float(m.get("win_rate", 0.0)),
                        float(m.get("draw_rate", 0.0)),
                        float(m.get("avg_return", 0.0)),
                        time.time(),
                    ]
                )
    except Exception:
        pass

    try:
        wrapped.close()
    except Exception:
        pass
    gc.collect()

    return results

def parse_args():
    parser = argparse.ArgumentParser()
    default_config = TrainingConfig()
    
    parser.add_argument("--timesteps", type=int, default=10_000_000)
    parser.add_argument("--iteration-steps", type=int, default=32768)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--pokemon-data", type=str, default="gen9randombattle.json")
    parser.add_argument("--battle-format", type=str, default="gen9randombattle")
    parser.add_argument("--lr", type=float, default=default_config.learning_rate)
    parser.add_argument("--lr-schedule", type=str, default="wang")
    parser.add_argument("--n-steps", type=int, default=default_config.n_steps)
    parser.add_argument("--batch-size", type=int, default=default_config.batch_size)
    parser.add_argument("--n-epochs", type=int, default=default_config.n_epochs)
    parser.add_argument("--gamma", type=float, default=0.9999)
    parser.add_argument("--experiment-name", type=str, default="selfplay")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--no-subproc", action="store_true")
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--eval-checkpoints", type=int, default=4)
    parser.add_argument("--eval-every-iters", type=int, default=1)
    
    return parser.parse_args()


def get_lr_schedule(
    initial_lr: float,
    total_steps: int,
    schedule_type: str = "linear",
) -> Callable[[float], float]:
    global_state = {"steps_done": 0, "total_steps": int(total_steps)}
    
    if schedule_type == "linear":
        def func(progress_remaining: float) -> float:
            x = global_state['steps_done'] / global_state['total_steps']
            return initial_lr * (1.0 - x)
        func._global_state = global_state
        return func
    elif schedule_type == "wang":
        def func(progress_remaining: float) -> float:
            x = global_state['steps_done'] / global_state['total_steps']
            return initial_lr / ((8 * x + 1) ** 1.5)
        func._global_state = global_state
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
    log_dir: Optional[Path] = None,
) -> Callable:
    def _init() -> Gen9RLEnvironment:
        timestamp = int(time.time() * 1000) % 100000
        worker_id = f"W{rank}_{timestamp}"
        
        if rank > 0:
            delay = rank * 2.0
            print(f"Worker {rank}: Staggering startup by {delay}s...")
            time.sleep(delay)
            
        from poke_env.environment import SingleAgentWrapper
        # If we are doing random battles, we MUST let the server generate the teams
        # Providing a team manually causes Showdown to reject the battle request.
        is_random_format = "random" in battle_format.lower()
        
        # Self-Play: Pick a random checkpoint, fallback to RandomPlayer
        checkpoints = []
        if os.path.exists(checkpoint_dir):
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".zip")]
            
        opponent_label = "RandomPlayer"

        if checkpoints and random.random() < 0.70: # 70% Self-Play
            ckpt_name = random.choice(checkpoints)
            ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
            try:
                opp_id = f"SP{rank}x{timestamp}"
                opponent_label, opponent = _build_opponent_player(
                    opponent_id=opp_id,
                    battle_format=battle_format,
                    pokemon_data=pokemon_data,
                    checkpoint_path=Path(ckpt_path),
                )
                print(f"Worker {rank}: Self-play against {ckpt_name}")
            except Exception as e:
                print(f"Worker {rank}: Failed to load {ckpt_path}, falling back to Random")
                opp_id = f"Rand{rank}x{timestamp}"
                opponent_label, opponent = _build_opponent_player(
                    opponent_id=opp_id,
                    battle_format=battle_format,
                    pokemon_data=pokemon_data,
                    checkpoint_path=None,
                )
        else:
            opp_id = f"Rand{rank}x{timestamp}"
            opponent_label, opponent = _build_opponent_player(
                opponent_id=opp_id,
                battle_format=battle_format,
                pokemon_data=pokemon_data,
                checkpoint_path=None,
            )
            print(f"Worker {rank}: Playing against RandomPlayer")
        
        env = Gen9RLEnvironment(
            pokemon_data=pokemon_data,
            battle_format=battle_format,
            account_configuration1=AccountConfiguration(worker_id, None),
            teambuilder=None if is_random_format else RandomBattleTeambuilder(pokemon_data=pokemon_data),
        )

        # Expose opponent identity via env.get_additional_info() for callbacks (ELO, logging, etc.)
        env.external_opponent_id = opponent_label
        
        if hasattr(opponent, '_team') and hasattr(env, 'agent2'):
            env.agent2._team = opponent._team
            
        wrapped = SingleAgentWrapper(env, opponent=opponent)
        
        if log_dir:
            wrapped = Monitor(wrapped, str(log_dir / f"worker_{rank}"))
        else:
            wrapped = Monitor(wrapped)
            
        return wrapped
    
    return _init


def main():
    args = parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
        
    if args.dry_run:
        args.timesteps = 1000
        args.iteration_steps = 100
        args.n_steps = 32
        args.batch_size = 16
        args.n_envs = 1
        args.no_subproc = True
        
    checkpoint_dir = Path(args.checkpoint_dir) / args.experiment_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir) / args.experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading Pokemon data from {args.pokemon_data}...")
    pokemon_data = load_pokemon_data(args.pokemon_data)
    
    n_envs = args.n_envs
    
    def create_env():
        print(f"\nCreating {n_envs} parallel environment(s) for Fictitious Self-Play...")
        env_fns = [
            make_parallel_env(
                rank=i,
                pokemon_data=pokemon_data,
                battle_format=args.battle_format,
                checkpoint_dir=str(checkpoint_dir),
                log_dir=log_dir,
            )
            for i in range(n_envs)
        ]
        
        if args.no_subproc:
             return DummyVecEnv(env_fns)
        else:
             return SubprocVecEnv(env_fns)

    train_env = create_env()
         
    elo_tracker = EloTracker(str(checkpoint_dir))
    
    lr_schedule = get_lr_schedule(args.lr, args.timesteps, args.lr_schedule)
    
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        model = RecurrentPPO.load(args.resume, env=train_env)
    else:
        print("Creating new RecurrentPPO model...")
        policy_kwargs = {
            "lstm_hidden_size": 256,
            "n_lstm_layers": 1,
            "net_arch": dict(pi=[64, 64], vf=[64, 64])
        }
        
        model = RecurrentPPO(
            MaskableRecurrentActorCriticPolicy,
            train_env,
            learning_rate=lr_schedule,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=0.754,
            clip_range=0.0829,
            ent_coef=0.0588,
            vf_coef=0.4375,
            max_grad_norm=0.543,
            verbose=1,
            tensorboard_log=str(log_dir),
            policy_kwargs=policy_kwargs
        )
        
    total_timesteps = 0
    iteration = 0
    if args.resume:
        total_timesteps = model.num_timesteps
        iteration = total_timesteps // args.iteration_steps
        
    current_callback = CallbackList([
        TensorboardCallback(), 
        EloCallback(elo_tracker, current_id="current"),
        StdoutWinRateCallback(max_opponents=5, elo_tracker=elo_tracker, current_id="current"),
    ])
    
    model._global_total_steps = args.timesteps
    
    is_first_iteration = True
    
    while total_timesteps < args.timesteps:
        iteration += 1
        steps_this_iter = min(args.iteration_steps, args.timesteps - total_timesteps)
        
        # Resample Fictitious Self-Play Opponents
        if not is_first_iteration:
            print("\n[Self-Play] Tearing down old environments and resampling opponents...")
            hard_close_vec_env(train_env)
            train_env = create_env()
            model.set_env(train_env)
        is_first_iteration = False
            
        if hasattr(lr_schedule, '_global_state'):
            lr_schedule._global_state['steps_done'] = total_timesteps
            
        print(f"\n==========================================================")
        print(f"=== Iteration {iteration} (Step {total_timesteps}/{args.timesteps}) ===")
        print(f"==========================================================")
        
        if callable(model.learning_rate):
            current_lr = model.learning_rate(1.0 - (total_timesteps / args.timesteps))
        else:
            current_lr = float(model.learning_rate)
        print(f"Progress: {total_timesteps/args.timesteps:.2%} (LR: {current_lr:.2e})")
        
        model.learn(
            total_timesteps=steps_this_iter,
            reset_num_timesteps=False,
            progress_bar=True,
            callback=current_callback
        )
        
        total_timesteps = model.num_timesteps
        
        checkpoint_path = checkpoint_dir / f"model_{iteration}.zip"
        model.save(str(checkpoint_path))
        print(f"Saved checkpoint: {checkpoint_path}")

        # Fixed evaluation set: win-rate over time vs stable opponents.
        if (not args.no_eval) and (args.eval_every_iters > 0) and (iteration % args.eval_every_iters == 0) and (not args.dry_run):
            # Persist iteration for CSV rows
            setattr(model, "_iteration", iteration)
            eval_ckpts = _load_or_create_eval_set(checkpoint_dir, int(args.eval_checkpoints))
            if eval_ckpts:
                print(f"[Eval] Fixed set: RandomPlayer + {len(eval_ckpts)} checkpoint(s)")
            else:
                print(f"[Eval] Fixed set: RandomPlayer only (no checkpoints yet)")

            eval_results = _fixed_eval(
                model=model,
                pokemon_data=pokemon_data,
                battle_format=args.battle_format,
                checkpoint_dir=checkpoint_dir,
                checkpoint_names=eval_ckpts,
                n_eval_episodes=int(args.eval_episodes),
                deterministic=True,
            )

            # Print to stdout
            # Aggregate across opponents (weighted by episodes)
            total_eps = sum(int(m.get("episodes", 0)) for m in eval_results.values())
            total_w = sum(int(m.get("wins", 0)) for m in eval_results.values())
            total_l = sum(int(m.get("losses", 0)) for m in eval_results.values())
            total_d = sum(int(m.get("draws", 0)) for m in eval_results.values())
            agg_win = (total_w / total_eps) if total_eps else 0.0
            agg_ret = ((total_w - total_l) / total_eps) if total_eps else 0.0
            print(
                f"[Eval] fixed_set: episodes={total_eps} W={total_w} L={total_l} D={total_d} | "
                f"win%={agg_win:.1%} avg_return={agg_ret:+.3f}"
            )
            for opp_id, m in eval_results.items():
                print(
                    f"[Eval] vs {opp_id}: n={int(m['episodes'])} "
                    f"W={int(m['wins'])} L={int(m['losses'])} D={int(m['draws'])} | "
                    f"win%={m['win_rate']:.1%} avg_return={m['avg_return']:+.3f}"
                )

            # Record to tensorboard
            import re

            def _tb_safe(s: str) -> str:
                return re.sub(r"[^0-9a-zA-Z_-]", "_", str(s))

            model.logger.record("fixed_eval/agg_win_rate", float(agg_win))
            model.logger.record("fixed_eval/agg_avg_return", float(agg_ret))
            for opp_id, m in eval_results.items():
                safe = _tb_safe(opp_id)
                model.logger.record(f"fixed_eval/win_rate/{safe}", float(m.get("win_rate", 0.0)))
                model.logger.record(f"fixed_eval/avg_return/{safe}", float(m.get("avg_return", 0.0)))
            model.logger.dump(step=int(model.num_timesteps))
        
    final_path = checkpoint_dir / "model_final.zip"
    model.save(str(final_path))
    print(f"\nTraining complete! Final model saved to: {final_path}")
    hard_close_vec_env(train_env)

if __name__ == "__main__":
    main()
