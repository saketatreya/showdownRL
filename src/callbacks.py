from stable_baselines3.common.callbacks import BaseCallback
from src.elo_tracker import EloTracker
import numpy as np
import time
from collections import defaultdict

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

class StdoutWinRateCallback(BaseCallback):
    """
    Prints episodic win/loss/draw stats to stdout during training.

    This is more informative than `ep_rew_mean` during self-play because it
    provides an interpretable outcome metric and an opponent breakdown.
    """

    def __init__(
        self,
        max_opponents: int = 5,
        elo_tracker: EloTracker | None = None,
        current_id: str = "current",
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.max_opponents = int(max_opponents)
        self.elo_tracker = elo_tracker
        self.current_id = str(current_id)
        self._reset_window()

    def _reset_window(self) -> None:
        self._t0 = time.time()
        self._episodes = 0
        self._wins = 0
        self._losses = 0
        self._draws = 0
        self._by_opponent = defaultdict(lambda: {"w": 0, "l": 0, "d": 0, "n": 0})

    def _on_training_start(self) -> None:
        self._reset_window()

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        if dones is None:
            terminated = self.locals.get("terminated")
            truncated = self.locals.get("truncated")
            if terminated is not None and truncated is not None:
                dones = np.logical_or(terminated, truncated)
            elif terminated is not None:
                dones = terminated
            else:
                return True

        infos = self.locals.get("infos", [])
        for i, done in enumerate(dones):
            if not bool(done):
                continue
            if i >= len(infos):
                continue
            info = infos[i] or {}
            if not isinstance(info, dict):
                continue
            if "error" in info:
                continue

            # Determine outcome
            result = info.get("result")
            if result not in {"win", "loss", "draw"}:
                # Fallback: some wrappers only provide battle_won
                if "battle_won" in info:
                    result = "win" if bool(info["battle_won"]) else "loss"
                else:
                    continue

            opponent_id = str(info.get("opponent_id", "unknown"))

            self._episodes += 1
            opp = self._by_opponent[opponent_id]
            opp["n"] += 1

            if result == "win":
                self._wins += 1
                opp["w"] += 1
            elif result == "loss":
                self._losses += 1
                opp["l"] += 1
            else:
                self._draws += 1
                opp["d"] += 1

        return True

    def _on_training_end(self) -> None:
        if self._episodes <= 0:
            return

        elapsed = max(0.0, time.time() - self._t0)
        win_rate = self._wins / self._episodes
        draw_rate = self._draws / self._episodes
        avg_return = (self._wins - self._losses) / self._episodes

        print(
            f"[WinRate] episodes={self._episodes} "
            f"W={self._wins} L={self._losses} D={self._draws} | "
            f"win%={win_rate:.1%} draw%={draw_rate:.1%} | "
            f"avg_return={avg_return:+.3f} | "
            f"elapsed={elapsed:.0f}s"
        )
        if self.elo_tracker is not None:
            try:
                elo = float(self.elo_tracker.get_rating(self.current_id))
                print(f"[WinRate] elo[{self.current_id}]={elo:.1f}")
            except Exception:
                pass

        # Print top opponents by games played in this window
        if self.max_opponents > 0 and self._by_opponent:
            top = sorted(self._by_opponent.items(), key=lambda kv: kv[1]["n"], reverse=True)[: self.max_opponents]
            for opponent_id, s in top:
                n = s["n"]
                if n <= 0:
                    continue
                opp_wr = s["w"] / n
                opp_dr = s["d"] / n
                opp_ret = (s["w"] - s["l"]) / n
                print(
                    f"[WinRate] vs {opponent_id}: "
                    f"n={n} W={s['w']} L={s['l']} D={s['d']} | "
                    f"win%={opp_wr:.1%} draw%={opp_dr:.1%} | "
                    f"avg_return={opp_ret:+.3f}"
                )

class EloCallback(BaseCallback):
    """
    Callback to update ELO ratings after each battle.
    """
    
    def __init__(
        self, 
        elo_tracker: EloTracker,
        current_id: str = "current",
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.elo_tracker = elo_tracker
        self.current_id = current_id
        
    def _on_step(self) -> bool:
        """
        Called after each step. Checks for battle completion.
        """
        dones = self.locals.get("dones")
        if dones is None:
            terminated = self.locals.get("terminated")
            truncated = self.locals.get("truncated")
            if terminated is not None and truncated is not None:
                dones = np.logical_or(terminated, truncated)
            elif terminated is not None:
                dones = terminated
            else:
                return True

        infos = self.locals.get("infos", [])
        
        for i, done in enumerate(dones):
            if not bool(done):
                continue
            if i >= len(infos):
                continue

            info = infos[i] or {}
            if not isinstance(info, dict):
                continue
            if "error" in info:
                continue

            opponent_id = info.get("opponent_id")
            if not opponent_id:
                continue

            if info.get("result") == "draw":
                score = 0.5
            elif "battle_won" in info:
                score = 1.0 if bool(info["battle_won"]) else 0.0
            elif info.get("result") in {"win", "loss"}:
                score = 1.0 if info["result"] == "win" else 0.0
            else:
                continue

            if str(opponent_id) == str(self.current_id):
                continue

            self.elo_tracker.update_from_battle(
                player1=self.current_id,
                player2=str(opponent_id),
                player1_score=score,
            )

            if self.verbose > 0:
                result = "D" if score == 0.5 else ("W" if score > 0.5 else "L")
                battle_tag = info.get("battle_tag", "?")
                print(f"[ELO] {self.current_id} vs {opponent_id}: {result} ({battle_tag})")
                    
        return True
