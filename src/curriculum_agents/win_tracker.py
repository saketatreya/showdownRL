"""
Win Rate Tracker for curriculum progression gating.
Tracks win rates against each curriculum agent using a rolling window.
Saves state to disk for persistence across restarts.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from collections import deque


class AgentStats:
    """Stats for a single curriculum agent with rolling window."""
    
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        # Rolling window of recent results (True = win, False = loss)
        self.recent_results: deque = deque(maxlen=window_size)
        # Total counts for all-time stats
        self.total_wins: int = 0
        self.total_losses: int = 0
    
    @property
    def total_games(self) -> int:
        return self.total_wins + self.total_losses
    
    @property
    def total_win_rate(self) -> float:
        """All-time win rate."""
        if self.total_games == 0:
            return 0.0
        return self.total_wins / self.total_games
    
    @property
    def recent_games(self) -> int:
        """Number of games in rolling window."""
        return len(self.recent_results)
    
    @property
    def recent_win_rate(self) -> float:
        """Win rate in rolling window (last N games)."""
        if len(self.recent_results) == 0:
            return 0.0
        return sum(self.recent_results) / len(self.recent_results)
    
    def record(self, won: bool):
        """Record a game result."""
        self.recent_results.append(won)
        if won:
            self.total_wins += 1
        else:
            self.total_losses += 1
    
    def to_dict(self) -> dict:
        """Serialize for JSON storage."""
        # Convert bools to ints for JSON compatibility (numpy bool is not JSON serializable)
        return {
            'total_wins': int(self.total_wins),
            'total_losses': int(self.total_losses),
            'recent_results': [int(r) for r in self.recent_results],
        }
    
    @classmethod
    def from_dict(cls, data: dict, window_size: int = 20) -> 'AgentStats':
        """Deserialize from JSON."""
        stats = cls(window_size)
        stats.total_wins = data.get('total_wins', 0)
        stats.total_losses = data.get('total_losses', 0)
        # Restore recent results
        recent = data.get('recent_results', [])
        for result in recent[-window_size:]:  # Only keep last window_size
            stats.recent_results.append(result)
        return stats


class WinRateTracker:
    """
    Tracks win rates against each curriculum agent using a rolling window.
    THREAD-SAFE & MULTI-PROCESS SAFE VERSION (using fcntl).
    
    Progression requires BOTH:
    - Time-based progress threshold met (e.g., 30% of training for MID)
    - 60% win rate in last 20 games against ALL agents in current phase
    """
    
    # Which agents must be beaten for each phase transition
    PHASE_REQUIREMENTS = {
        'early': ['max_damage', 'type_punisher'],
        'mid': ['revenge_killer', 'sacrifice_trader', 'hazard_stacker'],
        'late': ['setup_sweeper', 'priority_sniper', 'pivot_spammer'],
    }
    
    # Time thresholds (must ALSO meet these)
    TIME_THRESHOLDS = {
        'early': 0.0,   # Can start EARLY immediately
        'mid': 0.30,    # Need 30% progress to enter MID
        'late': 0.70,   # Need 70% progress to enter LATE
    }
    
    WIN_RATE_THRESHOLD = 0.60  # 60% required in rolling window
    WINDOW_SIZE = 20  # Rolling window size
    
    def __init__(self, save_path: str):
        """Initialize tracker with save path."""
        import fcntl
        self.save_path = Path(save_path)
        self.stats: Dict[str, AgentStats] = {}
        self._current_opponent: Optional[str] = None
        self._fcntl = fcntl # Store module ref
        
        # Initial load (read-only, no lock needed mostly, but good practice)
        self._load()
    
    def _load(self):
        """Load stats from disk (Read-Only access)."""
        if self.save_path.exists():
            try:
                # We use a shared lock for reading if we want to be super safe,
                # but for just reading stats for display/curriculum check, 
                # slight staleness is acceptable. 
                # However, to prevent reading garbage during a write, we use shared lock.
                with open(self.save_path, 'r') as f:
                    try:
                        self._fcntl.flock(f, self._fcntl.LOCK_SH)
                        data = json.load(f)
                    finally:
                        self._fcntl.flock(f, self._fcntl.LOCK_UN)
                        
                    for agent_name, stats_dict in data.items():
                        self.stats[agent_name] = AgentStats.from_dict(
                            stats_dict, self.WINDOW_SIZE
                        )
                # print(f"Loaded win rate tracker from {self.save_path}")
            except Exception as e:
                print(f"Warning: Could not load win tracker: {e}")
    
    def record_result(self, agent_name: str, won: bool):
        """
        Record a game result Atomically.
        1. Lock file.
        2. Read latest state.
        3. Update.
        4. Write state.
        5. Unlock.
        """
        try:
            # Ensure directory exists
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Open file in 'a+' mode to create if invalid, but we need to read it.
            # Best pattern: Open, Lock, Seek(0), Read, Seek(0), Truncate, Write.
            mode = 'r+' if self.save_path.exists() else 'w+'
            
            with open(self.save_path, mode) as f:
                # EXCLUSIVE LOCK - blocks other processes
                self._fcntl.flock(f, self._fcntl.LOCK_EX)
                
                try:
                    # READ current state
                    f.seek(0)
                    try:
                        content = f.read()
                        if content:
                            data = json.loads(content)
                        else:
                            data = {}
                    except json.JSONDecodeError:
                        data = {} # corrupted or empty
                    
                    # Parse into temp stats
                    current_stats = {}
                    for name, stats_dict in data.items():
                        current_stats[name] = AgentStats.from_dict(stats_dict, self.WINDOW_SIZE)
                        
                    # UPDATE target agent
                    if agent_name not in current_stats:
                        current_stats[agent_name] = AgentStats(self.WINDOW_SIZE)
                    
                    current_stats[agent_name].record(won)
                    
                    # Update local cache too while we are at it
                    self.stats = current_stats
                    
                    # WRITE back
                    save_data = {
                        agent: stats.to_dict()
                        for agent, stats in current_stats.items()
                    }
                    
                    f.seek(0)
                    f.truncate()
                    json.dump(save_data, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno()) # Force write to disk
                    
                finally:
                    self._fcntl.flock(f, self._fcntl.LOCK_UN)
                    
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to atomic record result for {agent_name}: {e}")

    def get_stats(self, agent_name: str) -> AgentStats:
        """Get stats for an agent (from local cache)."""
        if agent_name not in self.stats:
            self.stats[agent_name] = AgentStats(self.WINDOW_SIZE)
        return self.stats[agent_name]
    
    def needs_more_games(self, agent_name: str) -> bool:
        """Check if we need more games against this agent."""
        return self.get_stats(agent_name).recent_games < self.WINDOW_SIZE
    
    def get_agents_needing_games(self, phase: str) -> List[str]:
        """Get list of agents in phase that need more games."""
        agents = self.PHASE_REQUIREMENTS.get(phase, [])
        return [a for a in agents if self.needs_more_games(a)]
    
    def phase_win_rate_met(self, phase: str) -> bool:
        """
        Check if win rate requirement is met for a phase.
        All agents in phase must have >= threshold win rate in rolling window.
        """
        agents = self.PHASE_REQUIREMENTS.get(phase, [])
        for agent in agents:
            stats = self.get_stats(agent)
            if stats.recent_games < self.WINDOW_SIZE:
                return False
            if stats.recent_win_rate < self.WIN_RATE_THRESHOLD:
                return False
        return True
    
    def can_progress_to(self, target_phase: str, current_progress: float) -> bool:
        """
        Check if we can progress to target phase.
        Requires BOTH time threshold AND win rate threshold.
        """
        # Check time threshold
        time_threshold = self.TIME_THRESHOLDS.get(target_phase, 1.0)
        if current_progress < time_threshold:
            return False
        
        # For entering a phase, we need to have beaten the PREVIOUS phase's agents
        if target_phase == 'mid':
            return self.phase_win_rate_met('early')
        elif target_phase == 'late':
            return self.phase_win_rate_met('mid')
        else:  # early
            return True
    
    def get_current_phase(self, current_progress: float) -> str:
        """Determine current phase based on progress and win rates."""
        # Check late first (highest requirement)
        if self.can_progress_to('late', current_progress):
            return 'late'
        # Check mid
        if self.can_progress_to('mid', current_progress):
            return 'mid'
        # Default to early
        return 'early'
    
    def get_summary(self) -> str:
        """Get a summary string for display."""
        parts = []
        for agent, stats in self.stats.items():
            short_name = agent.replace('_', '')[:8]
            parts.append(f"{short_name}={stats.recent_win_rate:.0%}({stats.recent_games}g)")
        return ", ".join(parts) if parts else "No data yet"
    
    def get_phase_summary(self, phase: str) -> str:
        """Get summary for agents in a phase, showing both recent and total."""
        agents = self.PHASE_REQUIREMENTS.get(phase, [])
        parts = []
        for agent in agents:
            stats = self.get_stats(agent)
            # Check if requirement met (60% in last 20)
            if stats.recent_games >= self.WINDOW_SIZE and stats.recent_win_rate >= self.WIN_RATE_THRESHOLD:
                marker = "✓"
            else:
                marker = "✗"
            # Show: agent=RW:X%(wins/window) Total:Y%(wins/total)
            recent_wins = sum(stats.recent_results)
            total_wr = stats.total_win_rate
            short_str = f"{agent}=RW:{stats.recent_win_rate:.0%}({recent_wins}/{stats.recent_games}) Total:{total_wr:.0%}({stats.total_games}){marker}"
            parts.append(short_str)
        return ", ".join(parts) if parts else "N/A"
    
    def set_current_opponent(self, agent_type: str):
        """Track which opponent we're currently fighting."""
        self._current_opponent = agent_type
    
    @property
    def current_opponent(self) -> Optional[str]:
        return self._current_opponent
