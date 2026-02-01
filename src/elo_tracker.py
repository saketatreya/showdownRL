"""
ELO rating system for self-play opponent selection.
Tracks skill ratings of model checkpoints for better training progression.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple
import math


class EloTracker:
    """
    Tracks ELO ratings for checkpoints to improve self-play opponent selection.
    
    Prioritizes opponents near current skill level for optimal learning.
    """
    
    DEFAULT_ELO = 1000
    K_FACTOR = 32  # ELO K-factor
    
    def __init__(self, checkpoint_dir: str):
        """
        Initialize ELO tracker.
        
        Args:
            checkpoint_dir: Directory containing checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.elo_file = self.checkpoint_dir / "elo_ratings.json"
        self.ratings: Dict[str, float] = {}
        self.match_history: list = []
        
        self._load()
    
    def _load(self):
        """Load ELO ratings from disk."""
        if self.elo_file.exists():
            try:
                with open(self.elo_file, 'r') as f:
                    data = json.load(f)
                    self.ratings = data.get('ratings', {})
                    self.match_history = data.get('history', [])
            except:
                self.ratings = {}
                self.match_history = []
    
    def _save(self):
        """Save ELO ratings to disk."""
        self.checkpoint_dir.mkdir(exist_ok=True)
        with open(self.elo_file, 'w') as f:
            json.dump({
                'ratings': self.ratings,
                'history': self.match_history[-100:]  # Keep last 100 matches
            }, f, indent=2)
    
    def get_rating(self, checkpoint_name: str) -> float:
        """Get ELO rating for a checkpoint."""
        return self.ratings.get(checkpoint_name, self.DEFAULT_ELO)
    
    def update_from_battle(
        self, 
        player1: str, 
        player2: str, 
        player1_won: bool
    ):
        """
        Update ELO ratings after a battle.
        
        Args:
            player1: Name of first player/checkpoint
            player2: Name of second player/checkpoint  
            player1_won: True if player1 won
        """
        r1 = self.get_rating(player1)
        r2 = self.get_rating(player2)
        
        # Calculate expected scores
        e1 = 1 / (1 + math.pow(10, (r2 - r1) / 400))
        e2 = 1 - e1
        
        # Actual scores
        s1 = 1.0 if player1_won else 0.0
        s2 = 1 - s1
        
        # Update ratings
        self.ratings[player1] = r1 + self.K_FACTOR * (s1 - e1)
        self.ratings[player2] = r2 + self.K_FACTOR * (s2 - e2)
        
        # Record match
        self.match_history.append({
            'p1': player1,
            'p2': player2,
            'winner': player1 if player1_won else player2
        })
        
        self._save()
    
    def get_best_opponent(
        self, 
        current_checkpoint: str,
        available_checkpoints: list,
        margin: float = 100
    ) -> Optional[str]:
        """
        Select best opponent based on ELO ratings.
        
        Prefers opponents within `margin` ELO of current skill.
        
        Args:
            current_checkpoint: Current model checkpoint name
            available_checkpoints: List of available opponent checkpoints
            margin: ELO margin for opponent selection
            
        Returns:
            Best opponent checkpoint name, or None if no suitable opponent
        """
        if not available_checkpoints:
            return None
        
        current_elo = self.get_rating(current_checkpoint)
        
        # Score opponents by how close they are to current ELO
        scored = []
        for cp in available_checkpoints:
            if cp == current_checkpoint:
                continue
            cp_elo = self.get_rating(cp)
            distance = abs(cp_elo - current_elo)
            
            # Prefer slightly stronger opponents (to learn from)
            if cp_elo > current_elo:
                distance *= 0.8  # Discount distance to stronger opponents
            
            scored.append((cp, distance))
        
        if not scored:
            return None
        
        # Sort by distance and return closest
        scored.sort(key=lambda x: x[1])
        return scored[0][0]
    
    def get_stats(self) -> Dict:
        """Get summary statistics."""
        if not self.ratings:
            return {'n_checkpoints': 0, 'n_matches': 0}
        
        ratings = list(self.ratings.values())
        return {
            'n_checkpoints': len(self.ratings),
            'n_matches': len(self.match_history),
            'min_elo': min(ratings),
            'max_elo': max(ratings),
            'avg_elo': sum(ratings) / len(ratings),
        }
