"""
Bayesian belief tracker for opponent Pokemon.
Tracks probability distributions over roles, moves, items, abilities, and tera types.
"""

from typing import Dict, List, Optional, Set, Tuple, Any
import numpy as np
from collections import defaultdict

from .utils import normalize_species_name


class BeliefTracker:
    """
    Tracks beliefs about opponent Pokemon based on observed information.
    Uses Bayesian updates to maintain probability distributions.
    """
    
    def __init__(self, pokemon_data: Dict[str, Any]):
        """
        Initialize the belief tracker.
        
        Args:
            pokemon_data: Dictionary from gen9randombattle.json containing
                         role/move/item data for each species.
        """
        self.pokemon_data = pokemon_data
        # Normalize species names for lookup
        self._normalized_data = {}
        for species, data in pokemon_data.items():
            normalized = normalize_species_name(species)
            self._normalized_data[normalized] = data
            self._normalized_data[species] = data  # Keep original too
        
        # Per-pokemon beliefs: species -> BeliefState
        self.beliefs: Dict[str, 'PokemonBelief'] = {}
    
    def reset(self):
        """Reset all beliefs for a new battle."""
        self.beliefs.clear()
    
    def get_or_create_belief(self, species: str) -> 'PokemonBelief':
        """Get existing belief or create new one for a species."""
        normalized = normalize_species_name(species)
        
        if normalized not in self.beliefs:
            # Look up species data
            species_data = self._normalized_data.get(normalized, None)
            self.beliefs[normalized] = PokemonBelief(species, species_data)
        
        return self.beliefs[normalized]
    
    def update(self, species: str, 
               observed_move: Optional[str] = None,
               observed_item: Optional[str] = None,
               observed_ability: Optional[str] = None,
               observed_tera: Optional[str] = None):
        """
        Update beliefs based on newly observed information.
        
        Args:
            species: Pokemon species name
            observed_move: Move that was just used
            observed_item: Item that was revealed
            observed_ability: Ability that was revealed
            observed_tera: Tera type that was revealed
        """
        belief = self.get_or_create_belief(species)
        
        if observed_move:
            belief.observe_move(observed_move)
        if observed_item:
            belief.observe_item(observed_item)
        if observed_ability:
            belief.observe_ability(observed_ability)
        if observed_tera:
            belief.observe_tera(observed_tera)
    
    def get_belief_embedding(self, species: str, embedding_size: int = 10) -> np.ndarray:
        """
        Get fixed-size embedding vector for a Pokemon's beliefs.
        
        Args:
            species: Pokemon species name
            embedding_size: Size of the output vector
            
        Returns:
            numpy array of shape (embedding_size,)
        """
        belief = self.get_or_create_belief(species)
        return belief.to_embedding(embedding_size)


class PokemonBelief:
    """Belief state for a single opponent Pokemon."""
    
    # Maximum number of roles/moves/items to track
    MAX_ROLES = 4
    MAX_MOVES = 6
    MAX_ITEMS = 4
    
    def __init__(self, species: str, species_data: Optional[Dict[str, Any]]):
        """
        Initialize belief for a specific Pokemon.
        
        Args:
            species: Pokemon species name
            species_data: Data from gen9randombattle.json, or None if unknown
        """
        self.species = species
        self.species_data = species_data
        
        # Observed information
        self.observed_moves: Set[str] = set()
        self.move_pp_usage: Dict[str, int] = defaultdict(int)  # Track usage count
        self.observed_item: Optional[str] = None
        self.observed_ability: Optional[str] = None
        self.observed_tera: Optional[str] = None
        
        # Initialize role probabilities (uniform over available roles)
        self.role_probs: Dict[str, float] = {}
        if species_data and 'roles' in species_data:
            n_roles = len(species_data['roles'])
            for role in species_data['roles']:
                self.role_probs[role] = 1.0 / n_roles
        
        # Cache computed probabilities
        self._move_probs_cache: Optional[Dict[str, float]] = None
        self._item_probs_cache: Optional[Dict[str, float]] = None
    
    def observe_move(self, move: str):
        """
        Update beliefs after observing a move.
        Also increments PP usage counter.
        """
        move_lower = move.lower().replace(" ", "").replace("-", "")
        
        # Track usage regardless of whether it's new
        self.move_pp_usage[move_lower] += 1
        
        # Idempotency check: If we've already processed this move, don't re-update probabilities
        if move_lower in self.observed_moves:
            return
            
        self.observed_moves.add(move_lower)
        self._move_probs_cache = None  # Invalidate cache
        
        if not self.species_data or 'roles' not in self.species_data:
            return
        
        # Bayesian update: P(role | move) ∝ P(move | role) * P(role)
        new_probs = {}
        epsilon = 1e-9
        
        for role, role_data in self.species_data['roles'].items():
            role_moves = [m.lower().replace(" ", "").replace("-", "") 
                         for m in role_data.get('moves', [])]
            
            p_move_given_role = epsilon
            if move_lower in role_moves:
                # If role has N moves, probability of seeing this specific one 
                # assumes opponent picks from their pool. 
                # Simplification: Uniform choice from pool? 
                # Or just binary compatibility?
                # Binary compatibility is safer for "possibility" tracking.
                # If we use 1/len(moves), we harshly penalize roles with large movepools.
                # Let's stick to "Is this move possible for this role?".
                # If yes, likelihood = 1.0. If no, likelihood = epsilon.
                p_move_given_role = 1.0
            
            new_probs[role] = self.role_probs.get(role, 0.0) * p_move_given_role
        
        # Renormalize
        total = sum(new_probs.values())
        if total > 0:
            self.role_probs = {r: p / total for r, p in new_probs.items()}
        else:
            # If total is 0, it means the move is impossible for ALL known roles.
            # Fallback: Reset to uniform to recover from data error.
            n_roles = len(self.species_data['roles'])
            self.role_probs = {r: 1.0/n_roles for r in self.species_data['roles']}
    
    def observe_item(self, item: str):
        """Update beliefs after observing an item."""
        if self.observed_item == item:
            return
            
        self.observed_item = item
        self._item_probs_cache = None
        
        if not self.species_data or 'roles' not in self.species_data:
            return
        
        item_lower = item.lower().replace(" ", "").replace("-", "")
        
        # Update role probabilities based on item
        new_probs = {}
        epsilon = 1e-9
        
        for role, role_data in self.species_data['roles'].items():
            role_items = [i.lower().replace(" ", "").replace("-", "") 
                         for i in role_data.get('items', [])]
            
            p_item_given_role = epsilon
            if item_lower in role_items:
                p_item_given_role = 1.0
            elif not role_items:
                # If role has no specific items listed, any item is possible?
                # Usually data lists specific items. Assume permissive if empty?
                # No, data usually complete.
                pass
            
            new_probs[role] = self.role_probs.get(role, 0.0) * p_item_given_role
        
        # Renormalize
        total = sum(new_probs.values())
        if total > 0:
            self.role_probs = {r: p / total for r, p in new_probs.items()}
        else:
             # Fallback
            n_roles = len(self.species_data['roles'])
            self.role_probs = {r: 1.0/n_roles for r in self.species_data['roles']}
    
    def observe_ability(self, ability: str):
        """Update beliefs after observing an ability."""
        if self.observed_ability == ability:
            return
            
        self.observed_ability = ability
        
        if not self.species_data or 'roles' not in self.species_data:
            return
        
        ability_lower = ability.lower().replace(" ", "").replace("-", "")
        
        # Update role probabilities
        new_probs = {}
        epsilon = 1e-9
        
        for role, role_data in self.species_data['roles'].items():
            role_abilities = [a.lower().replace(" ", "").replace("-", "") 
                             for a in role_data.get('abilities', [])]
            
            p_abil_given_role = epsilon
            if ability_lower in role_abilities:
                p_abil_given_role = 1.0
            
            new_probs[role] = self.role_probs.get(role, 0.0) * p_abil_given_role
        
        # Renormalize
        total = sum(new_probs.values())
        if total > 0:
            self.role_probs = {r: p / total for r, p in new_probs.items()}
        else:
            # Fallback
            n_roles = len(self.species_data['roles'])
            self.role_probs = {r: 1.0/n_roles for r in self.species_data['roles']}
    
    def observe_tera(self, tera_type: str):
        """Update beliefs after observing tera type."""
        self.observed_tera = tera_type
    
    def is_move_possible(self, move_name: str) -> bool:
        """
        Check if a move is listed in any known role for this species.
        """
        if not self.species_data or 'roles' not in self.species_data:
            return False
            
        move_clean = move_name.lower().replace(" ", "").replace("-", "")
        
        for role_data in self.species_data['roles'].values():
            moves = [m.lower().replace(" ", "").replace("-", "") for m in role_data.get('moves', [])]
            if move_clean in moves:
                return True
        return False

    def get_unrevealed_move_probs(self) -> Dict[str, float]:
        """Get probability distribution over unrevealed moves."""
        if self._move_probs_cache is not None:
            return self._move_probs_cache
        
        move_probs: Dict[str, float] = defaultdict(float)
        
        if not self.species_data or 'roles' not in self.species_data:
            return {}
        
        for role, role_prob in self.role_probs.items():
            if role not in self.species_data['roles']:
                continue
            role_data = self.species_data['roles'][role]
            role_moves = role_data.get('moves', [])
            
            for move in role_moves:
                move_lower = move.lower().replace(" ", "").replace("-", "")
                if move_lower not in self.observed_moves:
                    # Weight by role probability
                    move_probs[move] += role_prob / len(role_moves)
        
        # Normalize
        total = sum(move_probs.values())
        if total > 0:
            move_probs = {m: p / total for m, p in move_probs.items()}
        
        self._move_probs_cache = dict(move_probs)
        return self._move_probs_cache
    
    def get_item_probs(self) -> Dict[str, float]:
        """Get probability distribution over items (if not revealed)."""
        if self.observed_item:
            return {self.observed_item: 1.0}
        
        if self._item_probs_cache is not None:
            return self._item_probs_cache
        
        item_probs: Dict[str, float] = defaultdict(float)
        
        if not self.species_data or 'roles' not in self.species_data:
            return {}
        
        for role, role_prob in self.role_probs.items():
            if role not in self.species_data['roles']:
                continue
            role_data = self.species_data['roles'][role]
            role_items = role_data.get('items', [])
            
            for item in role_items:
                item_probs[item] += role_prob / len(role_items) if role_items else 0
        
        # Normalize
        total = sum(item_probs.values())
        if total > 0:
            item_probs = {i: p / total for i, p in item_probs.items()}
        
        self._item_probs_cache = dict(item_probs)
        return self._item_probs_cache
    
    def get_role_entropy(self) -> float:
        """
        Calculate Shannon entropy of the role probability distribution.
        Returns:
            Entropy value in bits (log2). Higher = more uncertainty.
        """
        if not self.role_probs:
            return 0.0
            
        entropy = 0.0
        for prob in self.role_probs.values():
            if prob > 0:
                entropy -= prob * np.log2(prob)
        return entropy
    
    def to_embedding(self, size: int = 10) -> np.ndarray:
        """
        Convert beliefs to fixed-size embedding vector.
        
        Structure (for size=10):
        - [0:4] Top 4 role probabilities
        - [4:8] Top 4 unrevealed move probabilities  
        - [8] Has item been revealed (0/1)
        - [9] Number of observed moves / 4
        """
        embedding = np.zeros(size, dtype=np.float32)
        
        # Top role probabilities
        if self.role_probs:
            sorted_roles = sorted(self.role_probs.values(), reverse=True)
            for i, prob in enumerate(sorted_roles[:4]):
                if i < size:
                    embedding[i] = prob
        
        # Top unrevealed move probabilities
        move_probs = self.get_unrevealed_move_probs()
        if move_probs:
            sorted_moves = sorted(move_probs.values(), reverse=True)
            for i, prob in enumerate(sorted_moves[:4]):
                if 4 + i < size:
                    embedding[4 + i] = prob
        
        # Item revealed flag
        if 8 < size:
            embedding[8] = 1.0 if self.observed_item else 0.0
        
        # Observed moves fraction
        if 9 < size:
            embedding[9] = len(self.observed_moves) / 4.0
        
        return embedding
