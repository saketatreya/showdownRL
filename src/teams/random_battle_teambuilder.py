"""
Random Battle Teambuilder: Generates random-battle-style teams from gen9randombattle.json.

This allows us to use gen9anythinggoes (or gen9curriculumbattle) format with
teams that mimic the variety and balance of actual random battles.
"""

import random
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

from poke_env.teambuilder import Teambuilder


class RandomBattleTeambuilder(Teambuilder):
    """
    Generates random-battle-style teams from gen9randombattle.json.
    
    Each team has 6 Pokemon with:
    - Random role from available roles
    - Items/abilities/moves from that role
    - Level from the JSON
    - EVs/IVs from the JSON
    """
    
    def __init__(self, pokemon_data: Optional[Dict[str, Any]] = None, data_path: str = "gen9randombattle.json"):
        """
        Args:
            pokemon_data: Pre-loaded JSON data (if available)
            data_path: Path to gen9randombattle.json (if pokemon_data not provided)
        """
        if pokemon_data:
            self.pokemon_data = pokemon_data
        else:
            with open(data_path, 'r') as f:
                self.pokemon_data = json.load(f)
        
        # Cache list of valid Pokemon names
        self.valid_pokemon = list(self.pokemon_data.keys())
        
        # Filter out restricted legendaries for balance
        self.restricted = {
            'Calyrex-Shadow', 'Calyrex-Ice', 'Koraidon', 'Miraidon', 
            'Zacian', 'Zacian-Crowned', 'Zamazenta', 'Zamazenta-Crowned',
            'Eternatus', 'Kyogre', 'Groudon', 'Rayquaza', 'Mewtwo',
            'Dialga', 'Palkia', 'Giratina', 'Arceus',
            # Non-Gen9 Pokemon that may be in the JSON but aren't legal
            'Heliolisk', 'Helioptile', 'Ninjask', 'Nincada', 'Shedinja',
            'Swoobat', 'Woobat', 'Golisopod', 'Wimpod', 'Aurorus', 'Amaura',
            'Tyrantrum', 'Tyrunt', 'Cryogonal', 'Noivern', 'Noibat',
        }
        
        # Valid Pokemon for random selection (excluding base Arceus forms)
        self.selectable = [p for p in self.valid_pokemon 
                          if p not in self.restricted and not p.startswith('Arceus-')]
    
    def _generate_pokemon_set(self, species: str) -> str:
        """Generate a showdown-format set for a Pokemon."""
        data = self.pokemon_data.get(species)
        if not data:
            return ""
        
        level = data.get('level', 80)
        abilities = data.get('abilities', [''])
        items = data.get('items', ['Leftovers'])
        roles = data.get('roles', {})
        
        # Pick a random role
        if roles:
            role_name = random.choice(list(roles.keys()))
            role = roles[role_name]
        else:
            role = {}
        
        # Pick role-specific or top-level values
        ability = random.choice(role.get('abilities', abilities))
        item = random.choice(role.get('items', items))
        
        # ROBUST FIX: Validate item against Showdown's items.ts whitelist
        from ..teams.team_validator import is_valid_item
        if not is_valid_item(item):
            print(f"[RandomBattleTeambuilder] WARNING: Invalid item '{item}' for {species}, using Leftovers")
            item = 'Leftovers'

        moves = role.get('moves', ['Tackle'])
        tera_types = role.get('teraTypes', ['Normal'])
        evs = role.get('evs', data.get('evs', {}))
        ivs = role.get('ivs', data.get('ivs', {}))
        
        # Pick 4 moves (or fewer if not enough)
        selected_moves = random.sample(moves, min(4, len(moves)))
        
        # Pick a tera type
        tera_type = random.choice(tera_types)
        
        # Build EV spread string
        ev_parts = []
        ev_map = {'hp': 'HP', 'atk': 'Atk', 'def': 'Def', 'spa': 'SpA', 'spd': 'SpD', 'spe': 'Spe'}
        for stat, val in evs.items():
            if val and val > 0:
                ev_parts.append(f"{val} {ev_map.get(stat, stat)}")
        ev_str = " / ".join(ev_parts) if ev_parts else "252 HP / 252 Atk / 4 Spe"
        
        # Build IV spread string (only if different from 31)
        iv_parts = []
        for stat, val in ivs.items():
            if val is not None and val < 31:
                iv_parts.append(f"{val} {ev_map.get(stat, stat)}")
        iv_str = " / ".join(iv_parts) if iv_parts else ""
        
        # Infer nature from EVs
        nature = self._infer_nature(evs)
        
        # Build the set
        lines = [f"{species} @ {item}"]
        lines.append(f"Ability: {ability}")
        lines.append(f"Level: {level}")
        lines.append(f"Tera Type: {tera_type}")
        lines.append(f"EVs: {ev_str}")
        if iv_str:
            lines.append(f"IVs: {iv_str}")
        lines.append(f"{nature} Nature")
        for move in selected_moves:
            lines.append(f"- {move}")
        
        return "\n".join(lines)
    
    def _infer_nature(self, evs: Dict[str, int]) -> str:
        """Infer a reasonable nature from EV spread."""
        # Check for 0 atk (special attacker)
        if evs.get('atk') == 0:
            if evs.get('spe', 0) > 0:
                return "Timid"
            return "Modest"
        
        # Check for physical attacker
        if evs.get('atk', 0) > 0 and evs.get('spa', 0) == 0:
            if evs.get('spe', 0) > 0:
                return "Jolly"
            return "Adamant"
        
        # Mixed or bulk
        if evs.get('hp', 0) > 0:
            if evs.get('def', 0) > 0:
                return "Impish"
            if evs.get('spd', 0) > 0:
                return "Careful"
        
        # Default
        return "Hardy"
    
    def _generate_team(self) -> str:
        """Generate a complete 6-Pokemon team in showdown format."""
        # Randomly sample 6 unique Pokemon
        team_species = random.sample(self.selectable, min(6, len(self.selectable)))
        
        # Generate sets
        sets = []
        for species in team_species:
            pokemon_set = self._generate_pokemon_set(species)
            if pokemon_set:
                sets.append(pokemon_set)
        
        return "\n\n".join(sets)
    
    def yield_team(self) -> str:
        """Return a packed team string for poke-env with validation."""
        from .team_validator import validate_showdown_team, validate_packed_team
        
        max_retries = 10  # Increased retries
        for attempt in range(max_retries):
            showdown_team = self._generate_team()
            
            # Validate BEFORE packing (more reliable)
            is_valid, invalid_items = validate_showdown_team(showdown_team)
            
            if not is_valid:
                print(f"[RandomBattleTeambuilder] Attempt {attempt+1}: Invalid items {invalid_items}, regenerating...")
                continue
            
            try:
                parsed = self.parse_showdown_team(showdown_team)
                packed = self.join_team(parsed)
                # print(f"[RandomBattleTeambuilder] Yielded valid team on attempt {attempt+1}")
                return packed
            except Exception as e:
                print(f"[RandomBattleTeambuilder] Error packing team: {e}")
                continue
        
        # Final fallback: generate without validation
        print(f"[RandomBattleTeambuilder] WARNING: Exhausted {max_retries} retries, returning unvalidated team")
        showdown_team = self._generate_team()
        parsed = self.parse_showdown_team(showdown_team)
        return self.join_team(parsed)


# Convenience function
def create_random_teambuilder(pokemon_data: Optional[Dict] = None) -> RandomBattleTeambuilder:
    """Create a RandomBattleTeambuilder with optional pre-loaded data."""
    return RandomBattleTeambuilder(pokemon_data=pokemon_data)
