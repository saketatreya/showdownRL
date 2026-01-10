#!/usr/bin/env python3
"""
Replay Parser for Pokemon Showdown.
Parses replay logs into feature vectors for training the win predictor.

EXPANDED VERSION: Extracts as many features as possible from the original
observation space, acknowledging that some (like full team stats) are unavailable.

Feature dimensions (total: ~650):
- Active Pokemon P1: 97 dims (or subset available)
- Active Pokemon P2: 97 dims
- Team P1: 41 x 6 = 246 dims (revealed pokemon only)
- Team P2: 41 x 6 = 246 dims (revealed pokemon only)
- Field state: 35 dims (weather, terrain, hazards)
- Meta context: 23 dims (turn, advantages, etc.)
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from collections import defaultdict

import numpy as np


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Type mapping
TYPE_TO_IDX = {
    'Normal': 0, 'Fire': 1, 'Water': 2, 'Electric': 3, 'Grass': 4,
    'Ice': 5, 'Fighting': 6, 'Poison': 7, 'Ground': 8, 'Flying': 9,
    'Psychic': 10, 'Bug': 11, 'Rock': 12, 'Ghost': 13, 'Dragon': 14,
    'Dark': 15, 'Steel': 16, 'Fairy': 17
}

# Status to index
STATUS_TO_IDX = {
    'brn': 0, 'par': 1, 'slp': 2, 'frz': 3, 'psn': 4, 'tox': 5
}

# Weather mapping
WEATHER_TO_IDX = {
    'sunnyday': 0, 'desolateland': 0,
    'raindance': 1, 'primordialsea': 1,
    'sandstorm': 2,
    'hail': 3, 'snow': 3, 'snowscape': 3
}

# Terrain mapping
TERRAIN_TO_IDX = {
    'electricterrain': 0, 'electric': 0,
    'grassyterrain': 1, 'grassy': 1,
    'psychicterrain': 2, 'psychic': 2,
    'mistyterrain': 3, 'misty': 3
}

# Side conditions
SIDE_COND_TO_IDX = {
    'stealthrock': 0, 'spikes': 1, 'toxicspikes': 2, 'stickyweb': 3,
    'reflect': 4, 'lightscreen': 5, 'auroraveil': 6, 'tailwind': 7
}

# Pseudoweather / field conditions
PSEUDOWEATHER_TO_IDX = {
    'trickroom': 0, 'gravity': 1, 'magicroom': 2, 'wonderroom': 3
}


@dataclass
class ParsedPokemon:
    """Parsed Pokemon state from replay."""
    species: str
    nickname: Optional[str] = None
    level: int = 100
    hp: float = 1.0  # 0.0-1.0
    max_hp: int = 100
    status: Optional[str] = None
    fainted: bool = False
    active: bool = False
    types: List[str] = field(default_factory=list)
    moves: List[str] = field(default_factory=list)
    ability: Optional[str] = None
    item: Optional[str] = None
    boosts: Dict[str, int] = field(default_factory=lambda: {
        'atk': 0, 'def': 0, 'spa': 0, 'spd': 0, 'spe': 0, 'accuracy': 0, 'evasion': 0
    })
    terastallized: bool = False
    tera_type: Optional[str] = None
    gender: Optional[str] = None
    has_choice_lock: bool = False
    times_hit: int = 0  # Track damage dealt to this pokemon


@dataclass
class ParsedBattleState:
    """Complete battle state at a turn."""
    turn: int
    player1_team: Dict[str, ParsedPokemon] = field(default_factory=dict)
    player2_team: Dict[str, ParsedPokemon] = field(default_factory=dict)
    player1_active: Optional[str] = None
    player2_active: Optional[str] = None
    weather: Optional[str] = None
    weather_count: int = 0
    terrain: Optional[str] = None
    terrain_count: int = 0
    player1_side_conditions: Dict[str, int] = field(default_factory=dict)
    player2_side_conditions: Dict[str, int] = field(default_factory=dict)
    field_conditions: Dict[str, bool] = field(default_factory=dict)
    trick_room_active: bool = False
    gravity_active: bool = False
    

@dataclass
class StateLabel:
    """A state with its win label."""
    turn: int
    features: np.ndarray
    did_p1_win: bool


class ReplayParser:
    """
    Parses Pokemon Showdown replay logs into feature vectors.
    
    EXPANDED FEATURE SET (~650 dimensions):
    
    Active Pokemon (P1 & P2): 97 each = 194
    - HP (1)
    - Types (18 x 2 = 36)
    - Stats normalized (6) - estimated from level
    - Boosts (7)
    - Status (7 one-hot: 6 statuses + healthy)
    - Fainted (1)
    - Tera active (1)
    - Tera type (18)
    - Item flags (10) - inferred from moves/effects
    - Choice locked (1)
    - Ability flags (8) - inferred
    - Boost move used (1)
    
    Team State (P1 & P2): 41 x 6 each = 492
    - Per pokemon: HP, types, status, fainted (41 dims)
    
    Field State: 35
    - Weather (4)
    - Terrain (4)
    - Pseudoweather (9)
    - P1 hazards (8)
    - P2 hazards (8)
    - Weather/terrain duration (2)
    
    Meta Context: 23
    - Turn number (1)
    - HP advantages (2)
    - Fainted advantages (2)
    - Type matchup (2)
    - Speed tier (2)
    - Momentum (2)
    - Game phase (3)
    - Reserved (9)
    
    TOTAL: ~650 dimensions
    """
    
    # Feature dimensions
    ACTIVE_DIM = 97
    TEAM_POKEMON_DIM = 41
    TEAM_DIM = TEAM_POKEMON_DIM * 6
    FIELD_DIM = 35
    META_DIM = 23
    
    FEATURE_DIM = ACTIVE_DIM * 2 + TEAM_DIM * 2 + FIELD_DIM + META_DIM  # ~692
    
    def __init__(self, pokemon_data: Optional[Dict[str, Any]] = None):
        self.pokemon_data = pokemon_data or {}
        
        # Species type cache (loaded from pokemon_data or inferred)
        self.species_types: Dict[str, List[str]] = {}
        if pokemon_data:
            for species, data in pokemon_data.items():
                if 'types' in data:
                    self.species_types[self._normalize_species(species)] = data['types']
    
    def _normalize_species(self, species: str) -> str:
        """Normalize species name for lookup."""
        return species.lower().replace('-', '').replace(' ', '').replace('.', '').replace("'", "")
    
    def parse_replay_raw(self, replay_data: Dict[str, Any]) -> Tuple[List[ParsedBattleState], bool]:
        """
        Parse replay and return raw states + win label (for symmetry training).
        Returns (states, did_p1_win).
        """
        log = replay_data.get('log', '')
        winner = replay_data.get('winner', '')
        players = replay_data.get('players', [])
        
        # Determine if player 1 won
        if not winner and log:
            for line in log.split('\n'):
                if line.startswith('|win|'):
                    winner = line.split('|')[2]
                    break
        
        if not winner:
            return [], False

        did_p1_win = (winner == players[0]) if players else False
        
        # Parse the log
        states = self._parse_log(log)
        return states, did_p1_win

    def parse_replay(self, replay_data: Dict[str, Any]) -> List[StateLabel]:
        """
        Parse a replay into a list of (state, label) pairs.
        Returns one state per turn.
        """
        states, did_p1_win = self.parse_replay_raw(replay_data)
        
        if not states:
            return []
        
        # Convert to feature vectors
        labeled_states = []
        for state in states:
            try:
                features = self._state_to_features(state)
                labeled_states.append(StateLabel(
                    turn=state.turn,
                    features=features,
                    did_p1_win=did_p1_win
                ))
            except Exception as e:
                logger.debug(f"Error extracting features for turn {state.turn}: {e}")
                continue
        
        return labeled_states
    
    def _parse_log(self, log: str) -> List[ParsedBattleState]:
        """Parse battle log into sequence of states."""
        states = []
        current_state = ParsedBattleState(turn=0)
        
        for line in log.split('\n'):
            line = line.strip()
            if not line or not line.startswith('|'):
                continue
            
            parts = line.split('|')
            if len(parts) < 2:
                continue
            
            command = parts[1]
            
            try:
                if command == 'turn':
                    if current_state.turn > 0:
                        states.append(self._copy_state(current_state))
                    current_state.turn = int(parts[2])
                    
                elif command == 'switch' or command == 'drag':
                    self._handle_switch(current_state, parts)
                    
                elif command == 'move':
                    self._handle_move(current_state, parts)
                    
                elif command == '-damage':
                    self._handle_damage(current_state, parts)
                    
                elif command == '-heal':
                    self._handle_heal(current_state, parts)
                    
                elif command == 'faint':
                    self._handle_faint(current_state, parts)
                    
                elif command == '-status':
                    self._handle_status(current_state, parts)
                    
                elif command == '-curestatus':
                    self._handle_cure_status(current_state, parts)
                    
                elif command == '-boost':
                    self._handle_boost(current_state, parts, positive=True)
                    
                elif command == '-unboost':
                    self._handle_boost(current_state, parts, positive=False)
                    
                elif command == '-setboost':
                    self._handle_setboost(current_state, parts)
                    
                elif command == '-clearboost' or command == '-clearallboost':
                    self._handle_clearboost(current_state, parts)
                    
                elif command == '-weather':
                    self._handle_weather(current_state, parts)
                    
                elif command == '-fieldstart':
                    self._handle_field(current_state, parts, start=True)
                    
                elif command == '-fieldend':
                    self._handle_field(current_state, parts, start=False)
                    
                elif command == '-sidestart':
                    self._handle_side_condition(current_state, parts, start=True)
                    
                elif command == '-sideend':
                    self._handle_side_condition(current_state, parts, start=False)
                    
                elif command == '-terastallize':
                    self._handle_tera(current_state, parts)
                    
                elif command == '-ability':
                    self._handle_ability(current_state, parts)
                    
                elif command == '-item' or command == '-enditem':
                    self._handle_item(current_state, parts)
                    
                elif command == 'detailschange' or command == '-formechange':
                    self._handle_forme_change(current_state, parts)
                    
            except Exception as e:
                logger.debug(f"Error parsing line '{line}': {e}")
                continue
        
        # Add final state
        if current_state.turn > 0:
            states.append(current_state)
        
        return states
    
    def _get_player_and_pokemon(self, identifier: str) -> Tuple[int, str, str]:
        """Parse pokemon identifier like 'p1a: Pikachu'."""
        match = re.match(r'p([12])([a-z])?:\s*(.+)', identifier)
        if match:
            player = int(match.group(1))
            slot = match.group(2) or 'a'
            name = match.group(3)
            return player, slot, name
        return 0, 'a', identifier
    
    def _handle_switch(self, state: ParsedBattleState, parts: List[str]):
        """Handle switch/drag command."""
        if len(parts) < 4:
            return
        
        identifier = parts[2]
        details = parts[3]
        
        player, slot, nickname = self._get_player_and_pokemon(identifier)
        
        # Parse details: "Species, L50, M" or "Species, L50"
        details_parts = details.split(', ')
        species_raw = details_parts[0]
        species = species_raw.split('-')[0]  # Handle forms
        level = 100
        gender = None
        
        for part in details_parts[1:]:
            if part.startswith('L'):
                level = int(part[1:])
            elif part in ['M', 'F']:
                gender = part
        
        # Parse HP if present
        hp = 1.0
        if len(parts) > 4 and parts[4]:
            hp = self._parse_hp(parts[4])
        
        team = state.player1_team if player == 1 else state.player2_team
        
        # Clear active status from previous active
        for pokemon in team.values():
            pokemon.active = False
        
        # Create pokemon key (use species to track across switches)
        poke_key = self._normalize_species(species)
        
        # Infer types from species
        types = self.species_types.get(poke_key, [])
        if not types:
            # Try to infer from forme
            types = self._infer_types_from_forme(species_raw)
        
        if poke_key not in team:
            team[poke_key] = ParsedPokemon(
                species=species,
                nickname=nickname,
                level=level,
                hp=hp,
                active=True,
                types=types,
                gender=gender
            )
        else:
            team[poke_key].hp = hp
            team[poke_key].active = True
            team[poke_key].boosts = {k: 0 for k in team[poke_key].boosts}
            team[poke_key].has_choice_lock = False
        
        if player == 1:
            state.player1_active = poke_key
        else:
            state.player2_active = poke_key
    
    def _handle_move(self, state: ParsedBattleState, parts: List[str]):
        """Handle move command."""
        if len(parts) < 4:
            return
        
        identifier = parts[2]
        move_name = parts[3]
        
        player, slot, _ = self._get_player_and_pokemon(identifier)
        team = state.player1_team if player == 1 else state.player2_team
        active_key = state.player1_active if player == 1 else state.player2_active
        
        if active_key and active_key in team:
            pokemon = team[active_key]
            if move_name not in pokemon.moves and len(pokemon.moves) < 4:
                pokemon.moves.append(move_name)
            
            # Check for choice lock indication
            if '[from]lockedmove' in str(parts):
                pokemon.has_choice_lock = True
    
    def _handle_damage(self, state: ParsedBattleState, parts: List[str]):
        """Handle damage command."""
        if len(parts) < 4:
            return
        
        identifier = parts[2]
        hp_str = parts[3]
        
        player, slot, _ = self._get_player_and_pokemon(identifier)
        team = state.player1_team if player == 1 else state.player2_team
        active_key = state.player1_active if player == 1 else state.player2_active
        
        if active_key and active_key in team:
            team[active_key].hp = self._parse_hp(hp_str)
            team[active_key].times_hit += 1
    
    def _handle_heal(self, state: ParsedBattleState, parts: List[str]):
        """Handle heal command."""
        self._handle_damage(state, parts)
    
    def _handle_faint(self, state: ParsedBattleState, parts: List[str]):
        """Handle faint command."""
        if len(parts) < 3:
            return
        
        identifier = parts[2]
        player, slot, _ = self._get_player_and_pokemon(identifier)
        team = state.player1_team if player == 1 else state.player2_team
        active_key = state.player1_active if player == 1 else state.player2_active
        
        if active_key and active_key in team:
            team[active_key].fainted = True
            team[active_key].hp = 0.0
    
    def _handle_status(self, state: ParsedBattleState, parts: List[str]):
        """Handle status command."""
        if len(parts) < 4:
            return
        
        identifier = parts[2]
        status = parts[3]
        
        player, slot, _ = self._get_player_and_pokemon(identifier)
        team = state.player1_team if player == 1 else state.player2_team
        active_key = state.player1_active if player == 1 else state.player2_active
        
        if active_key and active_key in team:
            team[active_key].status = status
    
    def _handle_cure_status(self, state: ParsedBattleState, parts: List[str]):
        """Handle curestatus command."""
        if len(parts) < 3:
            return
        
        identifier = parts[2]
        player, slot, _ = self._get_player_and_pokemon(identifier)
        team = state.player1_team if player == 1 else state.player2_team
        active_key = state.player1_active if player == 1 else state.player2_active
        
        if active_key and active_key in team:
            team[active_key].status = None
    
    def _handle_boost(self, state: ParsedBattleState, parts: List[str], positive: bool):
        """Handle boost/unboost command."""
        if len(parts) < 5:
            return
        
        identifier = parts[2]
        stat = parts[3]
        amount = int(parts[4])
        
        if not positive:
            amount = -amount
        
        player, slot, _ = self._get_player_and_pokemon(identifier)
        team = state.player1_team if player == 1 else state.player2_team
        active_key = state.player1_active if player == 1 else state.player2_active
        
        if active_key and active_key in team:
            pokemon = team[active_key]
            if stat in pokemon.boosts:
                pokemon.boosts[stat] = max(-6, min(6, pokemon.boosts[stat] + amount))
    
    def _handle_setboost(self, state: ParsedBattleState, parts: List[str]):
        """Handle -setboost command."""
        if len(parts) < 5:
            return
        
        identifier = parts[2]
        stat = parts[3]
        amount = int(parts[4])
        
        player, slot, _ = self._get_player_and_pokemon(identifier)
        team = state.player1_team if player == 1 else state.player2_team
        active_key = state.player1_active if player == 1 else state.player2_active
        
        if active_key and active_key in team:
            pokemon = team[active_key]
            if stat in pokemon.boosts:
                pokemon.boosts[stat] = amount
    
    def _handle_clearboost(self, state: ParsedBattleState, parts: List[str]):
        """Handle clearboost command."""
        if len(parts) < 3:
            return
        
        identifier = parts[2]
        player, slot, _ = self._get_player_and_pokemon(identifier)
        team = state.player1_team if player == 1 else state.player2_team
        active_key = state.player1_active if player == 1 else state.player2_active
        
        if active_key and active_key in team:
            team[active_key].boosts = {k: 0 for k in team[active_key].boosts}
    
    def _handle_weather(self, state: ParsedBattleState, parts: List[str]):
        """Handle weather command."""
        if len(parts) < 3:
            return
        
        weather = parts[2].lower().replace(' ', '')
        if weather == 'none':
            state.weather = None
            state.weather_count = 0
        else:
            # Extract base weather name
            weather_base = weather.split(':')[0]
            state.weather = weather_base
            state.weather_count = min(state.weather_count + 1, 8)
    
    def _handle_field(self, state: ParsedBattleState, parts: List[str], start: bool):
        """Handle fieldstart/fieldend command."""
        if len(parts) < 3:
            return
        
        field_name = parts[2].lower().replace(' ', '').replace('move:', '')
        
        if 'terrain' in field_name:
            if start:
                terrain_key = field_name.replace('terrain', '')
                state.terrain = terrain_key
                state.terrain_count = 1
            else:
                state.terrain = None
                state.terrain_count = 0
        elif 'trickroom' in field_name:
            state.trick_room_active = start
        elif 'gravity' in field_name:
            state.gravity_active = start
        else:
            state.field_conditions[field_name] = start
    
    def _handle_side_condition(self, state: ParsedBattleState, parts: List[str], start: bool):
        """Handle sidestart/sideend command."""
        if len(parts) < 4:
            return
        
        side = parts[2]
        condition = parts[3].lower().replace(' ', '').replace('move:', '')
        
        side_conditions = state.player1_side_conditions if 'p1' in side else state.player2_side_conditions
        
        if start:
            side_conditions[condition] = side_conditions.get(condition, 0) + 1
        else:
            side_conditions.pop(condition, None)
    
    def _handle_tera(self, state: ParsedBattleState, parts: List[str]):
        """Handle terastallize command."""
        if len(parts) < 4:
            return
        
        identifier = parts[2]
        tera_type = parts[3]
        
        player, slot, _ = self._get_player_and_pokemon(identifier)
        team = state.player1_team if player == 1 else state.player2_team
        active_key = state.player1_active if player == 1 else state.player2_active
        
        if active_key and active_key in team:
            team[active_key].terastallized = True
            team[active_key].tera_type = tera_type
    
    def _handle_ability(self, state: ParsedBattleState, parts: List[str]):
        """Handle ability reveal."""
        if len(parts) < 4:
            return
        
        identifier = parts[2]
        ability = parts[3]
        
        player, slot, _ = self._get_player_and_pokemon(identifier)
        team = state.player1_team if player == 1 else state.player2_team
        active_key = state.player1_active if player == 1 else state.player2_active
        
        if active_key and active_key in team:
            team[active_key].ability = ability
    
    def _handle_item(self, state: ParsedBattleState, parts: List[str]):
        """Handle item reveal/consume."""
        if len(parts) < 4:
            return
        
        identifier = parts[2]
        item = parts[3]
        
        player, slot, _ = self._get_player_and_pokemon(identifier)
        team = state.player1_team if player == 1 else state.player2_team
        active_key = state.player1_active if player == 1 else state.player2_active
        
        if active_key and active_key in team:
            team[active_key].item = item
    
    def _handle_forme_change(self, state: ParsedBattleState, parts: List[str]):
        """Handle forme change."""
        pass  # Types might change, but we keep tracking by base species
    
    def _infer_types_from_forme(self, species: str) -> List[str]:
        """Infer types from forme name if not in data."""
        # Common forme type mappings
        forme_types = {
            'rotomwash': ['Electric', 'Water'],
            'rotomheat': ['Electric', 'Fire'],
            'rotommow': ['Electric', 'Grass'],
            'rotomfrost': ['Electric', 'Ice'],
            'rotomfan': ['Electric', 'Flying'],
        }
        return forme_types.get(self._normalize_species(species), [])
    
    def _parse_hp(self, hp_str: str) -> float:
        """Parse HP string like '150/200' or '0 fnt' to 0.0-1.0."""
        if 'fnt' in hp_str:
            return 0.0
        
        hp_str = hp_str.split()[0]
        
        if '/' in hp_str:
            parts = hp_str.split('/')
            current = float(parts[0])
            max_hp = float(parts[1])
            return current / max_hp if max_hp > 0 else 0.0
        
        try:
            return float(hp_str) / 100.0
        except:
            return 1.0
    
    def _copy_state(self, state: ParsedBattleState) -> ParsedBattleState:
        """Deep copy a battle state."""
        import copy
        return copy.deepcopy(state)
    
    def _state_to_features(self, state: ParsedBattleState, perspective: int = 1) -> np.ndarray:
        """
        Convert parsed battle state to feature vector.
        
        Args:
            state: The parsed state
            perspective: 1 for Player 1's view, 2 for Player 2's view.
        
        Layout (~692 dims):
        - [0:97]     P1 active pokemon
        - [97:194]   P2 active pokemon
        - [194:440]  P1 team (6 x 41)
        - [440:686]  P2 team (6 x 41)
        - [686:721]  Field state (35)
        - [721:744]  Meta context (23)
        """
        features = np.zeros(self.FEATURE_DIM, dtype=np.float32)
        offset = 0
        
        # Determine "My" team/active and "Opponent's" team/active based on perspective
        if perspective == 1:
            my_team = state.player1_team
            my_active_key = state.player1_active
            opp_team = state.player2_team
            opp_active_key = state.player2_active
        else:
            my_team = state.player2_team
            my_active_key = state.player2_active
            opp_team = state.player1_team
            opp_active_key = state.player1_active
        
        # My active pokemon
        my_active = None
        if my_active_key and my_active_key in my_team:
            my_active = my_team[my_active_key]
        self._encode_active_pokemon(features, offset, my_active, is_opponent=False)
        offset += self.ACTIVE_DIM
        
        # Opponent active pokemon
        opp_active = None
        if opp_active_key and opp_active_key in opp_team:
            opp_active = opp_team[opp_active_key]
        self._encode_active_pokemon(features, offset, opp_active, is_opponent=True)
        offset += self.ACTIVE_DIM
        
        # My team
        self._encode_team(features, offset, my_team, my_active)
        offset += self.TEAM_DIM
        
        # Opponent team
        self._encode_team(features, offset, opp_team, opp_active)
        offset += self.TEAM_DIM
        
        # Field state
        self._encode_field(features, offset, state, perspective)
        offset += self.FIELD_DIM
        
        # Meta context
        self._encode_meta(features, offset, state, perspective)
        
        return features
    
    def _encode_active_pokemon(
        self, 
        features: np.ndarray, 
        offset: int, 
        pokemon: Optional[ParsedPokemon],
        is_opponent: bool
    ):
        """
        Encode active pokemon (97 dims).
        
        Layout:
        - [0]: HP
        - [1:19]: Type 1 (18)
        - [19:37]: Type 2 (18)
        - [37:43]: Stats placeholder (6)
        - [43:50]: Boosts (7)
        - [50:57]: Status (7: 6 statuses + healthy)
        - [57]: Fainted
        - [58]: Tera active
        - [59:77]: Tera type (18)
        - [77:87]: Item flags (10)
        - [87]: Choice lock
        - [88:96]: Ability flags (8)
        - [96]: Boost move used
        """
        if pokemon is None:
            return  # Leave zeros
        
        # HP
        features[offset] = pokemon.hp
        
        # Type 1
        if pokemon.types and len(pokemon.types) > 0:
            t1 = pokemon.types[0]
            if t1 in TYPE_TO_IDX:
                features[offset + 1 + TYPE_TO_IDX[t1]] = 1.0
        
        # Type 2
        if pokemon.types and len(pokemon.types) > 1:
            t2 = pokemon.types[1]
            if t2 in TYPE_TO_IDX:
                features[offset + 19 + TYPE_TO_IDX[t2]] = 1.0
        
        # Stats (placeholder - we don't have actual stats from replays)
        # Just leave as zeros, or use level as proxy
        features[offset + 37] = pokemon.level / 100.0  # Level as first "stat"
        
        # Boosts (7 stats)
        boost_order = ['atk', 'def', 'spa', 'spd', 'spe', 'accuracy', 'evasion']
        for i, stat in enumerate(boost_order):
            features[offset + 43 + i] = pokemon.boosts.get(stat, 0) / 6.0
        
        # Status (7-dim one-hot)
        if pokemon.status:
            if pokemon.status in STATUS_TO_IDX:
                features[offset + 50 + STATUS_TO_IDX[pokemon.status]] = 1.0
        else:
            features[offset + 56] = 1.0  # Healthy
        
        # Fainted
        features[offset + 57] = 1.0 if pokemon.fainted else 0.0
        
        # Tera
        features[offset + 58] = 1.0 if pokemon.terastallized else 0.0
        if pokemon.tera_type and pokemon.tera_type in TYPE_TO_IDX:
            features[offset + 59 + TYPE_TO_IDX[pokemon.tera_type]] = 1.0
        
        # Item flags - inferred from item name if known
        # [0:10] = various item categories
        if pokemon.item:
            item_lower = pokemon.item.lower()
            if 'choice' in item_lower:
                features[offset + 77] = 1.0  # Choice item
            if 'leftovers' in item_lower or 'blacksludge' in item_lower:
                features[offset + 78] = 1.0  # Recovery item
            if 'lifeorb' in item_lower:
                features[offset + 79] = 1.0  # Damage boost
            if 'focussash' in item_lower:
                features[offset + 80] = 1.0  # Sash
            if 'assaultvest' in item_lower:
                features[offset + 81] = 1.0  # Assault vest
        
        # Choice lock
        features[offset + 87] = 1.0 if pokemon.has_choice_lock else 0.0
        
        # Ability flags - inferred from ability name if known
        if pokemon.ability:
            ability_lower = pokemon.ability.lower()
            if 'intimidate' in ability_lower:
                features[offset + 88] = 1.0
            if 'levitate' in ability_lower:
                features[offset + 89] = 1.0
            if 'sturdy' in ability_lower:
                features[offset + 90] = 1.0
            if 'multiscale' in ability_lower:
                features[offset + 91] = 1.0
    
    def _encode_team(
        self, 
        features: np.ndarray, 
        offset: int, 
        team: Dict[str, ParsedPokemon],
        active: Optional[ParsedPokemon]
    ):
        """
        Encode team state (6 x 41 = 246 dims).
        
        Per pokemon (41 dims):
        - [0]: HP
        - [1:19]: Type 1 (18)
        - [19:37]: Type 2 (18)
        - [37]: Status present
        - [38]: Fainted
        - [39]: Is active
        - [40]: Has been revealed
        """
        team_list = list(team.values())[:6]
        
        for i, pokemon in enumerate(team_list):
            poke_offset = offset + i * self.TEAM_POKEMON_DIM
            
            # HP
            features[poke_offset] = pokemon.hp
            
            # Type 1
            if pokemon.types and len(pokemon.types) > 0:
                t1 = pokemon.types[0]
                if t1 in TYPE_TO_IDX:
                    features[poke_offset + 1 + TYPE_TO_IDX[t1]] = 1.0
            
            # Type 2
            if pokemon.types and len(pokemon.types) > 1:
                t2 = pokemon.types[1]
                if t2 in TYPE_TO_IDX:
                    features[poke_offset + 19 + TYPE_TO_IDX[t2]] = 1.0
            
            # Status
            features[poke_offset + 37] = 1.0 if pokemon.status else 0.0
            
            # Fainted
            features[poke_offset + 38] = 1.0 if pokemon.fainted else 0.0
            
            # Is active
            features[poke_offset + 39] = 1.0 if pokemon.active else 0.0
            
            # Revealed
            features[poke_offset + 40] = 1.0  # If it's in team dict, it's revealed
    
    def _encode_field(self, features: np.ndarray, offset: int, state: ParsedBattleState, perspective: int = 1):
        """
        Encode field state (~35 dims).
        
        Layout:
        - [0:4]: Weather (one-hot)
        - [4:8]: Terrain (one-hot)
        - [8:12]: Pseudoweather (4)
        - [12]: Weather duration
        - [13]: Terrain duration
        - [14:22]: My side conditions (8)
        - [22:30]: Opponent side conditions (8)
        """
        # Weather
        if state.weather:
            weather_key = state.weather.lower().replace(' ', '')
            if weather_key in WEATHER_TO_IDX:
                features[offset + WEATHER_TO_IDX[weather_key]] = 1.0
        
        # Terrain
        if state.terrain:
            terrain_key = state.terrain.lower().replace(' ', '')
            if terrain_key in TERRAIN_TO_IDX:
                features[offset + 4 + TERRAIN_TO_IDX[terrain_key]] = 1.0
        
        # Pseudoweather
        if state.trick_room_active:
            features[offset + 8] = 1.0
        if state.gravity_active:
            features[offset + 9] = 1.0
        
        # Weather duration
        features[offset + 12] = min(state.weather_count / 8.0, 1.0)
        
        # Terrain duration
        features[offset + 13] = min(state.terrain_count / 5.0, 1.0)
        
        # Perspective-dependent side conditions
        if perspective == 1:
            my_side = state.player1_side_conditions
            opp_side = state.player2_side_conditions
        else:
            my_side = state.player2_side_conditions
            opp_side = state.player1_side_conditions
        
        # My side conditions
        for cond, idx in SIDE_COND_TO_IDX.items():
            if cond in my_side:
                features[offset + 14 + idx] = min(my_side[cond] / 3.0, 1.0)
        
        # Opponent side conditions
        for cond, idx in SIDE_COND_TO_IDX.items():
            if cond in opp_side:
                features[offset + 22 + idx] = min(opp_side[cond] / 3.0, 1.0)
    
    def _encode_meta(self, features: np.ndarray, offset: int, state: ParsedBattleState, perspective: int = 1):
        """
        Encode meta context (23 dims).
        
        Layout:
        - [0]: Turn number (normalized)
        - [1]: My total HP
        - [2]: Opp total HP
        - [3]: HP advantage (My - Opp)
        - [4]: My fainted count
        - [5]: Opp fainted count
        - [6]: Fainted advantage (Opp broken - My broken)
        - [7]: My revealed count
        - [8]: Opp revealed count
        - [9]: Game phase (early)
        - [10]: Game phase (mid)
        - [11]: Game phase (late)
        - [12:23]: Reserved
        """
        # Turn number
        features[offset] = min(state.turn / 100.0, 1.0)
        
        if perspective == 1:
            my_team = state.player1_team
            opp_team = state.player2_team
        else:
            my_team = state.player2_team
            opp_team = state.player1_team
        
        # HP totals
        my_total_hp = sum(p.hp for p in my_team.values())
        opp_total_hp = sum(p.hp for p in opp_team.values())
        features[offset + 1] = my_total_hp / 6.0
        features[offset + 2] = opp_total_hp / 6.0
        
        # HP advantage
        features[offset + 3] = (my_total_hp - opp_total_hp) / 6.0
        
        # Fainted counts
        my_fainted = sum(1 for p in my_team.values() if p.fainted)
        opp_fainted = sum(1 for p in opp_team.values() if p.fainted)
        features[offset + 4] = my_fainted / 6.0
        features[offset + 5] = opp_fainted / 6.0
        
        # Fainted advantage (Higher is better for me, so Opp fainted - My fainted)
        features[offset + 6] = (opp_fainted - my_fainted) / 6.0
        
        # Revealed counts
        features[offset + 7] = len(my_team) / 6.0
        features[offset + 8] = len(opp_team) / 6.0
        
        # Game phase
        if state.turn <= 5:
            features[offset + 9] = 1.0  # Early
        elif state.turn <= 20:
            features[offset + 10] = 1.0  # Mid
        else:
            features[offset + 11] = 1.0  # Late


def test_parser():
    """Test the parser with sample replays."""
    parser = ReplayParser()
    
    sample_path = Path("data/replays")
    if sample_path.exists():
        total_states = 0
        for replay_file in list(sample_path.rglob("*.json"))[:5]:  # Test 5 files
            if replay_file.name == "progress.json":
                continue
            
            with open(replay_file, 'r') as f:
                replay = json.load(f)
            
            states = parser.parse_replay(replay)
            total_states += len(states)
            print(f"Parsed {len(states)} states from {replay_file.name}")
            
            if states:
                print(f"  Feature shape: {states[0].features.shape}")
                print(f"  Feature dim: {parser.FEATURE_DIM}")
                print(f"  Non-zero features: {np.count_nonzero(states[0].features)}")
                print(f"  Range: [{states[0].features.min():.3f}, {states[0].features.max():.3f}]")
        
        print(f"\nTotal: {total_states} states from 5 replays")
        print(f"Feature dimension: {parser.FEATURE_DIM}")
    else:
        print("No sample replays available. Run scrape_replays.py first.")


if __name__ == "__main__":
    test_parser()
