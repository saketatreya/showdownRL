"""
Feature encoders for the Gen 9 Random Battle RL bot.
Decomposed from the monolithic ObservationBuilder.
"""

import numpy as np
from typing import Optional, Dict, List, Any
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.move import Move
from poke_env.battle.abstract_battle import AbstractBattle

from .utils import (
    type_to_onehot, status_to_onehot, boosts_to_array,
    get_item_flags, is_choice_item, get_ability_flags,
    MoveClassifier, calculate_speed, get_type_effectiveness,
    move_category_to_onehot, is_immune_by_ability
)
from .belief_tracker import BeliefTracker

def safe_get_priority(move):
    try:
        return move.priority
    except (KeyError, AttributeError):
        return 0

def safe_get_base_power(move):
    try:
        return move.base_power
    except (KeyError, AttributeError):
        return 0

def safe_get_accuracy(move):
    try:
        return move.accuracy
    except (KeyError, AttributeError):
        return 100


class ActivePokemonEncoder:
    """Encodes state of an active pokemon."""
    
    # Dimensions: HP(1) + Type1(18) + Type2(18) + Stats(6) + Boosts(7) + 
    #             Status(7) + Fainted(1) + Tera(1) + TeraType(18) + 
    #             ItemFlags(10) + ChoiceLock(1) + AbilityFlags(8) + BoostMove(1) = 97
    SIZE = 1 + 18 + 18 + 6 + 7 + 7 + 1 + 1 + 18 + 10 + 1 + 8 + 1  # = 97
    
    def encode(self, pokemon: Optional[Pokemon], is_opponent: bool = False) -> np.ndarray:
        embedding = np.zeros(self.SIZE, dtype=np.float32)
        if pokemon is None:
            return embedding
        
        idx = 0
        
        # HP fraction (1 dim)
        embedding[idx] = pokemon.current_hp_fraction if pokemon.current_hp_fraction else 0.0
        idx += 1
        
        # Type 1 (18 dims)
        type1 = pokemon.type_1.name.lower() if pokemon.type_1 else "normal"
        embedding[idx:idx+18] = type_to_onehot(type1)
        idx += 18
        
        # Type 2 (18 dims)
        type2 = pokemon.type_2.name.lower() if pokemon.type_2 else None
        embedding[idx:idx+18] = type_to_onehot(type2)
        idx += 18
        
        # Base stats (6 dims, normalized)
        if pokemon.base_stats:
            stats = ['hp', 'atk', 'def', 'spa', 'spd', 'spe']
            for stat in stats:
                embedding[idx] = pokemon.base_stats.get(stat, 100) / 200.0
                idx += 1
        else:
            idx += 6
        
        # Stat boosts (7 dims)
        if pokemon.boosts:
            boosts = boosts_to_array(pokemon.boosts)
            embedding[idx:idx+7] = boosts
        idx += 7
        
        # Status (7 dims)
        status_onehot = status_to_onehot(pokemon.status.name.lower() if pokemon.status else None)
        embedding[idx:idx+7] = status_onehot
        idx += 7
        
        # Fainted flag (1 dim)
        embedding[idx] = 1.0 if pokemon.fainted else 0.0
        idx += 1
        
        # Has terastallized (1 dim)
        embedding[idx] = 1.0 if pokemon.is_terastallized else 0.0
        idx += 1
        
        # Tera type (18 dims)
        if hasattr(pokemon, 'tera_type') and pokemon.tera_type:
            tera_type_name = pokemon.tera_type.name.lower()
            embedding[idx:idx+18] = type_to_onehot(tera_type_name)
        idx += 18
        
        # Item flags (10 dims)
        item_flags = get_item_flags(pokemon.item)
        embedding[idx:idx+10] = item_flags
        idx += 10
        
        # Choice lock-in flag (1 dim)
        is_locked = 0.0
        if is_choice_item(pokemon.item) and hasattr(pokemon, 'moves') and len(pokemon.moves) > 0:
            is_locked = 1.0
        embedding[idx] = is_locked
        idx += 1
        
        # Ability flags (8 dims)
        ability_flags = get_ability_flags(pokemon.ability)
        embedding[idx:idx+8] = ability_flags
        idx += 8
        
        # Has boosting moves available (1 dim)
        has_boost_move = 0.0
        if not is_opponent and hasattr(pokemon, 'moves'):
            for move_id in pokemon.moves:
                if MoveClassifier.is_boost(move_id):
                    has_boost_move = 1.0
                    break
        embedding[idx] = has_boost_move
        idx += 1
        
        return embedding

class TeamEncoder:
    """Encodes state of a team (bench + active)."""
    
    POKEMON_DIM = 41 
    SIZE = 41 * 6 # 246
    
    def encode(self, team: Dict[str, Pokemon], active: Optional[Pokemon], is_opponent: bool = False) -> np.ndarray:
        embedding = np.zeros(self.SIZE, dtype=np.float32)
        revealed_pokemon = list(team.values())[:6]
        
        for i in range(6):
            base_idx = i * self.POKEMON_DIM
            idx = base_idx
            
            if i < len(revealed_pokemon):
                pokemon = revealed_pokemon[i]
                
                # HP fraction (1 dim)
                embedding[idx] = pokemon.current_hp_fraction if pokemon.current_hp_fraction else 0.0
                idx += 1
                
                # Type 1 (18 dims)
                type1 = pokemon.type_1.name.lower() if pokemon.type_1 else "normal"
                embedding[idx:idx+18] = type_to_onehot(type1)
                idx += 18
                
                # Type 2 (18 dims)
                type2 = pokemon.type_2.name.lower() if pokemon.type_2 else None
                embedding[idx:idx+18] = type_to_onehot(type2)
                idx += 18
                
                # Status (1 dim)
                embedding[idx] = 1.0 if pokemon.status else 0.0
                idx += 1
                
                # Active flag (1 dim)
                is_active = 1.0 if (active and pokemon.species == active.species) else 0.0
                embedding[idx] = is_active
                idx += 1
                
                # Fainted flag (1 dim)
                embedding[idx] = 1.0 if pokemon.fainted else 0.0
                idx += 1
                
                # Is revealed flag (1 dim)
                embedding[idx] = 1.0
                idx += 1
                
            else:
                # Mark as unrevealed (-1.0 at last dim)
                embedding[base_idx + self.POKEMON_DIM - 1] = -1.0
                
        return embedding

class FieldEncoder:
    """Encodes global field state (Weather, Terrain, Hazards)."""
    
    SIZE = 35
    
    # Constants for weather/field IDs
    WEATHER_TO_ID = {'sun':0, 'sunnyday':0, 'desolateland':0, 
                     'rain':1, 'raindance':1, 'primordialsea':1, 
                     'sand':2, 'sandstorm':2, 
                     'hail':3, 'snow':3}
    GLOBAL_FIELD_TO_ID = {'electricterrain':1, 'grassyterrain':2, 'mistyterrain':3, 'psychicterrain':4,
                          'trickroom':5, 'gravity':6, 'magicroom':7, 'wonderroom':8}
    
    def encode(self, battle: AbstractBattle) -> np.ndarray:
        # Weather: [Sun, Rain, Sand, Snow, Other, None] (6 dims)
        weather_emb = np.zeros(6, dtype=np.float32)
        if battle.weather:
            found_weather = False
            for w in battle.weather:
                w_name = w.name.lower().replace(' ', '').replace('-', '')
                for k, v in self.WEATHER_TO_ID.items():
                    if k in w_name:
                        weather_emb[v] = 1.0
                        found_weather = True
                        break
            if not found_weather:
                weather_emb[4] = 1.0  # Other/unknown weather
        else:
            weather_emb[5] = 1.0  # None (index 5, not 0)
            
        field_emb = np.zeros(9, dtype=np.float32)
        if battle.fields:
            for f in battle.fields:
                f_name = f.name.lower().replace(' ', '').replace('-', '')
                if f_name in self.GLOBAL_FIELD_TO_ID:
                    field_emb[self.GLOBAL_FIELD_TO_ID[f_name]] = 1.0
        if np.sum(field_emb) == 0:
            field_emb[0] = 1.0 # None
            
        def encode_hazards(side):
            vec = np.zeros(10, dtype=np.float32)
            if not side: return vec
            s = side
            k = [str(x).lower().replace(' ', '').replace('-', '') for x in s.keys()]
            
            # Helper to get value
            def val(name):
                for key, value in s.items():
                    if name in str(key).lower().replace(' ', '').replace('-', ''):
                        return value if isinstance(value, int) else 1
                return 0

            if any('stealthrock' in x for x in k): vec[0] = 1.0
            vec[1] = min(val('spikes'), 3) / 3.0
            vec[2] = min(val('toxicspikes'), 2) / 2.0
            if any('stickyweb' in x for x in k): vec[3] = 1.0
            if any('reflect' in x for x in k): vec[4] = 1.0
            if any('lightscreen' in x for x in k): vec[5] = 1.0
            if any('auroraveil' in x for x in k): vec[6] = 1.0
            if any('tailwind' in x for x in k): vec[7] = 1.0
            if any('safeguard' in x for x in k): vec[8] = 1.0
            if any('mist' in x for x in k): vec[9] = 1.0
            return vec

        our_hazards = encode_hazards(battle.side_conditions)
        opp_hazards = encode_hazards(battle.opponent_side_conditions)
        
        return np.concatenate([weather_emb, field_emb, our_hazards, opp_hazards])

class MovesEncoder:
    """Encodes available moves state."""
    
    MOVE_DIM = 37
    SIZE = 37 * 4 # 148
    
    def encode(self, battle: AbstractBattle) -> np.ndarray:
        embedding = np.zeros(self.SIZE, dtype=np.float32)
        
        our_mon = battle.active_pokemon
        opp_mon = battle.opponent_active_pokemon
        our_types = [t.name.lower() for t in our_mon.types if t] if our_mon else []
        opp_type1 = opp_mon.type_1.name.lower() if opp_mon and opp_mon.type_1 else None
        opp_type2 = opp_mon.type_2.name.lower() if opp_mon and opp_mon.type_2 else None
        
        # Speed calc logic
        we_are_faster = False
        if our_mon and opp_mon:
             # Check status and side conditions
             our_para = our_mon.status.name.lower() == 'par' if our_mon.status else False
             opp_para = opp_mon.status.name.lower() == 'par' if opp_mon.status else False
             
             our_tailwind = 'tailwind' in battle.side_conditions
             opp_tailwind = 'tailwind' in battle.opponent_side_conditions
             
             # Calculate raw effective speed
             s1 = calculate_speed(our_mon.base_stats.get('spe', 100), our_mon.level, our_mon.boosts.get('spe',0), our_para, our_tailwind)
             s2 = calculate_speed(opp_mon.base_stats.get('spe', 100), opp_mon.level, opp_mon.boosts.get('spe',0), opp_para, opp_tailwind)
             
             # Determine simple order
             we_are_faster = s1 > s2
             
             # Handle Trick Room (reverses speed order)
             trick_room = False
             if battle.fields:
                 for f in battle.fields:
                     if f.name.lower().replace(' ', '') == 'trickroom':
                         trick_room = True
                         break
             
             if trick_room:
                 we_are_faster = not we_are_faster
             
        has_psychic_terrain = False
        if battle.fields:
             for f in battle.fields:
                 if 'psychic' in f.name.lower(): has_psychic_terrain = True

        for i, move in enumerate(battle.available_moves[:4]):
            idx = i * self.MOVE_DIM
            
            priority = safe_get_priority(move)
            move_type = move.type.name.lower() if move.type else None
            
            embedding[idx:idx+18] = type_to_onehot(move_type)
            idx += 18
            
            embedding[idx] = safe_get_base_power(move) / 200.0
            idx += 1
            acc = safe_get_accuracy(move)
            embedding[idx] = acc / 100.0 if acc is not True else 1.0
            idx += 1
            if move.max_pp:
                 embedding[idx] = move.current_pp / move.max_pp
            else:
                 embedding[idx] = 1.0
            idx += 1
            embedding[idx] = (priority + 7) / 14.0
            idx += 1
            
            cat_onehot = move_category_to_onehot(move.category.name.lower() if move.category else "status")
            embedding[idx:idx+3] = cat_onehot
            idx += 3
            
            eff = 1.0
            if opp_type1 and move_type:
                 eff = get_type_effectiveness(move_type, opp_type1, opp_type2)
            embedding[idx] = eff / 4.0
            idx += 1
            
            move_id = move.id
            embedding[idx] = 1.0 if MoveClassifier.is_boost(move_id) else 0.0
            idx += 1
            embedding[idx] = 1.0 if MoveClassifier.is_recovery(move_id) else 0.0
            idx += 1
            embedding[idx] = 1.0 if priority > 0 else 0.0
            idx += 1
            embedding[idx] = 1.0 if MoveClassifier.is_hazard(move_id) else 0.0
            idx += 1
            
            # V3
            is_immune = (eff == 0)
            embedding[idx] = 1.0 if is_immune else 0.0
            idx += 1
            
            is_stab = move_type in our_types if move_type else False
            embedding[idx] = 1.0 if is_stab else 0.0
            idx += 1
            
            will_move_first = False
            if priority > 0:
                 will_move_first = not has_psychic_terrain
            elif priority == 0:
                 will_move_first = we_are_faster
            embedding[idx] = 1.0 if will_move_first else 0.0
            idx += 1
            
            # Prankster blocked placeholder (not yet implemented - dark type immunity)
            embedding[idx] = 0.0
            idx += 1
            
            # Spin blocked placeholder (not yet implemented - ghost type blocks spin)
            embedding[idx] = 0.0
            idx += 1
            
            # V4
            ability_immune = False
            if opp_mon and opp_mon.ability and move_type:
                ability_immune = is_immune_by_ability(move_type, opp_mon.ability)
            embedding[idx] = 1.0 if ability_immune else 0.0
            idx += 1
            
            # Weather damage modifier (calculate dynamically)
            weather_mod = 1.0
            if battle.weather and move_type:
                for w in battle.weather:
                    w_name = w.name.lower()
                    if 'sun' in w_name or 'desolateland' in w_name:
                        if move_type == 'fire':
                            weather_mod = 1.5
                        elif move_type == 'water':
                            weather_mod = 0.5
                    elif 'rain' in w_name or 'primordialsea' in w_name:
                        if move_type == 'water':
                            weather_mod = 1.5
                        elif move_type == 'fire':
                            weather_mod = 0.5
            embedding[idx] = weather_mod / 1.5  # Normalize to 0-1 range
            idx += 1
            
        return embedding

class DamageEncoder:
    """Encodes damage estimates using belief calculator."""
    
    SIZE = 40
    
    def __init__(self, damage_calculator: Any, belief_tracker: BeliefTracker):
        self.dmg_calc = damage_calculator
        self.belief_tracker = belief_tracker
        
    def encode(self, battle: AbstractBattle) -> np.ndarray:
        embedding = np.zeros(self.SIZE, dtype=np.float32)
        if not battle.active_pokemon or not battle.opponent_active_pokemon:
             return embedding
             
        our_mon = battle.active_pokemon
        opp_mon = battle.opponent_active_pokemon
        dmg_calc = self.dmg_calc
        
        # Get types
        opp_type1 = opp_mon.type_1.name.lower() if opp_mon.type_1 else None
        opp_type2 = opp_mon.type_2.name.lower() if opp_mon.type_2 else None
        our_type1 = our_mon.type_1.name.lower() if our_mon.type_1 else None
        our_type2 = our_mon.type_2.name.lower() if our_mon.type_2 else None

        # ========== OUR MOVES DAMAGE (dims 0-7) ==========
        max_our_dmg = 0.0
        for i, move in enumerate(battle.available_moves[:4]):
            if not move or not move.base_power:
                continue
            try:
                # Use belief calculator
                result = dmg_calc.calculate_move_damage(battle, move, is_our_move=True)
                embedding[i * 2] = result.min_percent
                embedding[i * 2 + 1] = result.max_percent
                max_our_dmg = max(max_our_dmg, result.max_percent)
            except Exception:
                # Fallback heuristic
                if move.type:
                    eff = get_type_effectiveness(move.type.name.lower(), opp_type1, opp_type2)
                    rough_dmg = (move.base_power / 100.0) * eff * 0.3
                    embedding[i * 2] = rough_dmg * 0.85
                    embedding[i * 2 + 1] = rough_dmg
                    max_our_dmg = max(max_our_dmg, rough_dmg)
                    
        # ========== BENCH BEST DAMAGE (dims 8-12) ==========
        for i, pokemon in enumerate(battle.available_switches[:5]):
            best_dmg = 0.0
            poke_type1 = pokemon.type_1.name.lower() if pokemon.type_1 else None
            poke_type2 = pokemon.type_2.name.lower() if pokemon.type_2 else None
            
            for atk_type in [poke_type1, poke_type2]:
                if atk_type:
                    eff = get_type_effectiveness(atk_type, opp_type1, opp_type2)
                    best_dmg = max(best_dmg, eff * 0.4) # STAB estimate
            embedding[8 + i] = best_dmg
            
        # ========== BAYESIAN INCOMING DAMAGE (dims 13-20) ==========
        belief = self.belief_tracker.get_or_create_belief(opp_mon.species)
        predicted_moves = belief.get_unrevealed_move_probs()
        sorted_predictions = sorted(predicted_moves.items(), key=lambda x: x[1], reverse=True)[:4]
        
        max_incoming = 0.0
        
        for i, (move_name, prob) in enumerate(sorted_predictions):
            estimated_power = 80
            estimated_type = None
            
            # Look up move in belief data to guess type
            if belief.is_move_possible(move_name):
                estimated_type = opp_type1 # Assume STAB main type for now
            
            if estimated_type:
                eff = get_type_effectiveness(estimated_type, our_type1, our_type2)
                estimated_dmg = (estimated_power / 100.0) * eff * 0.35
            else:
                estimated_dmg = 0.25
            
            embedding[13 + i * 2] = prob
            embedding[14 + i * 2] = estimated_dmg * prob
            max_incoming = max(max_incoming, estimated_dmg * prob)
            
        # ========== OPP THREAT TO BENCH (dims 21-25) ==========
        for i, pokemon in enumerate(battle.available_switches[:5]):
            poke_type1 = pokemon.type_1.name.lower() if pokemon.type_1 else None
            poke_type2 = pokemon.type_2.name.lower() if pokemon.type_2 else None
            
            max_threat = 0.0
            for atk_type in [opp_type1, opp_type2]:
                if atk_type:
                    eff = get_type_effectiveness(atk_type, poke_type1, poke_type2)
                    max_threat = max(max_threat, eff * 0.35)
            embedding[21 + i] = max_threat
            
        # ========== SPEED/PRIORITY INFO (dims 26-29) ==========
        # Re-calc speeds
        our_speed = calculate_speed(our_mon.base_stats.get('spe', 100), our_mon.level, our_mon.boosts.get('spe',0), False, False)
        opp_speed = calculate_speed(opp_mon.base_stats.get('spe', 100), opp_mon.level, opp_mon.boosts.get('spe',0), False, False)
        
        embedding[26] = 1.0 if our_speed > opp_speed else 0.0
        embedding[27] = (our_speed - opp_speed) / 400.0
        
        has_priority = any(safe_get_priority(m) > 0 for m in battle.available_moves if m)
        embedding[28] = 1.0 if has_priority else 0.0
        embedding[29] = our_mon.current_hp_fraction if our_mon.current_hp_fraction else 0.0
        
        # ========== SETUP PROJECTIONS (dims 30-35) ==========
        for i in range(2): # Top 2 moves
             base_dmg = embedding[i * 2 + 1]
             embedding[30 + i] = base_dmg * 1.5
             embedding[32 + i] = base_dmg * 2.0
        
        # Safe to setup? (Incoming < 50% HP)
        current_hp = our_mon.current_hp_fraction if our_mon.current_hp_fraction else 1.0
        embedding[34] = 1.0 if max_incoming < current_hp * 0.5 else 0.0
        
        opp_boosts = sum(max(0, v) for v in opp_mon.boosts.values()) if opp_mon.boosts else 0
        embedding[35] = opp_boosts / 6.0
        
        # ========== OHKO/2HKO FLAGS (dims 36-39) ==========
        embedding[36] = 1.0 if max_our_dmg >= 1.0 else 0.0
        embedding[37] = 1.0 if max_our_dmg >= 0.5 else 0.0
        embedding[38] = 1.0 if max_incoming >= 1.0 else 0.0
        embedding[39] = 1.0 if max_incoming >= 0.5 else 0.0
        
        return embedding


class OpponentMovesEncoder:
    """Encodes opponent's revealed moves."""
    
    OPP_MOVE_DIM = 21
    SIZE = 21 * 4 # 84
    
    def encode(self, battle: AbstractBattle) -> np.ndarray:
        embedding = np.zeros(self.SIZE, dtype=np.float32)
        
        opp = battle.opponent_active_pokemon
        if opp is None:
            return embedding
        
        our_mon = battle.active_pokemon
        our_type1 = our_mon.type_1.name.lower() if our_mon and our_mon.type_1 else None
        our_type2 = our_mon.type_2.name.lower() if our_mon and our_mon.type_2 else None
        
        known_moves = list(opp.moves.values())[:4] if opp.moves else []
        
        for i, move in enumerate(known_moves):
            base_idx = i * self.OPP_MOVE_DIM
            idx = base_idx
            
            # Move type (18 dims)
            move_type = move.type.name.lower() if move.type else None
            embedding[idx:idx+18] = type_to_onehot(move_type)
            idx += 18
            
            # Base power (1 dim)
            bp = safe_get_base_power(move)
            embedding[idx] = bp / 200.0
            idx += 1
            
            # Priority (1 dim)
            prio = safe_get_priority(move)
            embedding[idx] = (prio + 7) / 14.0
            idx += 1
            
            # Effectiveness vs us (1 dim)
            if move.type:
                eff = get_type_effectiveness(move.type.name.lower(), our_type1, our_type2)
                embedding[idx] = eff / 4.0
            idx += 1
        
        return embedding

class MatchupEncoder:
    """Encodes switch candidate matchups."""
    
    SWITCH_MATCHUP_DIM = 5
    SIZE = 5 * 5 # 25
    
    def encode(self, battle: AbstractBattle) -> np.ndarray:
        embedding = np.zeros(self.SIZE, dtype=np.float32)
        
        opp = battle.opponent_active_pokemon
        if opp is None:
            return embedding
            
        opp_type1 = opp.type_1.name.lower() if opp.type_1 else None
        opp_type2 = opp.type_2.name.lower() if opp.type_2 else None
        opp_speed = opp.base_stats.get('spe', 100) if opp.base_stats else 100
        
        for i, pokemon in enumerate(battle.available_switches[:5]):
            base_idx = i * self.SWITCH_MATCHUP_DIM
            
            our_type1 = pokemon.type_1.name.lower() if pokemon.type_1 else None
            our_type2 = pokemon.type_2.name.lower() if pokemon.type_2 else None
            
            # Max outgoing damage (effectiveness)
            max_outgoing = 1.0
            for atk_type in [our_type1, our_type2]:
                if atk_type:
                    eff = get_type_effectiveness(atk_type, opp_type1, opp_type2)
                    max_outgoing = max(max_outgoing, eff)
            
            # Max incoming damage
            max_incoming = 1.0
            for atk_type in [opp_type1, opp_type2]:
                if atk_type:
                    eff = get_type_effectiveness(atk_type, our_type1, our_type2)
                    max_incoming = max(max_incoming, eff)
            
            # Matchup score
            embedding[base_idx] = (max_outgoing - max_incoming) / 4.0 + 0.5
            
            # Speed tier
            our_speed = pokemon.base_stats.get('spe', 100) if pokemon.base_stats else 100
            embedding[base_idx + 1] = 1.0 if our_speed > opp_speed else 0.0
            
            # HP remaining
            embedding[base_idx + 2] = pokemon.current_hp_fraction if pokemon.current_hp_fraction else 0.0
            
            # Type immunity
            has_immunity = 0.0
            for atk_type in [opp_type1, opp_type2]:
                if atk_type:
                    eff = get_type_effectiveness(atk_type, our_type1, our_type2)
                    if eff == 0:
                        has_immunity = 1.0
                        break
            embedding[base_idx + 3] = has_immunity
            
            # Setup opportunity
            setup_opportunity = 0.0
            if max_incoming <= 0.5 and max_outgoing >= 1.0:
                 setup_opportunity = 1.0
            elif max_incoming <= 1.0 and max_outgoing >= 2.0:
                 setup_opportunity = 0.75
            embedding[base_idx + 4] = setup_opportunity
            
        return embedding

class BeliefEncoder:
    """Encodes belief state for opponent pokemon."""
    
    BELIEF_DIM = 16
    SIZE = 16 * 6 # 96
    
    def __init__(self, belief_tracker: BeliefTracker):
        self.belief_tracker = belief_tracker
        
    def encode(self, battle: AbstractBattle) -> np.ndarray:
        embedding = np.zeros(self.SIZE, dtype=np.float32)
        
        for i, (slot, pokemon) in enumerate(list(battle.opponent_team.items())[:6]):
            base_idx = i * self.BELIEF_DIM
            species = pokemon.species if pokemon else None
            
            if species:
                belief = self.belief_tracker.get_or_create_belief(species)
                
                # Role probs (0-3)
                if belief.role_probs:
                    sorted_roles = sorted(belief.role_probs.values(), reverse=True)
                    for j, prob in enumerate(sorted_roles[:4]):
                        embedding[base_idx + j] = prob
                
                # Move probs (4-7)
                move_probs = belief.get_unrevealed_move_probs()
                if move_probs:
                    sorted_moves = sorted(move_probs.values(), reverse=True)
                    for j, prob in enumerate(sorted_moves[:4]):
                         embedding[base_idx + 4 + j] = prob
                         
                # Item probs (8-10)
                item_probs = belief.get_item_probs()
                if item_probs:
                    sorted_items = sorted(item_probs.values(), reverse=True)
                    for j, prob in enumerate(sorted_items[:3]):
                         embedding[base_idx + 8 + j] = prob
                
                # Tera predicted (11-13)
                if belief.species_data and 'roles' in belief.species_data:
                    tera_probs = {}
                    for role, role_data in belief.species_data['roles'].items():
                         role_prob = belief.role_probs.get(role, 0)
                         tera_types = role_data.get('teraTypes', [])
                         for tera in tera_types:
                             tera_probs[tera] = tera_probs.get(tera, 0) + role_prob / len(tera_types) if tera_types else 0
                    
                    sorted_tera = sorted(tera_probs.values(), reverse=True)
                    for j, prob in enumerate(sorted_tera[:3]):
                         embedding[base_idx + 11 + j] = prob
                         
                # Moves revealed count (14)
                embedding[base_idx + 14] = len(belief.observed_moves) / 4.0
                
                # Entropy (15)
                # Normalize logic: max entropy for 4 roles is log2(4)=2.0. Scale to roughly 0-1.
                embedding[base_idx + 15] = belief.get_role_entropy() / 2.0
                
        return embedding

class MetaEncoder:
    """Encodes game meta context."""
    
    SIZE = 23 # META_DIM
    
    def encode(self, battle: AbstractBattle) -> np.ndarray:
        embedding = np.zeros(self.SIZE, dtype=np.float32)
        
        # Turn count
        embedding[0] = min(battle.turn, 100) / 100.0
        
        # Pokemon counts
        alive_ours = sum(1 for p in battle.team.values() if not p.fainted)
        revealed_opp = len(battle.opponent_team)
        fainted_opp = sum(1 for p in battle.opponent_team.values() if p.fainted)
        remaining_opp = 6 - fainted_opp
        
        embedding[1] = alive_ours / 6.0
        embedding[2] = revealed_opp / 6.0
        embedding[3] = fainted_opp / 6.0
        embedding[4] = remaining_opp / 6.0
        
        # HP sums
        hp_ours = sum(p.current_hp_fraction for p in battle.team.values() if p.current_hp_fraction)
        hp_opp_revealed = sum(p.current_hp_fraction for p in battle.opponent_team.values() if p.current_hp_fraction)
        unrevealed_count = 6 - revealed_opp
        hp_opp_estimated = hp_opp_revealed + unrevealed_count
        
        embedding[5] = hp_ours / 6.0
        embedding[6] = hp_opp_revealed / 6.0
        embedding[7] = hp_opp_estimated / 6.0
        embedding[8] = (hp_ours - hp_opp_estimated) / 6.0
        embedding[9] = (alive_ours - remaining_opp) / 6.0
        
        embedding[10] = 1.0 if battle.can_tera else 0.0
        
        opp_used_tera = 0.0
        for p in battle.opponent_team.values():
             if hasattr(p, 'is_terastallized') and p.is_terastallized:
                 opp_used_tera = 1.0
                 break
        embedding[11] = opp_used_tera
        
        embedding[12] = 1.0 if alive_ours > remaining_opp else 0.0 # Winning
        embedding[13] = 1.0 if alive_ours < remaining_opp else 0.0 # Losing
        embedding[14] = 1.0 if (battle.turn >= 50 or (alive_ours + remaining_opp) <= 6) else 0.0 # Late game
        
        # Lock status - Our
        our_choice_locked = 0.0
        our_mon = battle.active_pokemon
        if our_mon and our_mon.item and 'choice' in our_mon.item.lower().replace(' ', ''):
             # Heuristic check
             if hasattr(battle, '_last_request') or len(battle.team) > 0: # Simple existence check
                 our_choice_locked = 1.0 if battle.turn > 1 else 0.0
        embedding[15] = our_choice_locked
        
        # Lock status - Opp
        opp_choice_locked = 0.0
        opp_mon = battle.opponent_active_pokemon
        if opp_mon and opp_mon.item and 'choice' in opp_mon.item.lower().replace(' ', ''):
             if len(opp_mon.moves) >= 1:
                 opp_choice_locked = 1.0
        embedding[16] = opp_choice_locked
        
        # Tera defensive/offensive hints
        tera_def = 0.0
        tera_off = 0.0
        if battle.can_tera and our_mon:
             our_tera = getattr(our_mon, 'tera_type', None)
             if our_tera and opp_mon:
                 tera_str = our_tera.name.lower()
                 # Def
                 opp_stabs = []
                 if opp_mon.type_1: opp_stabs.append(opp_mon.type_1.name.lower())
                 if opp_mon.type_2: opp_stabs.append(opp_mon.type_2.name.lower())
                 for stab in opp_stabs:
                     if get_type_effectiveness(stab, tera_str, None) <= 0.5:
                         tera_def = 1.0
                         break
                 # Off
                 opp_t1 = opp_mon.type_1.name.lower() if opp_mon.type_1 else None
                 opp_t2 = opp_mon.type_2.name.lower() if opp_mon.type_2 else None
                 if get_type_effectiveness(tera_str, opp_t1, opp_t2) >= 2.0:
                      tera_off = 1.0
        embedding[17] = tera_def
        embedding[18] = tera_off
        
        # Protect risk
        opp_protect = 0.0
        if opp_mon and opp_mon.moves:
             for m in opp_mon.moves:
                 if m.lower().replace(' ','') in {'protect','detect','spikyshield'}:
                     opp_protect = 0.5
                     break
        embedding[19] = opp_protect
        
        # Weather benefits
        w_us = 0.0
        w_opp = 0.0
        if battle.weather:
             w_name = list(battle.weather.keys())[0].name.lower()
             if our_mon:
                 types = [t.name.lower() for t in our_mon.types if t]
                 if 'sun' in w_name and 'fire' in types: w_us = 1.0
                 elif 'rain' in w_name and 'water' in types: w_us = 1.0
             if opp_mon:
                 types = [t.name.lower() for t in opp_mon.types if t]
                 if 'sun' in w_name and 'fire' in types: w_opp = 1.0
                 elif 'rain' in w_name and 'water' in types: w_opp = 1.0
        embedding[20] = w_us
        embedding[21] = w_opp
        
        # Critical HP
        crit_hp = 0.0
        if our_mon and our_mon.current_hp_fraction < 0.25: crit_hp = 1.0
        embedding[22] = crit_hp
        
        return embedding

class ActionMaskEncoder:
    """Encodes legal action mask."""
    
    SIZE = 26
    
    def encode(self, battle: AbstractBattle) -> np.ndarray:
        mask = np.zeros(self.SIZE, dtype=np.float32)
        
        # Moves 0-3
        for i, move in enumerate(battle.available_moves[:4]):
            mask[i] = 1.0
            
        # Switches 4-8
        for i, pokemon in enumerate(battle.available_switches[:5]):
            if not battle.trapped:
                mask[4 + i] = 1.0
                
        # Terastallize 9-12
        if battle.can_tera:
             for i, move in enumerate(battle.available_moves[:4]):
                 mask[9 + i] = 1.0
                 
        return mask
