"""
Utility functions for the Gen 9 Random Battle RL bot.
Includes type mappings, data loading, and helper functions.
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
import numpy as np

# Pokemon type effectiveness chart
# Order: Normal, Fire, Water, Electric, Grass, Ice, Fighting, Poison, 
#        Ground, Flying, Psychic, Bug, Rock, Ghost, Dragon, Dark, Steel, Fairy
TYPE_LIST = [
    "normal", "fire", "water", "electric", "grass", "ice",
    "fighting", "poison", "ground", "flying", "psychic", "bug",
    "rock", "ghost", "dragon", "dark", "steel", "fairy"
]

TYPE_TO_ID = {t: i for i, t in enumerate(TYPE_LIST)}
NUM_TYPES = len(TYPE_LIST)

# Type effectiveness matrix (attacker x defender)
# 0 = immune, 0.5 = not very effective, 1 = neutral, 2 = super effective
TYPE_CHART = np.array([
    # Nor Fir Wat Ele Gra Ice Fig Poi Gro Fly Psy Bug Roc Gho Dra Dar Ste Fai
    [1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, .5,  0,  1,  1, .5,  1],  # Normal
    [1, .5, .5,  1,  2,  2,  1,  1,  1,  1,  1,  2, .5,  1, .5,  1,  2,  1],  # Fire
    [1,  2, .5,  1, .5,  1,  1,  1,  2,  1,  1,  1,  2,  1, .5,  1,  1,  1],  # Water
    [1,  1,  2, .5, .5,  1,  1,  1,  0,  2,  1,  1,  1,  1, .5,  1,  1,  1],  # Electric
    [1, .5,  2,  1, .5,  1,  1, .5,  2, .5,  1, .5,  2,  1, .5,  1, .5,  1],  # Grass
    [1, .5, .5,  1,  2, .5,  1,  1,  2,  2,  1,  1,  1,  1,  2,  1, .5,  1],  # Ice
    [2,  1,  1,  1,  1,  2,  1, .5,  1, .5, .5, .5,  2,  0,  1,  2,  2, .5],  # Fighting
    [1,  1,  1,  1,  2,  1,  1, .5, .5,  1,  1,  1, .5, .5,  1,  1,  0,  2],  # Poison
    [1,  2,  1,  2, .5,  1,  1,  2,  1,  0,  1, .5,  2,  1,  1,  1,  2,  1],  # Ground
    [1,  1,  1, .5,  2,  1,  2,  1,  1,  1,  1,  2, .5,  1,  1,  1, .5,  1],  # Flying
    [1,  1,  1,  1,  1,  1,  2,  2,  1,  1, .5,  1,  1,  1,  1,  0, .5,  1],  # Psychic
    [1, .5,  1,  1,  2,  1, .5, .5,  1, .5,  2,  1,  1, .5,  1,  2, .5, .5],  # Bug
    [1,  2,  1,  1,  1,  2, .5,  1, .5,  2,  1,  2,  1,  1,  1,  1, .5,  1],  # Rock
    [0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  2,  1,  1,  2,  1, .5,  1,  1],  # Ghost
    [1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  2,  1, .5,  0],  # Dragon
    [1,  1,  1,  1,  1,  1, .5,  1,  1,  1,  2,  1,  1,  2,  1, .5,  1, .5],  # Dark
    [1, .5, .5, .5,  1,  2,  1,  1,  1,  1,  1,  1,  2,  1,  1,  1, .5,  2],  # Steel
    [1, .5,  1,  1,  1,  1,  2, .5,  1,  1,  1,  1,  1,  1,  2,  2, .5,  1],  # Fairy
], dtype=np.float32)


@lru_cache(maxsize=1024)
def get_type_effectiveness(atk_type: str, def_type1: str, def_type2: Optional[str] = None) -> float:
    """Calculate type effectiveness multiplier. Cached for performance."""
    atk_lower = atk_type.lower() if atk_type else 'normal'
    if atk_lower not in TYPE_TO_ID:
        return 1.0
    atk_id = TYPE_TO_ID[atk_lower]
    
    mult = 1.0
    if def_type1:
        def1_lower = def_type1.lower()
        if def1_lower in TYPE_TO_ID:
            mult *= TYPE_CHART[atk_id, TYPE_TO_ID[def1_lower]]
    if def_type2:
        def2_lower = def_type2.lower()
        if def2_lower in TYPE_TO_ID:
            mult *= TYPE_CHART[atk_id, TYPE_TO_ID[def2_lower]]
    
    return mult


# Ability-based type immunities
ABILITY_IMMUNITIES: Dict[str, Set[str]] = {
    'levitate': {'ground'},
    'flashfire': {'fire'},
    'waterabsorb': {'water'},
    'voltabsorb': {'electric'},
    'lightningrod': {'electric'},
    'stormdrain': {'water'},
    'sapsipper': {'grass'},
    'motordrive': {'electric'},
    'dryskin': {'water'},
    'eartheater': {'ground'},
    'heatproof': set(),  # Not immune, just resistant
    'thickfat': set(),   # Not immune, just resistant
}


def is_immune_by_ability(move_type: str, ability: Optional[str]) -> bool:
    """
    Check if ability grants immunity to move type.
    
    Args:
        move_type: The attacking move's type
        ability: The defender's ability
        
    Returns:
        True if ability makes the pokemon immune to this move type
    """
    if not ability:
        return False
    ability_norm = ability.lower().replace(' ', '').replace('-', '')
    type_norm = move_type.lower() if move_type else ''
    immune_types = ABILITY_IMMUNITIES.get(ability_norm, set())
    return type_norm in immune_types


def type_to_onehot(type_name: Optional[str]) -> np.ndarray:
    """Convert Pokemon type to one-hot encoding."""
    arr = np.zeros(NUM_TYPES, dtype=np.float32)
    if type_name:
        t_id = TYPE_TO_ID.get(type_name.lower())
        if t_id is not None:
            arr[t_id] = 1.0
    return arr


# Status conditions
STATUS_LIST = ["none", "brn", "par", "slp", "frz", "psn", "tox"]
STATUS_TO_ID = {s: i for i, s in enumerate(STATUS_LIST)}
NUM_STATUS = len(STATUS_LIST)


def status_to_onehot(status: Optional[str]) -> np.ndarray:
    """Convert status to one-hot encoding."""
    arr = np.zeros(NUM_STATUS, dtype=np.float32)
    if status is None:
        arr[0] = 1.0
    elif status.lower() in STATUS_TO_ID:
        arr[STATUS_TO_ID[status.lower()]] = 1.0
    else:
        arr[0] = 1.0  # Unknown status treated as none
    return arr


# Weather conditions
WEATHER_LIST = ["none", "sunnyday", "raindance", "sandstorm", "hail", "snow"]
WEATHER_TO_ID = {w: i for i, w in enumerate(WEATHER_LIST)}
NUM_WEATHER = len(WEATHER_LIST)


# Global Field conditions (Terrains + Rooms + Gravity)
GLOBAL_FIELD_LIST = [
    "none", 
    "electricterrain", "grassyterrain", "mistyterrain", "psychicterrain",
    "trickroom", "gravity", "magicroom", "wonderroom"
]
GLOBAL_FIELD_TO_ID = {t: i for i, t in enumerate(GLOBAL_FIELD_LIST)}
NUM_GLOBAL_FIELD = len(GLOBAL_FIELD_LIST)


# Side conditions (hazards/screens)
SIDE_CONDITIONS = [
    "spikes", "stealthrock", "stickyweb", "toxicspikes",
    "reflect", "lightscreen", "auroraveil", "tailwind"
]
NUM_SIDE_CONDITIONS = len(SIDE_CONDITIONS)


def load_pokemon_data(filepath: str = "gen9randombattle.json") -> Dict[str, Any]:
    """Load Pokemon role/set data from JSON file."""
    path = Path(filepath)
    if not path.exists():
        # Try relative to project root
        path = Path(__file__).parent.parent / filepath
    
    with open(path, 'r') as f:
        return json.load(f)


def normalize_species_name(name: str) -> str:
    """Normalize Pokemon species name for lookup."""
    # Remove special characters and lowercase
    return name.lower().replace("-", "").replace(" ", "").replace(".", "").replace("'", "")


def get_species_from_pokemon(pokemon) -> str:
    """Extract normalized species name from a Pokemon object."""
    if hasattr(pokemon, 'species'):
        return normalize_species_name(pokemon.species)
    return normalize_species_name(str(pokemon))


# Stat boost multipliers
BOOST_MULTIPLIERS = {
    -6: 2/8, -5: 2/7, -4: 2/6, -3: 2/5, -2: 2/4, -1: 2/3,
    0: 1.0,
    1: 3/2, 2: 4/2, 3: 5/2, 4: 6/2, 5: 7/2, 6: 8/2
}


def boost_to_multiplier(boost: int) -> float:
    """Convert stat boost stage to multiplier."""
    boost = max(-6, min(6, boost))
    return BOOST_MULTIPLIERS[boost]


def boosts_to_array(boosts: Dict[str, int]) -> np.ndarray:
    """Convert boosts dict to normalized array.
    
    Clamps values to [-6, 6] to prevent the agent from perceiving
    or chasing boosts beyond the game's maximum.
    """
    # Order: atk, def, spa, spd, spe, accuracy, evasion
    stat_order = ['atk', 'def', 'spa', 'spd', 'spe', 'accuracy', 'evasion']
    arr = np.zeros(7, dtype=np.float32)
    for i, stat in enumerate(stat_order):
        if stat in boosts:
            # Clamp to valid range and normalize to [-1, 1]
            clamped = max(-6, min(6, boosts[stat]))
            arr[i] = clamped / 6.0
    return arr


def calculate_speed(base_speed: int, level: int, boost: int = 0, 
                    paralyzed: bool = False, tailwind: bool = False, is_scarfed: bool = False) -> float:
    """Calculate effective speed for speed tier comparison."""
    # Simplified: assume 31 IVs, 85 EVs (random battle typical)
    stat = int(((2 * base_speed + 31 + 85 // 4) * level / 100 + 5))
    
    # Apply boost
    stat = int(stat * boost_to_multiplier(boost))
    
    # Paralysis halves speed
    if paralyzed:
        stat = stat // 2
    
    # Tailwind doubles speed
    if tailwind:
        stat = stat * 2
        
    # Choice Scarf 1.5x
    if is_scarfed:
        stat = int(stat * 1.5)
    
    return float(stat)


# Move categories
MOVE_CATEGORY_LIST = ["physical", "special", "status"]
MOVE_CATEGORY_TO_ID = {c: i for i, c in enumerate(MOVE_CATEGORY_LIST)}


def move_category_to_onehot(category: str) -> np.ndarray:
    """Convert move category to one-hot encoding."""
    arr = np.zeros(3, dtype=np.float32)
    cat = category.lower() if category else "status"
    if cat in MOVE_CATEGORY_TO_ID:
        arr[MOVE_CATEGORY_TO_ID[cat]] = 1.0
    return arr


# =============================================================================
# Strategic Ability Categories
# =============================================================================

# Abilities that grant type immunities - crucial for matchup calculations
IMMUNITY_ABILITIES = {
    'levitate': 'ground',
    'flashfire': 'fire',
    'waterabsorb': 'water',
    'voltabsorb': 'electric',
    'lightningrod': 'electric',
    'motordrive': 'electric',
    'sapsipper': 'grass',
    'dryskin': 'water',
    'stormdrain': 'water',
    'eartheater': 'ground',
    'wellbakedbody': 'fire',
}

# Abilities that affect priority
PRIORITY_ABILITIES = {
    'prankster': 'status',      # +1 priority on status moves
    'galewings': 'flying',      # +1 priority on Flying at full HP
    'triage': 'healing',        # +3 priority on healing moves
    'stall': 'last',            # Always moves last
    'myceliummight': 'status',  # Status bypasses abilities but moves last
}

# Weather-setting abilities
WEATHER_ABILITIES = {
    'drought': 'sunnyday',
    'drizzle': 'raindance',
    'sandstream': 'sandstorm',
    'snowwarning': 'snow',
    'orichalcumpulse': 'sunnyday',
    'hadronengine': 'electricterrain',
}

# Terrain-setting abilities
TERRAIN_ABILITIES = {
    'electricsurge': 'electricterrain',
    'psychicsurge': 'psychicterrain',
    'grassysurge': 'grassyterrain',
    'mistysurge': 'mistyterrain',
    'seedsower': 'grassyterrain',
}

# Abilities that punish contact moves
CONTACT_PUNISHMENT_ABILITIES = {
    'roughskin', 'ironbarbs', 'flamebody', 'static', 'effectspore',
    'poisonpoint', 'cutecharm', 'gooey', 'tanglinghair',
}

# Abilities that boost specific stats
BOOST_ABILITIES = {
    'intimidate': 'atk_drop',
    'clearbody': 'stat_protection',
    'whitesmoke': 'stat_protection',
    'competitive': 'spa_boost',
    'defiant': 'atk_boost',
    'speedboost': 'spe_boost',
    'beastboost': 'highest_boost',
    'moody': 'random_boost',
}


def get_ability_flags(ability: Optional[str]) -> np.ndarray:
    """
    Get strategic flags for an ability.
    
    Returns 8 flags:
    - grants_immunity (1): Ability grants type immunity
    - affects_priority (1): Ability changes move priority  
    - sets_weather (1): Ability sets weather
    - sets_terrain (1): Ability sets terrain
    - punishes_contact (1): Ability punishes contact moves
    - boosts_stats (1): Ability boosts stats in some way
    - is_intimidate (1): Specifically Intimidate (very common)
    - is_known (1): Ability is known at all
    """
    flags = np.zeros(8, dtype=np.float32)
    
    if not ability:
        return flags
    
    ability_lower = ability.lower().replace(' ', '').replace('-', '')
    
    flags[0] = 1.0 if ability_lower in IMMUNITY_ABILITIES else 0.0
    flags[1] = 1.0 if ability_lower in PRIORITY_ABILITIES else 0.0
    flags[2] = 1.0 if ability_lower in WEATHER_ABILITIES else 0.0
    flags[3] = 1.0 if ability_lower in TERRAIN_ABILITIES else 0.0
    flags[4] = 1.0 if ability_lower in CONTACT_PUNISHMENT_ABILITIES else 0.0
    flags[5] = 1.0 if ability_lower in BOOST_ABILITIES else 0.0
    flags[6] = 1.0 if ability_lower == 'intimidate' else 0.0
    flags[7] = 1.0  # Is known
    
    return flags


def get_immunity_type(ability: Optional[str]) -> Optional[str]:
    """Get the type this ability grants immunity to, if any."""
    if not ability:
        return None
    ability_lower = ability.lower().replace(' ', '').replace('-', '')
    return IMMUNITY_ABILITIES.get(ability_lower)


# =============================================================================
# Strategic Item Categories
# =============================================================================

# Choice items that lock you into one move
CHOICE_ITEMS = {'choiceband', 'choicespecs', 'choicescarf'}

# Items that boost damage
DAMAGE_BOOST_ITEMS = {
    'lifeorb': 1.3,
    'choiceband': 1.5,
    'choicespecs': 1.5,
    'expertbelt': 1.2,
}

# Items that affect speed
SPEED_ITEMS = {
    'choicescarf': 1.5,
    'ironball': 0.5,
    'machobrace': 0.5,
    'laggingtail': 'last',
    'fullincense': 'last',
}

# Healing items
HEALING_ITEMS = {
    'leftovers', 'blacksludge', 'sitrusberry', 'aguavberry', 
    'figyberry', 'iapapaberry', 'magoberry', 'wikiberry', 'shellbell'
}

# Status immunity items
STATUS_IMMUNITY_ITEMS = {
    'lumberry', 'chestoberry', 'pechaberry', 'rawstberry', 
    'cheriberry', 'aspearberry', 'persimberry', 'safetygoggles'
}

# Contact punishment
CONTACT_PUNISH_ITEMS = {'rockyhelmet', 'stickybarb'}

# Type immunity items
TYPE_IMMUNITY_ITEMS = {'airballoon'}

# Bulk items
BULK_ITEMS = {'assaultvest', 'eviolite'}

# Survival items (One-time protection)
SURVIVAL_ITEMS = {'focussash', 'focusband'} 


def get_item_flags(item: Optional[str]) -> np.ndarray:
    """
    Get strategic flags for an item.
    
    Returns 10 flags:
    1. is_choice (1): Locks moves (Choice Band/Specs/Scarf)
    2. is_damage_boost (1): Boosts damage (Life Orb/Band/Specs/Belt/Plates)
    3. is_speed_boost (1): Boosts speed (Scarf)
    4. is_healing (1): Restores HP (Leftovers/Berries)
    5. is_survival (1): Prevents OHKO (Sash/Band)
    6. is_bulk (1): Boosts Def/SpD (Vest/Eviolite)
    7. is_punish (1): Punishes contact (Helmet)
    8. is_status_immune (1): Prevents status (Lum/Goggles)
    9. is_type_immune (1): Grants immunity (Balloon)
    10. is_known (1): Item is known
    """
    flags = np.zeros(10, dtype=np.float32)
    
    if not item:
        return flags
    
    item_lower = item.lower().replace(' ', '').replace('-', '')
    
    # 1. Choice Lock
    flags[0] = 1.0 if item_lower in CHOICE_ITEMS else 0.0
    
    # 2. Damage Boost (Choice items are also damage boosters usually)
    # We include lifeorb, expertbelt, and existing damage lists
    flags[1] = 1.0 if (item_lower in DAMAGE_BOOST_ITEMS or item_lower in {'lifeorb', 'expertbelt'}) else 0.0
    
    # 3. Speed Boost
    flags[2] = 1.0 if item_lower in SPEED_ITEMS else 0.0
    
    # 4. Healing
    flags[3] = 1.0 if item_lower in HEALING_ITEMS else 0.0
    
    # 5. Survival (Sash)
    flags[4] = 1.0 if item_lower in SURVIVAL_ITEMS else 0.0
    
    # 6. Bulk (Vest)
    flags[5] = 1.0 if item_lower in BULK_ITEMS else 0.0
    
    # 7. Punish (Helmet)
    flags[6] = 1.0 if item_lower in CONTACT_PUNISH_ITEMS else 0.0
    
    # 8. Status Immunity (Lum)
    flags[7] = 1.0 if item_lower in STATUS_IMMUNITY_ITEMS else 0.0
    
    # 9. Type Immunity (Balloon)
    flags[8] = 1.0 if item_lower in TYPE_IMMUNITY_ITEMS else 0.0
    
    # 10. Is Known
    flags[9] = 1.0
    
    return flags


def is_choice_item(item: Optional[str]) -> bool:
    """Check if item is a Choice item that locks into one move."""
    if not item:
        return False
    return item.lower().replace(' ', '').replace('-', '') in CHOICE_ITEMS


# =============================================================================
# Move Classification
# =============================================================================

class MoveClassifier:
    """
    Classifies moves into strategic categories.
    
    Provides centralized move set definitions used across the codebase
    for identifying setup moves, recovery, hazards, etc.
    """
    
    # Setup/Boost moves
    BOOST_MOVES = frozenset({
        'swordsdance', 'nastyplot', 'dragondance', 'calmmind',
        'bulkup', 'irondefense', 'agility', 'quiverdance',
        'shellsmash', 'growth', 'coil', 'curse', 'workup',
        'howl', 'bellydrum', 'tailglow', 'geomancy', 'shiftgear',
        'honeclaws', 'rockpolish', 'autotomize', 'cottonguard',
        'cosmicpower', 'amnesia', 'acidarmor', 'barrier', 'stockpile'
    })
    
    # Recovery moves
    RECOVERY_MOVES = frozenset({
        'recover', 'softboiled', 'roost', 'moonlight',
        'morningsun', 'synthesis', 'slackoff', 'milkdrink',
        'shoreup', 'wish', 'rest', 'healorder', 'strengthsap',
        'leechseed', 'drainpunch', 'gigadrain', 'hornleech',
        'absorb', 'megadrain', 'drainingkiss', 'oblivionwing',
        'painsplit', 'ingrain', 'aquaring', 'healingwish'
    })
    
    # Entry hazard moves
    HAZARD_MOVES = frozenset({
        'stealthrock', 'spikes', 'toxicspikes', 'stickyweb'
    })
    
    # Hazard removal moves
    HAZARD_REMOVAL = frozenset({
        'defog', 'rapidspin', 'courtchange', 'tidyup', 'mortalspin'
    })
    
    # Priority moves
    PRIORITY_MOVES = frozenset({
        'extremespeed', 'aquajet', 'machpunch', 'bulletpunch',
        'iceshard', 'shadowsneak', 'suckerpunch', 'quickattack',
        'accelerock', 'firstimpression', 'fakeout', 'feint',
        'vacuumwave', 'watershuriken', 'jetpunch', 'grassyglide'
    })
    
    # Protect variants
    PROTECT_MOVES = frozenset({
        'protect', 'detect', 'kingsshield', 'spikyshield',
        'banefulbunker', 'obstruct', 'silktrap', 'burningbulwark'
    })
    
    # Pivoting moves
    PIVOT_MOVES = frozenset({
        'uturn', 'voltswitch', 'flipturn', 'partingshot',
        'batonpass', 'teleport', 'shedtail'
    })
    
    # Status-inflicting moves
    STATUS_INFLICTING = frozenset({
        'willowisp', 'thunderwave', 'toxic', 'spore',
        'sleeppowder', 'hypnosis', 'yawn', 'nuzzle',
        'glare', 'stunspore', 'poisonpowder'
    })
    
    @classmethod
    def normalize(cls, move_id: str) -> str:
        """Normalize move ID for comparison."""
        return move_id.lower().replace(' ', '').replace('-', '').replace("'", '')
    
    @classmethod
    def is_boost(cls, move_id: str) -> bool:
        """Check if move is a stat-boosting move."""
        return cls.normalize(move_id) in cls.BOOST_MOVES
    
    @classmethod
    def is_recovery(cls, move_id: str) -> bool:
        """Check if move is a recovery/healing move."""
        return cls.normalize(move_id) in cls.RECOVERY_MOVES
    
    @classmethod
    def is_hazard(cls, move_id: str) -> bool:
        """Check if move sets entry hazards."""
        return cls.normalize(move_id) in cls.HAZARD_MOVES
    
    @classmethod
    def is_hazard_removal(cls, move_id: str) -> bool:
        """Check if move removes hazards."""
        return cls.normalize(move_id) in cls.HAZARD_REMOVAL
    
    @classmethod
    def is_priority(cls, move_id: str) -> bool:
        """Check if move has positive priority."""
        return cls.normalize(move_id) in cls.PRIORITY_MOVES
    
    @classmethod
    def is_protect(cls, move_id: str) -> bool:
        """Check if move is a protect variant."""
        return cls.normalize(move_id) in cls.PROTECT_MOVES
    
    @classmethod
    def is_pivot(cls, move_id: str) -> bool:
        """Check if move is a pivoting move."""
        return cls.normalize(move_id) in cls.PIVOT_MOVES
    
    @classmethod
    def is_status_inflicting(cls, move_id: str) -> bool:
        """Check if move inflicts status."""
        return cls.normalize(move_id) in cls.STATUS_INFLICTING

