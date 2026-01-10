"""
Team Validator: Validates generated teams against Pokemon Showdown data.

Extracts valid items, abilities, and moves from pokemon-showdown/data/*.ts
"""

import re
from pathlib import Path
from functools import lru_cache

# Path to pokemon-showdown data
SHOWDOWN_DATA_PATH = Path(__file__).parent.parent.parent / "pokemon-showdown" / "data"

DEBUG = False  # Enable verbose logging


@lru_cache(maxsize=1)
def get_valid_items() -> set:
    """Extract all valid item IDs from pokemon-showdown/data/items.ts"""
    items_file = SHOWDOWN_DATA_PATH / "items.ts"
    
    if not items_file.exists():
        print(f"[TeamValidator] Warning: items.ts not found at {items_file}")
        return set()
    
    content = items_file.read_text()
    
    # Extract item IDs (keys in the Items object)
    # Pattern: starts with tab, lowercase alphanumeric, ends with colon
    pattern = r'^\t([a-z0-9]+):'
    matches = re.findall(pattern, content, re.MULTILINE)
    
    valid_items = set(matches)
    print(f"[TeamValidator] Loaded {len(valid_items)} valid items from {items_file}")
    
    return valid_items


def normalize_item_id(item_name: str) -> str:
    """Convert an item name to its ID (lowercase, no spaces/special chars)"""
    # Remove spaces, hyphens, apostrophes, etc.
    return re.sub(r'[^a-z0-9]', '', item_name.lower())


def is_valid_item(item_name: str) -> bool:
    """Check if an item name is valid according to pokemon-showdown data"""
    valid_items = get_valid_items()
    
    if not valid_items:
        # If we can't load the data, assume valid (fail open)
        return True
    
    item_id = normalize_item_id(item_name)
    return item_id in valid_items


def validate_packed_team(packed_team: str) -> tuple[bool, list[str]]:
    """
    Validate a packed team string.
    
    Packed format (poke-env style):
    Pokemon are separated by ']'
    Each Pokemon has fields separated by '|':
    species|item|ability|moves|nature|evs|gender|ivs|shiny|level|...
    
    Returns:
        (is_valid, list_of_invalid_items)
    """
    if DEBUG:
        print(f"[TeamValidator] validate_packed_team CALLED")
        print(f"[TeamValidator] Packed team (first 200 chars): {packed_team[:200]}...")
    
    if not packed_team:
        print(f"[TeamValidator] Empty team!")
        return False, ["Empty team"]
    
    valid_items = get_valid_items()
    if not valid_items:
        print(f"[TeamValidator] No valid items loaded, assuming valid")
        return True, []  # Can't validate, assume valid
    
    invalid_items = []
    
    # Split by ] to get each Pokemon
    pokemon_entries = packed_team.split(']')
    
    if DEBUG:
        print(f"[TeamValidator] Found {len(pokemon_entries)} Pokemon entries")
    
    for i, entry in enumerate(pokemon_entries):
        if not entry.strip():
            continue
        
        # Split by | to get fields
        fields = entry.split('|')
        
        if DEBUG:
            print(f"[TeamValidator] Pokemon {i}: {len(fields)} fields, entry='{entry[:80]}...'")
        
        # Item is the 2nd field (index 1)
        if len(fields) > 1:
            item = fields[1].strip()
            
            if DEBUG:
                print(f"[TeamValidator] Pokemon {i} item field: '{item}'")
            
            if item:
                item_id = normalize_item_id(item)
                is_valid = item_id in valid_items
                
                if DEBUG:
                    print(f"[TeamValidator] Item '{item}' -> ID '{item_id}' -> Valid: {is_valid}")
                
                if not is_valid:
                    invalid_items.append(item)
    
    is_valid = len(invalid_items) == 0
    
    if DEBUG:
        print(f"[TeamValidator] Validation result: valid={is_valid}, invalid_items={invalid_items}")
    
    return is_valid, invalid_items


def validate_showdown_team(showdown_team: str) -> tuple[bool, list[str]]:
    """
    Validate a showdown-format team (before packing).
    
    This is more reliable than validating packed format.
    Looks for lines like: "Pokemon @ ItemName"
    """
    if DEBUG:
        print(f"[TeamValidator] validate_showdown_team CALLED")
    
    if not showdown_team:
        return False, ["Empty team"]
    
    valid_items = get_valid_items()
    if not valid_items:
        return True, []
    
    invalid_items = []
    
    # Pattern: "PokemonName @ ItemName"
    item_pattern = r'^.+\s*@\s*(.+)$'
    
    for line in showdown_team.split('\n'):
        match = re.match(item_pattern, line.strip())
        if match:
            item = match.group(1).strip()
            item_id = normalize_item_id(item)
            
            if DEBUG:
                print(f"[TeamValidator] Found item line: '{line.strip()}' -> item='{item}' -> id='{item_id}'")
            
            if item_id and item_id not in valid_items:
                invalid_items.append(item)
                if DEBUG:
                    print(f"[TeamValidator] INVALID ITEM: '{item}'")
    
    is_valid = len(invalid_items) == 0
    return is_valid, invalid_items
