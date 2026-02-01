from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ShadowPokemon:
    """
    A wrapper around a real Pokemon object that overrides specific attributes.
    All non-overridden attributes are delegated to the real object.
    """
    def __init__(self, real_pokemon, overrides: Dict[str, Any]):
        self._real_pokemon = real_pokemon
        self._overrides = overrides
        
    def __getattr__(self, name):
        if name in self._overrides:
            return self._overrides[name]
        return getattr(self._real_pokemon, name)
        
    @property
    def stats(self):
        return self._overrides.get('stats', self._real_pokemon.stats)
        
    @property
    def item(self):
        return self._overrides.get('item', self._real_pokemon.item)
        
    @property
    def ability(self):
        return self._overrides.get('ability', self._real_pokemon.ability)
        
    @property
    def level(self):
        return self._overrides.get('level', self._real_pokemon.level)


class ShadowBattle:
    """
    A lightweight wrapper around a real Battle object that intercepts
    get_pokemon() calls to return 'Shadow' (mock) Pokemon objects.
    
    This allows us to run damage calculations with hypothetical stats
    (from belief key) without mutating the real battle state or 
    creating deep copies of the entire Battle object.
    
    All other attribute accesses are delegated to the real battle.
    """
    
    def __init__(self, real_battle, shadow_pokemon_map: Dict[str, Any]):
        """
        Args:
            real_battle: The actual poke_env Battle object
            shadow_pokemon_map: Dictionary mapping identifiers (e.g. 'p1: Pikachu') 
                              to ShadowPokemon/MockPokemon objects.
        """
        self._real_battle = real_battle
        self._shadow_map = shadow_pokemon_map
        
    def get_pokemon(self, identifier: str):
        """
        Intercepts pokemon lookup. Returns shadow object if mapped,
        otherwise delegates to real battle.
        """
        if identifier in self._shadow_map:
            return self._shadow_map[identifier]
        return self._real_battle.get_pokemon(identifier)
    
    @property
    def opponent_active_pokemon(self):
        """
        Explicitly intercept this property as it's often used directly.
        We need to return the shadow version if the active pokemon is shadowed.
        """
        # We need to know WHICH identifier corresponds to opponent active
        # The real battle knows.
        real_opp = self._real_battle.opponent_active_pokemon
        if not real_opp:
            return None
            
        # We need the identifier (e.g. 'p2: Scizor')
        # We can try to construct it or ask the real object
        opponent_role = self._real_battle.opponent_role or "p2"
        try:
             # Try to get identifier using the role
             ident = real_opp.identifier(opponent_role)
             if ident in self._shadow_map:
                 return self._shadow_map[ident]
        except Exception:
             pass
             
        # Fallback: check if any shadow object wraps the real opponent?
        # No, just return real
        return real_opp

    def __getattr__(self, name):
        """Delegate all other attributes to the real battle."""
        return getattr(self._real_battle, name)
