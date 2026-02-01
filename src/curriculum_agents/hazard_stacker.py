"""
HazardStacker: Sets rocks turn 1, then punishes switches.
Teaches the agent that hazards and switching have costs.
"""

from poke_env.battle import AbstractBattle

from .base import BaseCurriculumPlayer


class HazardStacker(BaseCurriculumPlayer):
    """
    Sets Stealth Rock turn 1, then clicks hardest hit.
    
    BEHAVIOR:
    - Turn 1: Use Stealth Rock if available
    - After: Click strongest STAB move to punish switches
    LESSON: Agent learns hazard damage matters.
    EXPLOITABLE BY: Rapid Spin/Defog, playing aggressively.
    """
    
    HAZARD_MOVES = {'stealthrock', 'spikes', 'toxicspikes', 'stickyweb'}
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self._hazards_set removed (dead code)
    
    def choose_move(self, battle: AbstractBattle):
        # Log moveset on turn 1
        # Log moveset on turn 1
        if not self._logged_moveset and battle.turn == 1:
            self._logged_moveset = True
            # print(f"\n[HazardStacker] Turn 1 Moveset:")
            # print(f"  Active: {battle.active_pokemon.species if battle.active_pokemon else 'None'}")
            # for move in battle.available_moves:
            #     hazard_mark = " ⭐HAZARD" if move.id.lower() in self.HAZARD_MOVES else ""
            #     print(f"    - {move.id:20s} (Power: {move.base_power or 0:3d}){hazard_mark}")
            # print()
        
        # Check if we have a useful hazard move (one that isn't maxed out)
        useful_hazard = self._get_useful_hazard_move(battle)
        
        # Log decision every turn (silenced)
        # print(f"[HazardStacker] T{battle.turn}: UsefulHazard={useful_hazard.id if useful_hazard else 'None'}")
        
        # Try to set hazards if valuable
        if useful_hazard:
            # print(f"  -> SETTING HAZARDS: {useful_hazard.id}")
            return self.create_order(useful_hazard)
        
        # Hazards up or no hazard move: click hardest hit
        # print(f"  -> Clicking best move (hazards maxed/unavailable)")
        return self._click_best_move(battle)
    
    def _get_useful_hazard_move(self, battle: AbstractBattle):
        """Get a hazard-setting move if available and not maxed on opponent side."""
        for move in battle.available_moves:
            if move.id.lower() in self.HAZARD_MOVES:
                if not self._is_hazard_maxed(move.id, battle.opponent_side_conditions):
                    return move
        return None

    def _is_hazard_maxed(self, move_id: str, side_conditions: dict) -> bool:
        """Check if a specific hazard is already maxed out on the opponent's side."""
        move_id = move_id.lower()
        # Convert keys to strings to handle SideCondition enums safely
        conditions = {str(k).lower(): v for k, v in side_conditions.items()}
        
        if move_id == 'stealthrock':
            # Max 1 layer (Check value > 0)
            for k, layers in conditions.items():
                if 'stealth' in k and 'rock' in k:
                    return layers > 0
            return False
            
        if move_id == 'stickyweb':
            # Max 1 layer (Check value > 0)
            for k, layers in conditions.items():
                if 'sticky' in k and 'web' in k:
                    return layers > 0
            return False
            
        if move_id == 'spikes':
            # Max 3 layers
            # Note: poke-env usually stores Spikes count as the value
            for k, layers in conditions.items():
                if 'spikes' in k and 'toxic' not in k: 
                    return layers >= 3
            return False 
            
        if move_id == 'toxicspikes':
            # Max 2 layers
            for k, layers in conditions.items():
                if 'toxic' in k and 'spikes' in k:
                    return layers >= 2
            return False

        return False
    
    def _battle_finished_callback(self, battle):
        """Reset for new battle."""
        super()._battle_finished_callback(battle)
        # self._hazards_set removed
