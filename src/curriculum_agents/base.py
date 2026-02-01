"""
Base class for curriculum opponents.
Provides shared utilities for type effectiveness, speed calculation, etc.
"""

import random
from typing import Optional, List

from poke_env.data import GenData
from poke_env.player import Player
from poke_env.battle import AbstractBattle, Pokemon, Move
from poke_env.player.battle_order import BattleOrder

from ..utils import load_pokemon_data, get_type_effectiveness
from ..belief_tracker import BeliefTracker
from ..damage_calc import BeliefDamageCalculator


class BaseCurriculumPlayer(Player):
    """
    Base class for curriculum opponents.
    Provides shared utilities for type effectiveness, speed calculation, etc.
    """
    
    def __init__(self, **kwargs):
        # Extract pokemon_data before calling super, as Player doesn't accept it
        pokemon_data = kwargs.pop('pokemon_data', None) 
        
        # Extract teambuilder - we now pass it DIRECTLY to poke-env for per-battle rotation!
        teambuilder = kwargs.pop('teambuilder', None)
        
        # CRITICAL FIX: Pass the Teambuilder OBJECT directly to poke-env
        # This allows poke-env to call yield_team() for EACH NEW BATTLE
        # Instead of using the same team forever
        if teambuilder and 'team' not in kwargs:
            # Pass the Teambuilder object, not the string!
            kwargs['team'] = teambuilder
            # print(f"[DEBUG] Teambuilder OBJECT passed to {self.__class__.__name__} for per-battle rotation")
        else:
            if teambuilder:
                # print(f"[DEBUG] Teambuilder provided but 'team' already in kwargs")
                pass
            else:
                # print(f"[DEBUG] No teambuilder provided for {self.__class__.__name__}")
                pass
                
        super().__init__(**kwargs)
        self._rng = random.Random()
        self._last_opponent_move = None
        self._opponent_switched_last_turn = False
        self._logged_moveset = False
        
        # Memory leak fix: Only track current battle state
        self._current_battle_tag = None
        self._last_turn_data = {}
        
        # Initialize damage calculator components
        if pokemon_data:
             self.pokemon_data = pokemon_data
        else:
             self.pokemon_data = load_pokemon_data()
             
        self.belief_tracker = BeliefTracker(self.pokemon_data)
        self.damage_calculator = BeliefDamageCalculator(self.pokemon_data, self.belief_tracker)
        
        # Cache the FIRST team immediately to prevent "next_team returned None" startup race conditions
        self._cached_team = None
        if self._team and hasattr(self._team, 'yield_team'):
            try:
                self._cached_team = self._team.yield_team()
                # print(f"[DEBUG] {self.__class__.__name__} pre-cached team: {self._cached_team[:20].replace(chr(10), '|') if self._cached_team else 'None'}...")
            except Exception as e:
                print(f"[ERROR] Failed to pre-cache team: {e}")
    
    # =========================================================================
    # Type Effectiveness Utilities
    # =========================================================================

    async def accept_challenge(self, opponent_username: str, team: Optional[str] = None):
        """Override to debug team selection."""
        # Use our pre-cached team if available, otherwise generate new
        current_team = team or self._cached_team
        
        # If still no team, try to generate one (fallback)
        if not current_team and self._team and hasattr(self._team, 'yield_team'):
            try:
                current_team = self._team.yield_team()
                # print(f"[DEBUG] {self.__class__.__name__} FORCE-GENERATED team: {current_team[:20].replace(chr(10), '|') if current_team else 'None'}...")
            except Exception as e:
                print(f"[ERROR] Failed to yield team: {e}")
        
        # Log what we are using
        if current_team:
             # print(f"[DEBUG] {self.__class__.__name__} accepting challenge with team len={len(current_team)}")
             # print(current_team[:100] + "...")
             pass
        
        await super().accept_challenge(opponent_username, team=current_team)
        
        # ROTATE TEAM AFTER ACCEPTING: Prepare for NEXT battle
        if self._team and hasattr(self._team, 'yield_team'):
            try:
                self._cached_team = self._team.yield_team()
                # print(f"[DEBUG] {self.__class__.__name__} rotated to next team")
            except Exception as e:
                print(f"[ERROR] Failed to rotate team: {e}")

    async def _handle_challenge_request(self, challenge: dict):
        """
        Internal hook to handle incoming challenges.
        We override this to ensure we generated a team BEFORE accept_challenge is called.
        """
        # opponent = challenge.get('challenger')
        # print(f"[DEBUG] {self.__class__.__name__} receiving challenge from {opponent}")
        await super()._handle_challenge_request(challenge)

    def _get_effectiveness(self, move: Move, target: Pokemon) -> float:
        """Calculate type effectiveness of move against target."""
        if not move or not target:
            return 1.0
        
        try:
            type_chart = GenData.from_gen(9).type_chart
            move_type = move.type
            
            # If move represents a type object or string
            if hasattr(move_type, "damage_multiplier"):
                return move_type.damage_multiplier(
                    target.type_1, 
                    target.type_2, 
                    type_chart=type_chart
                )
            
            # Fallback for older interface behavior or strings
            return target.damage_multiplier(move_type, type_chart=type_chart)
        except (KeyError, AttributeError, TypeError):
            # Fallback if type lookup fails
            return 1.0
    
    def _get_defensive_effectiveness(self, atk_type: str, defender: Pokemon) -> float:
        """Calculate how effective an attack type is against defender."""
        if not defender:
            return 1.0
        
        try:
            def_type1 = defender.type_1.name.lower() if defender.type_1 else None
            def_type2 = defender.type_2.name.lower() if defender.type_2 else None
            return get_type_effectiveness(atk_type.lower(), def_type1, def_type2)
        except (KeyError, AttributeError):
            return 1.0
    
    def _type_resists(self, pokemon: Pokemon, atk_type: str) -> float:
        """Returns effectiveness multiplier (≤0.5 = resist, 0 = immune)."""
        return self._get_defensive_effectiveness(atk_type, pokemon)
    
    # =========================================================================
    # Move Selection Utilities
    # =========================================================================
    
    def _get_best_move_by_effectiveness(self, battle: AbstractBattle) -> Optional[Move]:
        """Returns move with highest type effectiveness × power."""
        if not battle.available_moves:
            return None
        
        opp = battle.opponent_active_pokemon
        
        best_move = None
        best_score = -1
        
        for move in battle.available_moves:
            eff = self._get_effectiveness(move, opp)
            power = move.base_power or 0
            score = eff * power
            
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move
    
    def _get_damaging_moves(self, battle: AbstractBattle) -> List[Move]:
        """Returns all moves with base_power > 0."""
        return [m for m in battle.available_moves if m.base_power and m.base_power > 0]
    
    def _get_status_moves(self, battle: AbstractBattle) -> List[Move]:
        """Returns all non-damaging moves."""
        return [m for m in battle.available_moves if not m.base_power or m.base_power == 0]
    
    def _get_hardest_hitting_move(self, battle: AbstractBattle) -> Optional[Move]:
        """Returns move with highest raw power × effectiveness."""
        damaging = self._get_damaging_moves(battle)
        if not damaging:
            return None
        
        opp = battle.opponent_active_pokemon
        return max(damaging, key=lambda m: 
            (m.base_power or 0) * self._get_effectiveness(m, opp))
    
    # =========================================================================
    # Switch Selection Utilities
    # =========================================================================
    
    def _pick_best_switch(self, battle: AbstractBattle) -> Optional[BattleOrder]:
        """Pick switch with best type matchup against opponent."""
        if not battle.available_switches:
            return None
        
        opp = battle.opponent_active_pokemon
        
        best_switch = None
        best_score = -999
        
        for switch in battle.available_switches:
            score = self._calculate_matchup_score(switch, opp)
            if score > best_score:
                best_score = score
                best_switch = switch
        
        if best_switch:
            return self.create_order(best_switch)
        return None
    
    def _get_healthiest_switch(self, battle: AbstractBattle) -> Optional[Pokemon]:
        """Returns switch with highest HP fraction."""
        if not battle.available_switches:
            return None
        return max(battle.available_switches, key=lambda p: p.current_hp_fraction)
    
    def _get_resist_switch(self, battle: AbstractBattle) -> Optional[Pokemon]:
        """Returns a switch that resists opponent's likely attacks."""
        if not battle.available_switches:
            return None
        
        opp = battle.opponent_active_pokemon
        if not opp:
            return None
        
        # Estimate opponent's most threatening type
        opp_types = []
        if opp.type_1:
            opp_types.append(opp.type_1.name.lower())
        if opp.type_2:
            opp_types.append(opp.type_2.name.lower())
        
        # Find a switch that resists these types
        for switch in battle.available_switches:
            resists_all = True
            for t in opp_types:
                if self._type_resists(switch, t) > 0.5:
                    resists_all = False
                    break
            if resists_all and opp_types:
                return switch
        
        return None
    
    # =========================================================================
    # Matchup & Damage Estimation
    # =========================================================================
    
    def _calculate_matchup_score(self, our_mon: Pokemon, their_mon: Pokemon) -> float:
        """
        Returns -1 to +1 score for matchup (positive = we win).
        Based on type effectiveness in both directions.
        """
        if not our_mon or not their_mon:
            return 0.0
        
        # Our offensive effectiveness
        our_stab_types = []
        if our_mon.type_1:
            our_stab_types.append(our_mon.type_1.name.lower())
        if our_mon.type_2:
            our_stab_types.append(our_mon.type_2.name.lower())
        
        our_offense = max(
            [self._get_defensive_effectiveness(t, their_mon) for t in our_stab_types] or [1.0]
        )
        
        # Their offensive effectiveness against us
        their_stab_types = []
        if their_mon.type_1:
            their_stab_types.append(their_mon.type_1.name.lower())
        if their_mon.type_2:
            their_stab_types.append(their_mon.type_2.name.lower())
        
        their_offense = max(
            [self._get_defensive_effectiveness(t, our_mon) for t in their_stab_types] or [1.0]
        )
        
        # Positive = we deal more SE damage than we take
        return (our_offense - their_offense) / 4.0  # Normalize to roughly -1 to 1
    
    def _estimates_ohko(self, move: Move, target: Pokemon, battle: AbstractBattle) -> bool:
        """Estimate if move will KO target using accurate damage calc."""
        if not move or not target:
            return False
            
        try:
            # Use real belief calculation
            # Note: We need to update belief tracker potentially? 
            # Ideally the agent should be updating it, but BaseCurriculumPlayer doesn't enforce standard update loop.
            # We will use what information is available.
            res = self.damage_calculator.calculate_move_damage(battle, move, is_our_move=True)
            return res.min_percent >= 1.0 # Guaranteed OHKO
            
        except Exception:
            # Fallback heuristic if calc fails
            eff = self._get_effectiveness(move, target)
            power = move.base_power or 0
            target_hp = target.current_hp_fraction
            return eff >= 2.0 and power >= 80 and target_hp < 0.6
    
    def _estimate_incoming_effectiveness(self, battle: AbstractBattle) -> float:
        """Estimate effectiveness of opponent's STAB against our active."""
        opp = battle.opponent_active_pokemon
        active = battle.active_pokemon
        
        if not opp or not active:
            return 1.0
        
        opp_types = []
        if opp.type_1:
            opp_types.append(opp.type_1.name.lower())
        if opp.type_2:
            opp_types.append(opp.type_2.name.lower())
        
        if not opp_types:
            return 1.0
        
        return max(self._get_defensive_effectiveness(t, active) for t in opp_types)
    
    # =========================================================================
    # Battle State Tracking
    # =========================================================================
    
    def _update_tracking(self, battle: AbstractBattle):
        """Update turn-over-turn tracking."""
        battle_id = battle.battle_tag
        current_turn = battle.turn
        
        # Reset cache if new battle
        if battle_id != self._current_battle_tag:
            self._current_battle_tag = battle_id
            self._last_turn_data = {}

        prev_opp = self._last_turn_data.get('opp_species')
        
        current_opp = None
        if battle.opponent_active_pokemon:
            current_opp = battle.opponent_active_pokemon.species
        
        # Detect opponent switch
        self._opponent_switched_last_turn = (
            prev_opp is not None and 
            current_opp is not None and 
            prev_opp != current_opp
        )
        
        # Update cache
        self._last_turn_data = {
            'turn': current_turn,
            'opp_species': current_opp,
        }
    
    def _did_opponent_switch(self, battle: AbstractBattle) -> bool:
        """Returns True if opponent switched last turn."""
        self._update_tracking(battle)
        return self._opponent_switched_last_turn
    
    def _click_best_move(self, battle: AbstractBattle) -> BattleOrder:
        """Default action: click best effectiveness move logging silenced."""
        best = self._get_best_move_by_effectiveness(battle)
        
        # Log decision for training verification
        # if battle.turn <= 3:  # Log first 3 turns per battle
        #     agent_name = self.__class__.__name__
        #     opp = battle.opponent_active_pokemon
        #     opp_name = opp.species if opp else "Unknown"
        #     our_mon = battle.active_pokemon
        #     our_name = our_mon.species if our_mon else "Unknown"
            
        #     print(f"\n[{agent_name}] Turn {battle.turn} Decision:")
        #     print(f"  Matchup: {our_name} vs {opp_name}")
        #     if battle.available_moves:
        #         print(f"  Available Moves:")
        #         for m in battle.available_moves[:4]:
        #             eff = self._get_effectiveness(m, opp) if opp else 1.0
        #             score = (m.base_power or 0) * eff
        #             marker = " <-- BEST" if m == best else ""
        #             print(f"    - {m.id:20s} (Power:{m.base_power or 0:3d} x Eff:{eff:.1f} = {score:.0f}){marker}")
        #     print(f"  Selected: {best.id if best else 'NONE - switching'}")
        
        if best:
            return self.create_order(best)
        if battle.available_moves:
            return self.create_order(battle.available_moves[0])
        return self.choose_random_move(battle)
