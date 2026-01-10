"""
SetupSweeper: Boosts once, then sweeps.
Teaches the agent to pressure boosters and use phazing.
"""

from poke_env.battle import AbstractBattle

from .base import BaseCurriculumPlayer
from ..utils import MoveClassifier


class SetupSweeper(BaseCurriculumPlayer):
    """
    Boosts once if safe, then attacks.
    
    BEHAVIOR:
    - If at +0 and opponent HP > 50%: Use one boost move
    - Otherwise: Click strongest move
    - Cap at +2 (don't over-boost)
    LESSON: Agent learns to pressure boosters.
    EXPLOITABLE BY: Phazing, faster revenge killers, Haze.
    """
    
    BOOST_MOVES = MoveClassifier.BOOST_MOVES
    MAX_BOOST = 2
    
    def choose_move(self, battle: AbstractBattle):
        # Log moveset on turn 1
        if not self._logged_moveset and battle.turn == 1:
            self._logged_moveset = True
            # print(f"\n[SetupSweeper] Turn 1 Moveset:")
            # print(f"  Active: {battle.active_pokemon.species if battle.active_pokemon else 'None'}")
            # for move in battle.available_moves:
            #     boost_mark = " ⭐BOOST" if move.id.lower() in self.BOOST_MOVES else ""
            #     print(f"    - {move.id:20s} (Power: {move.base_power or 0:3d}){boost_mark}")
            # print()
        
        active = battle.active_pokemon
        opp = battle.opponent_active_pokemon
        
        # Log current boost status
        # boosts = active.boosts if active else {}
        # atk = boosts.get('atk', 0)
        # spa = boosts.get('spa', 0)
        # spe = boosts.get('spe', 0)
        # opp_hp = opp.current_hp_fraction if opp else 1.0
        
        should_boost = self._should_boost(active, opp, battle)
        boost_move = self._get_boost_move(battle)
        
        # print(f"[SetupSweeper] T{battle.turn}: Boosts(atk={atk},spa={spa},spe={spe}), OppHP={opp_hp:.0%}, ShouldBoost={should_boost}")
        
        # Check if we should boost
        if should_boost and boost_move:
            # print(f"  -> BOOSTING: {boost_move.id}")
            return self.create_order(boost_move)
        
        # Attack
        # print(f"  -> Clicking best move")
        return self._click_best_move(battle)
    
    def _should_boost(self, active, opp, battle) -> bool:
        """Should we use a setup move?"""
        if not active or not opp:
            return False
        
        # Don't boost if opponent is low (just finish them)
        if opp.current_hp_fraction < 0.50:
            return False
        
        # Don't over-boost
        boosts = active.boosts
        atk = boosts.get('atk', 0)
        spa = boosts.get('spa', 0)
        spe = boosts.get('spe', 0)
        
        if max(atk, spa, spe) >= self.MAX_BOOST:
            return False
        
        return True
    
    def _get_boost_move(self, battle: AbstractBattle):
        """Get a stat-boosting move."""
        for move in battle.available_moves:
            if move.id.lower() in self.BOOST_MOVES:
                return move
        return None
