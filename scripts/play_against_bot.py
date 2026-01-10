"""
Play against trained RL bot with detailed observation visualization.
This script shows exactly what the bot "sees" in its observation vector
and explains its decision-making process.
"""

import argparse
import asyncio
import sys
import numpy as np
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from poke_env import AccountConfiguration
from poke_env.battle import AbstractBattle
from poke_env.player.battle_order import BattleOrder

from src.trained_player import TrainedPlayer
from src.utils import load_pokemon_data, get_type_effectiveness, calculate_speed
from sb3_contrib import RecurrentPPO


class DebugTrainedPlayer(TrainedPlayer):
    """
    A version of TrainedPlayer that explains its observations and decisions.
    Shows everything the bot "sees" in its observation vector.
    """
    
    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        """Choose move with detailed visualization of the decision process."""
        print("\n" + "=" * 80)
        print(f" TURN {battle.turn} ".center(80, "="))
        print("=" * 80)
        
        # 1. Show Detailed Battle State (what the bot sees in its observation)
        self._show_observation_breakdown(battle)
        
        # Update beliefs (same as parent)
        self._update_beliefs(battle)
        
        # Get observation
        obs = self.obs_builder.embed_battle(battle)
        
        # 2. Show action probabilities
        self._show_action_probabilities(battle, obs)
        
        # 3. Make decision
        action, self._lstm_states = self.model.predict(
            obs,
            state=self._lstm_states,
            episode_start=np.array([self._episode_start]),
            deterministic=self.deterministic
        )
        self._episode_start = False
        
        # 4. Show selected action
        print("-" * 40)
        chosen_readable = self._get_readable_action(int(action), battle)
        print(f"✅ DECISION: {chosen_readable}")
        print("=" * 80 + "\n")
        
        return self._action_to_order(battle, action)

    def _show_observation_breakdown(self, battle: AbstractBattle):
        """Display the observation vector components in a readable format."""
        
        bot = battle.active_pokemon
        opp = battle.opponent_active_pokemon
        
        # ============ ACTIVE POKEMON (56 dims each) ============
        print("\n📊 OBSERVATION BREAKDOWN")
        print("-" * 40)
        
        # Bot's Active Pokemon
        print(f"\n🤖 BOT ACTIVE: {bot.species.upper()} (Lvl {bot.level})")
        print(f"   HP:     {bot.current_hp}/{bot.max_hp} ({bot.current_hp_fraction:.1%})")
        print(f"   Types:  {bot.type_1.name if bot.type_1 else '?'}" + 
              (f"/{bot.type_2.name}" if bot.type_2 else ""))
        print(f"   Status: {bot.status.name if bot.status else 'None'}")
        if any(v != 0 for v in bot.boosts.values()):
            print(f"   Boosts: {dict((k, v) for k, v in bot.boosts.items() if v != 0)}")
        print(f"   Tera:   {'Yes' if bot.is_terastallized else 'No'} (Type: {bot.tera_type.name if hasattr(bot, 'tera_type') and bot.tera_type else '?'})")
        print(f"   Item:   {bot.item if bot.item else 'Unknown'}")
        print(f"   Ability:{bot.ability if bot.ability else 'Unknown'}")
        
        # Bot's Team (39 dims × 6)
        bench = [p for p in battle.team.values() if not p.active and not p.fainted]
        if bench:
            print(f"\n   BENCH ({len(bench)}):")
            for p in bench:
                status_str = f" [{p.status.name}]" if p.status else ""
                print(f"      - {p.species}: {p.current_hp_fraction:.0%} HP{status_str}")
        
        print()
        
        # Opponent's Active Pokemon
        if opp:
            print(f"👤 OPP ACTIVE: {opp.species.upper()} (Lvl {opp.level})")
            print(f"   HP:     {opp.current_hp_fraction:.1%}")
            print(f"   Types:  {opp.type_1.name if opp.type_1 else '?'}" + 
                  (f"/{opp.type_2.name}" if opp.type_2 else ""))
            print(f"   Status: {opp.status.name if opp.status else 'None'}")
            if any(v != 0 for v in opp.boosts.values()):
                print(f"   Boosts: {dict((k, v) for k, v in opp.boosts.items() if v != 0)}")
            
            # Known moves (21 dims × 4)
            known_moves = list(opp.moves.keys())
            if known_moves:
                print(f"   Known Moves: {', '.join(known_moves)}")
            else:
                print(f"   Known Moves: None revealed yet")
        else:
            print("👤 OPP ACTIVE: None (fainted)")
        
        # Opponent's Team
        opp_team = [p for p in battle.opponent_team.values() if not p.active]
        if opp_team:
            print(f"\n   OPP BENCH ({len(opp_team)} seen):")
            for p in opp_team:
                hp_str = f"{p.current_hp_fraction:.0%} HP" if p.current_hp_fraction else "?"
                fainted_str = " [FAINTED]" if p.fainted else ""
                print(f"      - {p.species}: {hp_str}{fainted_str}")
        
        # ============ BAYESIAN BELIEFS (60 dims for 6 opp pokemon) ============
        if opp:
            print("\n🔮 BAYESIAN BELIEF STATE:")
            print("-" * 40)
            
            # Get belief for opponent's active pokemon
            belief = self.belief_tracker.get_or_create_belief(opp.species)
            
            # Role Probabilities
            if belief.role_probs:
                sorted_roles = sorted(belief.role_probs.items(), key=lambda x: x[1], reverse=True)
                print(f"   Role Probabilities (based on revealed info):")
                for role, prob in sorted_roles[:4]:
                    bar = "█" * int(prob * 20)
                    print(f"      {role:<20}: {prob:.1%} {bar}")
            else:
                print(f"   Role Probabilities: Unknown species - no data")
            
            # Predicted Moves
            predicted_moves = belief.get_unrevealed_move_probs()
            if predicted_moves:
                sorted_moves = sorted(predicted_moves.items(), key=lambda x: x[1], reverse=True)[:6]
                print(f"\n   Predicted Unrevealed Moves:")
                for move_name, prob in sorted_moves:
                    print(f"      {move_name:<16}: {prob:.1%}")
            
            # Predicted Items
            if not belief.observed_item:
                predicted_items = belief.get_item_probs()
                if predicted_items:
                    sorted_items = sorted(predicted_items.items(), key=lambda x: x[1], reverse=True)[:3]
                    print(f"\n   Predicted Items:")
                    for item_name, prob in sorted_items:
                        print(f"      {item_name:<16}: {prob:.1%}")
            else:
                print(f"\n   Item: {belief.observed_item} (revealed)")
        
        # ============ DAMAGE MATRIX (24 dims) ============
        if opp and battle.available_moves:
            print("\n⚔️ DAMAGE MATRIX (Estimated % Damage):")
            print("-" * 40)
            
            from src.damage_calc import BeliefDamageCalculator
            try:
                dmg_calc = BeliefDamageCalculator(self.pokemon_data, self.belief_tracker)
                
                for move in battle.available_moves:
                    if not move.base_power:
                        # Status move
                        print(f"   {move.id:<15}: Status Move")
                        continue
                    
                    try:
                        result = dmg_calc.calculate_move_damage(battle, move, is_our_move=True)
                        
                        # Format output
                        ohko_str = " [OHKO!]" if result.is_ohko else ""
                        twohko_str = " [2HKO]" if result.is_2hko and not result.is_ohko else ""
                        
                        min_pct = result.min_percent * 100
                        max_pct = result.max_percent * 100
                        
                        print(f"   {move.id:<15}: {min_pct:>5.1f}% - {max_pct:>5.1f}%{ohko_str}{twohko_str}")
                    except Exception as e:
                        # Fallback to simple calc
                        t1 = opp.type_1.name.lower() if opp.type_1 else None
                        t2 = opp.type_2.name.lower() if opp.type_2 else None
                        eff = get_type_effectiveness(move.type.name.lower(), t1, t2)
                        rough = (move.base_power / 100.0) * eff * 25  # Very rough estimate
                        print(f"   {move.id:<15}: ~{rough:.0f}% (approx)")
            except Exception as e:
                print(f"   (Damage calc error: {e})")
        
        # ============ MOVES (26 dims × 4) ============
        print("\n💥 MOVE ANALYSIS (Bot's Available Moves):")
        print("-" * 40)
        
        for i, move in enumerate(battle.available_moves):
            # Calculate effectiveness
            if opp:
                t1 = opp.type_1.name.lower() if opp.type_1 else None
                t2 = opp.type_2.name.lower() if opp.type_2 else None
                eff = get_type_effectiveness(move.type.name.lower(), t1, t2)
            else:
                eff = 1.0
            
            # STAB check
            is_stab = move.type in (bot.type_1, bot.type_2)
            stab_str = " [STAB]" if is_stab else ""
            
            # Effectiveness icon
            total_mult = eff * (1.5 if is_stab else 1.0)
            if eff == 0:
                icon = "⬛ IMMUNE"
            elif total_mult >= 3.0:
                icon = "🟩🟩 SUPER EFF"
            elif total_mult >= 1.5:
                icon = "🟩 SUPER EFF"
            elif eff < 1.0:
                icon = "🟥 NOT EFF"
            else:
                icon = "⬜ Neutral"
            
            # Category
            cat = move.category.name if move.category else "?"
            
            # Priority
            priority_str = f" [+{move.priority}]" if move.priority > 0 else ""
            
            print(f"   [{i}] {move.id:<15} | Pow:{move.base_power:>3} | {cat:<8} | {icon}{stab_str}{priority_str}")
        
        # ============ DAMAGE ESTIMATES (24 dims) ============
        if opp:
            print("\n📈 DAMAGE/SPEED ANALYSIS:")
            print("-" * 40)
            
            # Speed comparison
            our_base_speed = bot.base_stats.get('spe', 100) if bot.base_stats else 100
            opp_base_speed = opp.base_stats.get('spe', 100) if opp.base_stats else 100
            
            our_paralyzed = bot.status and bot.status.name.lower() == 'par'
            opp_paralyzed = opp.status and opp.status.name.lower() == 'par'
            
            our_boost = bot.boosts.get('spe', 0) if bot.boosts else 0
            opp_boost = opp.boosts.get('spe', 0) if opp.boosts else 0
            
            our_speed = calculate_speed(our_base_speed, bot.level, our_boost, our_paralyzed, False)
            opp_speed = calculate_speed(opp_base_speed, opp.level, opp_boost, opp_paralyzed, False)
            
            speed_icon = "⚡ FASTER" if our_speed > opp_speed else "🐢 SLOWER"
            print(f"   Speed: {speed_icon} (Us: {our_speed:.0f} vs Opp: {opp_speed:.0f})")
            
            # Priority moves
            priority_moves = [m.id for m in battle.available_moves if m.priority > 0]
            if priority_moves:
                print(f"   Priority Moves: {', '.join(priority_moves)}")
        
        # ============ FIELD STATE (30 dims) ============
        print("\n🌍 FIELD STATE:")
        print("-" * 40)
        
        # Weather
        if battle.weather:
            weather_list = [w.name for w in battle.weather]
            print(f"   Weather: {', '.join(weather_list)}")
        else:
            print(f"   Weather: None")
        
        # Terrain/Fields
        if battle.fields:
            field_list = [f.name for f in battle.fields]
            print(f"   Fields:  {', '.join(field_list)}")
        else:
            print(f"   Fields:  None")
        
        # Our side conditions
        if battle.side_conditions:
            sc_list = [str(sc) for sc in battle.side_conditions]
            print(f"   Our Side Conditions: {sc_list}")
        
        # Opponent side conditions
        if battle.opponent_side_conditions:
            osc_list = [str(sc) for sc in battle.opponent_side_conditions]
            print(f"   Opp Side Conditions: {osc_list}")
        
        # ============ SWITCH MATCHUPS (3 dims × 5) ============
        if battle.available_switches:
            print("\n🔄 SWITCH OPTIONS:")
            print("-" * 40)
            
            for i, pokemon in enumerate(battle.available_switches[:5]):
                # Calculate simple matchup score
                if opp:
                    our_t1 = pokemon.type_1.name.lower() if pokemon.type_1 else None
                    our_t2 = pokemon.type_2.name.lower() if pokemon.type_2 else None
                    opp_t1 = opp.type_1.name.lower() if opp.type_1 else None
                    opp_t2 = opp.type_2.name.lower() if opp.type_2 else None
                    
                    # Our best offense
                    offense = max(
                        get_type_effectiveness(our_t1, opp_t1, opp_t2) if our_t1 else 1.0,
                        get_type_effectiveness(our_t2, opp_t1, opp_t2) if our_t2 else 1.0
                    )
                    
                    # Their best offense vs us
                    defense = max(
                        get_type_effectiveness(opp_t1, our_t1, our_t2) if opp_t1 else 1.0,
                        get_type_effectiveness(opp_t2, our_t1, our_t2) if opp_t2 else 1.0
                    )
                    
                    matchup = "🟢 GOOD" if offense > defense else ("🔴 BAD" if defense > offense else "⚪ EVEN")
                else:
                    matchup = "?"
                
                hp_str = f"{pokemon.current_hp_fraction:.0%}"
                print(f"   [{4+i}] {pokemon.species:<15} | HP: {hp_str} | {matchup}")
        
        # ============ ACTION MASK (26 dims) ============
        print("\n🎯 VALID ACTIONS (Action Mask):")
        print("-" * 40)
        valid_moves = len(battle.available_moves)
        valid_switches = len(battle.available_switches)
        can_tera = battle.can_tera
        force_switch = battle.force_switch
        
        print(f"   Moves: {valid_moves} available, Switches: {valid_switches} available")
        print(f"   Can Tera: {can_tera}, Force Switch: {force_switch}")
        
        # ============ META CONTEXT (8 dims) ============
        print("\n📋 META CONTEXT:")
        print("-" * 40)
        
        alive_ours = sum(1 for p in battle.team.values() if not p.fainted)
        alive_opp = sum(1 for p in battle.opponent_team.values() if not p.fainted)
        hp_ours = sum(p.current_hp_fraction for p in battle.team.values() if p.current_hp_fraction)
        hp_opp = sum(p.current_hp_fraction for p in battle.opponent_team.values() if p.current_hp_fraction)
        
        print(f"   Turn: {battle.turn}")
        print(f"   Alive: Us {alive_ours}/6, Opp {alive_opp}/6 (Advantage: {alive_ours - alive_opp:+d})")
        print(f"   Total HP: Us {hp_ours:.1f}/6, Opp {hp_opp:.1f}/6 (Advantage: {hp_ours - hp_opp:+.1f})")
        print(f"   Can Tera: {'Yes' if battle.can_tera else 'No'}")

    def _show_action_probabilities(self, battle: AbstractBattle, obs: np.ndarray):
        """Display the model's action probabilities."""
        print("\n🤔 BOT THINKING (Action Probabilities):")
        print("-" * 40)
        
        obs_tensor = torch.as_tensor(obs).float().unsqueeze(0)
        
        # Prepare LSTM states
        if self._lstm_states is None:
            n_layers = self.model.policy.lstm_actor.num_layers
            hidden_size = self.model.policy.lstm_actor.hidden_size
            h = torch.zeros((n_layers, 1, hidden_size))
            c = torch.zeros((n_layers, 1, hidden_size))
            current_states = (h, c)
        else:
            h, c = self._lstm_states
            current_states = (torch.as_tensor(h).float(), torch.as_tensor(c).float())
        
        episode_starts = torch.tensor([float(self._episode_start)])
        
        try:
            with torch.no_grad():
                dist_result = self.model.policy.get_distribution(
                    obs_tensor, current_states, episode_starts
                )
                
                if isinstance(dist_result, tuple):
                    dist, _ = dist_result
                else:
                    dist = dist_result
                
                probs = dist.distribution.probs[0].numpy()
                
                # Show top 5 actions
                top_indices = np.argsort(probs)[::-1][:5]
                for idx in top_indices:
                    readable = self._get_readable_action(idx, battle)
                    print(f"   [{idx:>2}] {readable:<35} : {probs[idx]:.1%}")
                    
        except Exception as e:
            print(f"   (Could not compute probabilities: {e})")

    def _get_readable_action(self, action: int, battle: AbstractBattle) -> str:
        """Convert action index to human-readable string, matching actual execution logic."""
        n_moves = len(battle.available_moves)
        n_switches = len(battle.available_switches)
        can_tera = battle.can_tera
        
        if action < 4:
            # Move actions 0-3
            if action < n_moves:
                move = battle.available_moves[action]
                return f"Move: {move.id}"
            elif n_moves > 0:
                # Fallback to first move
                return f"Move: {battle.available_moves[0].id} (fallback from {action})"
            elif n_switches > 0:
                return f"Switch: {battle.available_switches[0].species} (no moves, fallback)"
            else:
                return "Default (no options)"
        
        elif action < 9:
            # Switch actions 4-8
            switch_idx = action - 4
            if switch_idx < n_switches:
                return f"Switch: {battle.available_switches[switch_idx].species}"
            elif n_switches > 0:
                return f"Switch: {battle.available_switches[0].species} (fallback from {action})"
            elif n_moves > 0:
                return f"Move: {battle.available_moves[0].id} (no switches, fallback)"
            else:
                return "Default (no options)"
        
        elif action < 13:
            # Tera moves 9-12
            move_idx = action - 9
            if move_idx < n_moves and can_tera:
                return f"Tera + {battle.available_moves[move_idx].id}"
            elif move_idx < n_moves:
                return f"Move: {battle.available_moves[move_idx].id} (can't tera)"
            elif n_moves > 0:
                return f"Move: {battle.available_moves[0].id} (fallback from tera {action})"
            else:
                return f"Default (fallback from {action})"
        
        else:
            # Reserved actions 13-25
            if n_moves > 0:
                return f"Move: {battle.available_moves[0].id} (reserved action {action})"
            elif n_switches > 0:
                return f"Switch: {battle.available_switches[0].species} (reserved action {action})"
            else:
                return f"Default (reserved action {action})"


async def main():
    parser = argparse.ArgumentParser(description="Play against trained RL bot with debug output")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.zip)")
    parser.add_argument("--bot-name", type=str, default="TrainedBot",
                        help="Username for the bot")
    
    args = parser.parse_args()
    
    print(f"Loading model from {args.checkpoint}...")
    
    model = RecurrentPPO.load(args.checkpoint)
    pokemon_data = load_pokemon_data("gen9randombattle.json")
    
    player = DebugTrainedPlayer(
        model=model,
        pokemon_data=pokemon_data,
        battle_format="gen9randombattle",
        account_configuration=AccountConfiguration(args.bot_name, None),
        deterministic=True
    )
    
    print("\n" + "=" * 50)
    print(f"🤖 Bot '{args.bot_name}' is ready!")
    print("=" * 50)
    print("Instructions:")
    print("1. Open http://localhost:8000 in your browser.")
    print("2. Log in with any username (e.g., 'Guest123').")
    print(f"3. Find user '{args.bot_name}' in the user list.")
    print(f"4. Challenge '{args.bot_name}' to a 'Gen 9 Random Battle'.")
    print("\nWaiting for challenges... (Press Ctrl+C to stop)")
    
    while True:
        await player.accept_challenges(None, 1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBot stopped.")
