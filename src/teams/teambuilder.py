"""Curriculum teambuilder with pre-defined teams."""
import random
from poke_env.teambuilder import Teambuilder


class CurriculumTeambuilder(Teambuilder):
    """Randomly selects from a pool of curated teams."""
    
    def __init__(self, teams: list):
        """
        Args:
            teams: List of showdown-formatted team strings
        """
        self.packed_teams = []
        
        for team in teams:
            # Parse and pack each team
            try:
                parsed = self.parse_showdown_team(team)
                packed = self.join_team(parsed)
                self.packed_teams.append(packed)
            except Exception as e:
                print(f"Error parsing team: {e}")
                # Continue to next team if one fails
                continue
        
        if not self.packed_teams:
            raise ValueError("Must provide at least one valid team")
        
        self.n_teams = len(self.packed_teams)
    
    def yield_team(self) -> str:
        """Return a random team from the pool."""
        idx = random.randint(0, self.n_teams - 1)
        return self.packed_teams[idx]


class AgentSpecificTeambuilder(Teambuilder):
    """
    Adapter that binds a CurriculumTeambuilder to a specific agent type.
    Allows poke-env Players to request a new team for their specific type each battle.
    """
    def __init__(self, main_builder: 'CurriculumTeambuilder', agent_type: str):
        self.main_builder = main_builder
        self.agent_type = agent_type
        
    def yield_team(self) -> str:
        """
        Delegate to main builder to get a team for THIS agent type.
        """
        import sys
        
        team = self.main_builder.yield_team()
        
        # Concise logging for team rotation
        preview = team.replace('\n', '|')[:40]
        # Only log if desired, or keep it minimal
        # print(f"[Teambuilder] {self.agent_type}: {preview}...")
        
        # We can also rely on the [CurriculumWrapper] logs for opponent type
        # But verifying team change is good. Let's keep a very short one.
        # print(f"[Team] New for {self.agent_type}") # Even shorter
        sys.stdout.flush()
        
        return team
