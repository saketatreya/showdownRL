"""
Configuration models with validation.

Uses Pydantic for type checking and validation of training/reward configs.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional


class RewardConfig(BaseModel):
    """Configuration for reward function weights."""
    
    w_hp: float = Field(default=1.5, ge=0, description="Weight for HP balance")
    w_fainted: float = Field(default=4.0, ge=0, description="Weight for fainted diff")
    w_matchup: float = Field(default=0.2, ge=0, description="Weight for type matchup")
    w_boosts: float = Field(default=0.3, ge=0, description="Weight for stat boosts")
    w_hazards: float = Field(default=0.4, ge=0, description="Weight for entry hazards")
    w_status: float = Field(default=0.2, ge=0, description="Weight for status conditions")
    
    # V2 New Reward Weights
    w_speed: float = Field(default=0.15, ge=0, description="Weight for having speed advantage")
    # w_accuracy REMOVED (double counting)
    w_hazards_up: float = Field(default=0.25, ge=0, description="Weight for having hazards on opp side")
    w_hazards_our: float = Field(default=0.2, ge=0, description="Weight for clearing hazards on our side")
    continuous_boost_bonus: float = Field(default=0.05, ge=0, description="Per-turn bonus per boost stage")
    
    victory_bonus: float = Field(default=15.0, ge=0, description="Bonus for winning")
    defeat_penalty: float = Field(default=-12.0, le=0, description="Penalty for losing")
    
    # V3 Extended: Caps and penalties
    boost_cap: int = Field(default=3, ge=1, le=6, description="Max total boosts for reward calc")
    boost_multiplier: float = Field(default=0.02, ge=0, description="Reward per boost stage")
    switch_tax: float = Field(default=0.10, ge=0, description="Penalty per switch")
    move_fail_penalty: float = Field(default=0.15, ge=0, description="Penalty for failed moves")
    
    # V4: Momentum penalty to prevent boost farming
    momentum_penalty: float = Field(default=0.1, ge=0, description="Penalty per turn without damage after grace period")
    momentum_grace_turns: int = Field(default=2, ge=0, description="Turns without damage before penalty applies")
    
    # V5: attack_bonus REMOVED (Perverse Incentive)
    
    @field_validator('w_fainted')
    @classmethod
    def fainted_weight_reasonable(cls, v: float) -> float:
        if v > 20:
            raise ValueError('w_fainted seems too high (>20)')
        return v


class TrainingConfig(BaseModel):
    """Configuration for PPO training hyperparameters."""
    
    # Wang-Jett-Meng Tuned Values (Optuna-optimized)
    learning_rate: float = Field(default=3e-4, gt=0, lt=1, description="Initial LR (increased for faster convergence)")
    lr_schedule: str = Field(default="linear", description="Anneal to 0 over training")
    n_steps: int = Field(default=2048, ge=64)
    batch_size: int = Field(default=1024, ge=16, description="Wang's optimized batch size")
    n_epochs: int = Field(default=7, ge=1, description="Wang's optimized epochs")
    gamma: float = Field(default=0.9999, ge=0, le=1, description="Near-perfect foresight")
    gae_lambda: float = Field(default=0.754, ge=0, le=1, description="Wang's optimal")
    clip_range: float = Field(default=0.0829, ge=0, le=1, description="Wang's optimal")
    clip_range_vf: float = Field(default=0.0184, ge=0, le=1, description="Value function clipping")
    ent_coef: float = Field(default=0.0588, ge=0, description="Entropy coefficient (Wang's)")
    vf_coef: float = Field(default=0.4375, ge=0, description="Value function coefficient (Wang's)")
    max_grad_norm: float = Field(default=0.543, gt=0, description="Gradient clipping")
    
    # Architecture (LSTM-based)
    observation_dim: int = Field(default=1163, ge=1, description="Observation space dimension")
    lstm_hidden_size: int = Field(default=256, ge=32, description="LSTM hidden state size")
    n_lstm_layers: int = Field(default=1, ge=1, le=4)
    mlp_hidden_sizes: list = Field(default=[64, 64], description="MLP layers after LSTM")


class SelfPlayConfig(BaseModel):
    """Configuration for self-play training."""
    
    pool_size: int = Field(default=10, ge=1, description="Number of opponents in pool")
    save_every_n_steps: int = Field(default=50000, ge=1000)
    elo_k_factor: float = Field(default=32.0, gt=0)
    initial_elo: float = Field(default=1000.0, ge=0)
    
    # Opponent selection
    use_elo_matching: bool = Field(default=True)
    elo_range: float = Field(default=200.0, ge=0, description="Max ELO diff for matching")


class EnvironmentConfig(BaseModel):
    """Configuration for the battle environment."""
    
    battle_format: str = Field(default="gen9randombattle")
    max_turns: int = Field(default=100, ge=10)
    team_size: int = Field(default=6, ge=1, le=6)
    
    # Observation space options
    observation_version: int = Field(default=2, ge=1, le=3)
    normalize_observations: bool = Field(default=True)
    
    # V3: Observation thresholds (previously magic numbers)
    late_game_turn: int = Field(default=50, ge=10, description="Turn at which late_game=1.0")
    winning_threshold: int = Field(default=0, ge=0, description="Alive diff to be 'winning'")


# Default configurations
DEFAULT_REWARD_CONFIG = RewardConfig()
DEFAULT_TRAINING_CONFIG = TrainingConfig()
DEFAULT_SELF_PLAY_CONFIG = SelfPlayConfig()

# Curriculum Reward Phases (Refined via Deep Dive Analysis)
REWARD_CURRICULUM = {
    'early': {  # 0-20%: "Learn to fight"
        'w_fainted': 3.0,
        'w_hp': 2.0,            # High HP weight to teach survival
        'w_matchup': 0.5,       # INCREASED from 0.4: Teach type advantage usage
        'w_boosts': 0.1,
        'w_hazards': 0.05,
        'w_hazards_up': 0.1,
        'w_hazards_our': 0.2,   # Reward clearing mechanics
        'w_status': 0.2,
        'w_speed': 0.15,        # Granular speed now (higher weight warranted)
        'victory_bonus': 15.0,
        'defeat_penalty': -12.0,
        'step_cost': 0.005,
        'switch_tax': 0.10,     # REDUCED from 0.3: Don't discourage switching in bad matchups
        'continuous_boost_bonus': 0.0, 
        'boost_cap': 3,
        'boost_multiplier': 0.02,
        'move_fail_penalty': 0.5,
        'momentum_penalty': 0.1,
        'momentum_grace_turns': 3,
    },
    'mid': {  # 20-50%: "Learn strategy"
        'w_fainted': 2.5,
        'w_hp': 1.5,
        'w_matchup': 0.6,       # Matchup becomes primary driver
        'w_boosts': 0.3,
        'w_hazards': 0.1,
        'w_hazards_up': 0.15,
        'w_hazards_our': 0.25,
        'w_status': 0.3,
        'w_speed': 0.2,
        'victory_bonus': 18.0,
        'defeat_penalty': -14.0,
        'step_cost': 0.01,
        'switch_tax': 0.10,     # Keep low
        'continuous_boost_bonus': 0.02, 
        'boost_cap': 4,
        'boost_multiplier': 0.02,
        'move_fail_penalty': 0.6,
        'momentum_penalty': 0.15,
        'momentum_grace_turns': 3,
    },
    'late': {  # 50%+: "Win games"
        'w_fainted': 2.0,
        'w_hp': 1.0, 
        'w_matchup': 0.5,
        'w_boosts': 0.15,
        'w_hazards': 0.05,
        'w_hazards_up': 0.15,
        'w_hazards_our': 0.3,
        'w_status': 0.2,
        'w_speed': 0.1,
        'victory_bonus': 25.0,
        'defeat_penalty': -20.0,
        'step_cost': 0.02,
        'switch_tax': 0.15,    # Slight tax to prevent mindless pivoting
        'continuous_boost_bonus': 0.0, 
        'boost_cap': 3,
        'boost_multiplier': 0.02,
        'move_fail_penalty': 0.7,
        'momentum_penalty': 0.2,
        'momentum_grace_turns': 3,
    }
}
