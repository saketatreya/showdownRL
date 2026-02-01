"""
Configuration models with validation.

Uses Pydantic for type checking and validation of training/reward configs.
"""

from pydantic import BaseModel, Field


class TrainingConfig(BaseModel):
    """Configuration for PPO training hyperparameters (Wang 2024 values)."""
    
    learning_rate: float = Field(default=3e-4, gt=0, lt=1)
    lr_schedule: str = Field(default="wang")
    n_steps: int = Field(default=2048, ge=64)
    batch_size: int = Field(default=1024, ge=16)
    n_epochs: int = Field(default=7, ge=1)
    gamma: float = Field(default=0.9999, ge=0, le=1)
    gae_lambda: float = Field(default=0.754, ge=0, le=1)
    clip_range: float = Field(default=0.0829, ge=0, le=1)
    ent_coef: float = Field(default=0.0588, ge=0)
    vf_coef: float = Field(default=0.4375, ge=0)
    max_grad_norm: float = Field(default=0.543, gt=0)
    
    # Architecture
    observation_dim: int = Field(default=1163, ge=1)
    lstm_hidden_size: int = Field(default=256, ge=32)
    n_lstm_layers: int = Field(default=1, ge=1, le=4)
    mlp_hidden_sizes: list = Field(default=[64, 64])


class SelfPlayConfig(BaseModel):
    """Configuration for self-play training."""
    
    pool_size: int = Field(default=10, ge=1)
    save_every_n_steps: int = Field(default=50000, ge=1000)
    elo_k_factor: float = Field(default=32.0, gt=0)
    initial_elo: float = Field(default=1000.0, ge=0)
    use_elo_matching: bool = Field(default=True)
    elo_range: float = Field(default=200.0, ge=0)


class EnvironmentConfig(BaseModel):
    """Configuration for the battle environment."""
    
    battle_format: str = Field(default="gen9randombattle")
    max_turns: int = Field(default=100, ge=10)
    team_size: int = Field(default=6, ge=1, le=6)
    normalize_observations: bool = Field(default=True)


# Default configurations
DEFAULT_TRAINING_CONFIG = TrainingConfig()
DEFAULT_SELF_PLAY_CONFIG = SelfPlayConfig()
