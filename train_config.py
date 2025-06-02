from typing import Dict, Union
from dataclasses import dataclass, field

@dataclass
class TrainConfig:
    env_name: str = 'Walker2d-v5' # 논문에서는 v1이 사용되었으나 deprecated되어 v4 사용.

    alpha: float = 0.3
    reward_scale: float = 1.
    eval_steps: int = 50_000
    eval_num_episodes: int = 5
    tau: float = 0.001
    learning_rate: float = 3e-4
    discount_rate: float = 0.99
    replay_buffer_size: int = 1e6
    min_replay_buffer_size: int = 1e5
    num_layers: int = 2
    hidden_dim: int = 256
    batch_size: int = 256
    num_train_steps: int = 1_000_000
