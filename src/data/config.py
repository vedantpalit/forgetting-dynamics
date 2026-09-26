"""Data configuration."""
from dataclasses import dataclass


@dataclass
class DataConfig:
    task_type: str = "biography"
    batch_size: int = 128
    eval_batch_size: int = 1024
    # Derived from the preprocessed biography data at runtime
    sequence_length: int = 64
    vocab_size: int = 64
    # Biography-specific
    biography_data_path: str = "data/biography/preprocessed.npz"
    support_size: int = 1
    num_train_templates: int = 20
