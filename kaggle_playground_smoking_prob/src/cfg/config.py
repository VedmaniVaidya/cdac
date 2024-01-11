from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class Config:
    do_train: bool = field(
        default=True, metadata={"help": "Whether to run training."}
    )
    do_eval: bool = field(
        default=True, metadata={"help": "Whether to run eval on the dev set."}
    )
    do_predict: bool = field(
        default=True, metadata={"help": "Whether to run predictions on the test set."}
    )
    folds: int = field(
        default=5, metadata={"help": "Number of folds for cross validation."}
    )
    seed: int = field(
        default=42, metadata={"help": "Random seed for initialization"}
    )
    batch_size: int = field(
        default=128, metadata={"help": "Batch size for training."}
    )
    eval_batch_size: int = field(
        default=128, metadata={"help": "Batch size for training."}
    )
    model_uid: str = field(
        default=None, metadata={"help": "Unique identifier for the model."}
    )
    
