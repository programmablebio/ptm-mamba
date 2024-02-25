import os
from typing import Dict

from transformers import TrainingArguments


def get_training_args(config_dict: Dict) -> TrainingArguments:
    config = TrainingArguments(**config_dict)

    if not os.path.isdir(config.output_dir):
        print(f"creating checkpoint directory at {config.output_dir}")
        os.makedirs(config.output_dir)

    return config
