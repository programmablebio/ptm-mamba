import pandas as pd
from typing import Dict, Literal, Optional
from datasets import Dataset, load_dataset
from datasets.dataset_dict import DatasetDict
from pydantic import BaseModel
from protein_lm.modeling.getters.ptm_dataset import get_ptm_dataset
from protein_lm.modeling.getters.uniref_dataset import get_uniref_dataset


def get_dataset(config_dict: Dict, tokenizer) -> Dataset:
    if config_dict["dataset"] == "ptm":
        return get_ptm_dataset(config_dict, tokenizer)
    elif config_dict["dataset"] == "uniref50":
        return get_uniref_dataset(config_dict, tokenizer)
    else:
        raise ValueError(f"Invalid dataset {config_dict['dataset']}!")
