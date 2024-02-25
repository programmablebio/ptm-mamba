import os
import pandas as pd

from Bio import SeqIO
from typing import Dict, Literal, Optional
from datasets import Dataset, load_dataset
from datasets.dataset_dict import DatasetDict
from typing import Dict, Literal, Optional
from protein_lm.modeling.getters.ptm_dataset import DatasetConfig, train_val_test_split


def read_fasta_file(fasta_file_path, subsample_size):
    ids = []
    seqs = []
    with open(fasta_file_path, "r") as fasta_file:
        for i, record in enumerate(SeqIO.parse(fasta_file, "fasta")):
            if subsample_size and i >= subsample_size:
                break
            ids.append(record.id)
            seqs.append(str(record.seq))

    return {"id": ids, "seq": seqs}


def load_uniref_dataset(seq_dict, config) -> DatasetDict:
    ds = Dataset.from_dict(seq_dict)
    ds_dict = DatasetDict({"train": ds})
    return train_val_test_split(ds_dict, config)


def seq2token(batch, tokenizer, sequence_column_name, max_sequence_length):
    batch["input_ids"] = tokenizer(
        batch[sequence_column_name],
        add_special_tokens=True,
        max_sequence_length=max_sequence_length,
    )
    return batch


def get_uniref_dataset(config: Dict, tokenizer) -> Dataset:
    # config = DatasetConfig(**config_dict)
    if config.cache_dir is not None and os.path.exists(config.cache_dir):
        split_dict = DatasetDict.load_from_disk(config.cache_dir)
        return split_dict
    seq_dict = read_fasta_file(config.dataset_loc, config.subsample_size)
    split_dict = load_uniref_dataset(seq_dict, config)
    split_dict = split_dict.map(
        lambda e: seq2token(
            batch=e,
            tokenizer=tokenizer,
            sequence_column_name="seq",
            max_sequence_length=config.max_sequence_length,
        ),
        batched=True,
    )
    if config.cache_dir is not None:
        os.makedirs(config.cache_dir, exist_ok=True)
        split_dict.save_to_disk(config.cache_dir)
    return split_dict
