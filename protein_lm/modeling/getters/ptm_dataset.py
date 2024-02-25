import pandas as pd
from typing import Dict, Literal, Optional
from datasets import Dataset, load_dataset
from datasets.dataset_dict import DatasetDict
from pydantic import BaseModel




class DatasetConfig(BaseModel):
    dataset: Literal['ptm', 'uniref50']
    dataset_type: Literal["csv", "huggingface", "fasta"]

    # The path if local or the huggingface dataset name if huggingface
    dataset_loc: str

    # sample size to limit to, if any, usually for debugging
    subsample_size: Optional[int] = None

    """
    Args for splitting into train, val, test
    to be updated once we have more options
    """
    # split seed
    split_seed: Optional[int] = None
    # size of validation dataset
    val_size: int
    # size of test dataset
    test_size: int

    # name of the column that contains the sequence
    sequence_column_name: str

    max_sequence_length: Optional[int] = None
    cache_dir: Optional[str] = None


def set_labels(result):
    result["labels"] = result["input_ids"].copy()
    return result


def train_val_test_split(
    dataset_dict: DatasetDict,
    config: DatasetConfig,
) -> DatasetDict:
    """
    Given a dictionary of datasets that only contains the split "train",
    optionally subsamples it, and then splits it
    so that it has potentially 3 splits: "train", "val", "test", where
    "val" and "test" splits do not exist if the specified sizes are 0
    """
    assert set(dataset_dict.keys()) == {
        "train"
    }, f"{train_val_test_split.__name__} expects its input to have the keys \
        ['train'] but the input has keys {list(dataset_dict.keys())}"

    dataset = dataset_dict["train"]

    val_size = config.val_size
    test_size = config.test_size

    assert isinstance(
        dataset, Dataset
    ), f"Invalid dataset type {type(dataset)}, only datasets.Dataset allowed"

    dataset = dataset.shuffle(seed=config.split_seed)

    if config.subsample_size is not None:
        dataset = dataset.select(range(config.subsample_size))

    valtest_size = val_size + test_size

    if valtest_size > 0:
        train_valtest = dataset.train_test_split(
            test_size=val_size + test_size,
            shuffle=False,
        )
        split_dict = {
            "train": train_valtest["train"],
        }
        if test_size > 0 and val_size > 0:
            test_val = train_valtest["test"].train_test_split(
                test_size=test_size,
                shuffle=False,
            )
            split_dict["val"] = test_val["train"]
            split_dict["test"] = test_val["test"]
        elif val_size > 0:
            split_dict["val"] = train_valtest["test"]
        else:
            split_dict["test"] = train_valtest["test"]
    else:
        split_dict = {
            "train": dataset,
        }

    split_dataset_dict = DatasetDict(split_dict)
    return split_dataset_dict




def load_ptm_dataset(df: pd.DataFrame, config: DatasetConfig) -> DatasetDict:
    ds = Dataset.from_pandas(df)
    ds_dict = DatasetDict({"train": ds})
    return train_val_test_split(ds_dict, config)


def create_token_dict_from_dataframe(
    df, seq_col="ori_seq", pos_col="pos", token_col="token"
):
    result_dict = {}

    for index, row in df.iterrows():
        ac_id = row[seq_col]
        pos = row[pos_col]
        token = row[token_col]

        if ac_id not in result_dict:
            result_dict[ac_id] = {}

        result_dict[ac_id][pos] = token

    return result_dict


def subsitute_tokens(sequence_lst, token_dict):
    if isinstance(sequence_lst, list):
        return [
            _substitute_token(sequence, token_dict[sequence])
            for sequence in sequence_lst
        ]
    elif isinstance(sequence_lst, str):
        return _substitute_token(sequence_lst, token_dict[sequence_lst])


def _substitute_token(sequence, token_dict):
    result = list(sequence)
    for position, new_tokens in token_dict.items():
        result[position] = new_tokens
    return "".join(result)


def construct_ptm_seq(
    batch, tokenizer, ptm_token_dict, sequence_column_name, max_sequence_length
):
    """
    apply transform to the batch to replace the tokens with the PTM tokens
    """
    batch['wt_seq'] = batch[sequence_column_name]
    batch[sequence_column_name] = subsitute_tokens(
        batch[sequence_column_name], ptm_token_dict
    )
    batch['ptm_seq'] = batch[sequence_column_name]
    batch["input_ids"] = tokenizer(
        batch[sequence_column_name],
        add_special_tokens=True,
        max_sequence_length=max_sequence_length,
    )
    # batch["labels"] = batch["input_ids"]
    return batch


def get_ptm_dataset(config_dict: Dict, tokenizer) -> Dataset:
    config = DatasetConfig(**config_dict)

    if config.dataset_type == "csv":
        df = pd.read_csv(config.dataset_loc)
        ptm_token_dict = create_token_dict_from_dataframe(df)
        df.drop(df.filter(regex="Unnamed"), axis=1, inplace=True)
        df.drop_duplicates(subset=config.sequence_column_name, inplace=True)
        split_dict = load_ptm_dataset(df, config)
    else:
        raise ValueError(f"Invalid dataset_type {config.dataset_type}!")

    split_dict = split_dict.map(
        lambda e: construct_ptm_seq(
            batch=e,
            tokenizer=tokenizer,
            ptm_token_dict=ptm_token_dict,
            sequence_column_name=config.sequence_column_name,
            max_sequence_length=config.max_sequence_length,
        ),
        batched=True,
        keep_in_memory=True,
    )

    return split_dict



if __name__ == "__main__":
    from protein_lm.tokenizer.tokenizer import PTMTokenizer
    from transformers import DataCollatorWithPadding, default_data_collator

    tokenizer = PTMTokenizer()
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer, padding="longest", max_length=1024, return_tensors="pt"
    )
    config_dict = {
        "dataset_type": "csv",
        "dataset_loc": "protein_lm/dataset/ptm_labels.csv",
        "subsample_size": None,
        "val_size": 0,
        "test_size": 0,
        "sequence_column_name": "ori_seq",
        "max_sequence_length": None,
        "dataset": "ptm",
    }
    dataset = get_ptm_dataset(config_dict, tokenizer)
    samples = dataset["train"][:8]
