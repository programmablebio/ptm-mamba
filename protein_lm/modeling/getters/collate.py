import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Sampler
from omegaconf import DictConfig, OmegaConf
from functools import partial


def crop_seq(input_ids, max_seq_len):
    """
    randomly crop sequences to max_seq_len
    Args:
        input_ids: tensor of shape (seq_len)
        max_seq_len: int
    """
    seq_len = len(input_ids)
    if seq_len <= max_seq_len:
        return input_ids
    else:
        start_idx = torch.randint(0, seq_len - max_seq_len + 1, (1,)).item()
        return input_ids[start_idx : start_idx + max_seq_len]

class DataCollatorWithPadding:
    def __init__(
        self,
        max_tokens,
        tokenizer,
        batch_by_tokens=False,
        max_seq_len=None,
    ) -> None:
        self.max_tokens = max_tokens
        self.tokenizer = tokenizer
        self.batch_by_tokens = batch_by_tokens
        self.max_seq_len = max_seq_len
        self.crop_fn = partial(crop_seq, max_seq_len=max_seq_len) if max_seq_len is not None else lambda x: x
        
    def __call__(self, batch):
        """
        generate a batch of data g
        Args:
        batch: list of dictionaries with keys 'input_ids' and 'labels'

        """
        input_ids = [self.crop_fn(i["input_ids"]) for i in batch]
        input_ids = pad_sequence(
            [torch.tensor(x) for x in input_ids],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        if self.batch_by_tokens:
            # keep a few sequences to make the total number of tokens in the batch <= max_tokens
            total_tokens = input_ids.numel()
            if total_tokens > self.max_tokens:
                max_num_seq = self.max_tokens // input_ids.shape[-1] + 1
                # randomly select max_num_seq sequences from the batch to keep
                indices = torch.randperm(len(input_ids))[:max_num_seq]
                input_ids = input_ids[indices]
        pad_mask = input_ids != self.tokenizer.pad_token_id

        return {
            "input_ids": input_ids,
            "pad_mask": pad_mask,
        }


class SequenceLengthSampler(Sampler):
    def __init__(self, dataset, sort=True, sample_len_ascending=True):
        """
        Args:
            dataset: a dataset with keys 'input_ids' and 'labels'
            sample_len_ascending: if True, sample shorter sequences first
        """
        self.dataset = dataset
        self.indices = list(range(len(dataset)))
        if sort is True:
            self.indices.sort(
                key=lambda x: len(dataset[x]["input_ids"]),
                reverse=not sample_len_ascending
            )

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
