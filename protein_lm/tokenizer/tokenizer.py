import torch
from typing import List, Union, Optional
from rust_trie import Trie
import os


class Tokenizer:
    def __init__(self, tokens: List[str], unk_token_id: Optional[int] = None):
        self.ids_to_tokens = tokens
        self.trie = Trie(unk_token_id)
        for token in tokens:
            self.trie.add(token)
        # If unk_token_id is not provided, add <unk> to the end of the tokens list
        if unk_token_id is None:
            self.ids_to_tokens += ["<unk>"]
        self.pad_token_id = self.ids_to_tokens.index("<pad>")
        self.mask_token_id = self.ids_to_tokens.index("<mask>")

    def __call__(self, sequences: Union[str, List], *args, **kwargs):
        if isinstance(sequences, str):
            return self.encode(sequences, *args, **kwargs)
        else:
            return self.batch_encode(sequences, *args, **kwargs)

    def encode(
        self,
        sequence: str,
        add_special_tokens: bool = False,
        return_tensor: bool = False,
        max_sequence_length: Optional[int] = None,
    ) -> List[int]:
        if max_sequence_length is not None:
            if add_special_tokens:
                max_sequence_length -= 2
            if len(sequence) > max_sequence_length:
                # randomly crop the sequence
                start_idx = torch.randint(
                    0, len(sequence) - max_sequence_length + 1, (1,)
                )
                sequence = sequence[start_idx : start_idx + max_sequence_length]

        if add_special_tokens:
            sequence = "<cls>" + sequence + "<eos>"
        output = self.trie.tokenize(sequence)
        if return_tensor:
            output = torch.tensor(output, dtype=torch.long)
        return output

    def batch_encode(
        self,
        sequences: List[str],
        add_special_tokens: bool = False,
        return_tensors: bool = False,
        max_sequence_length: Optional[int] = None,
    ) -> List[List[int]]:
        output = []
        if max_sequence_length is None and return_tensors:
            max_sequence_length = max([len(sequence) for sequence in sequences])
            if add_special_tokens:
                max_sequence_length += 2
        # if max_sequence_length is not None:
        #     sequences = [
        #         sequence[
        #             : (max_sequence_length - 2)
        #             if add_special_tokens
        #             else max_sequence_length
        #         ]
        #         for sequence in sequences
        #     ]
        for sequence in sequences:
            output.append(
                self.encode(
                    sequence,
                    add_special_tokens,
                    return_tensors,
                    max_sequence_length=max_sequence_length,
                )
            )
        if return_tensors:
            tensor_out = torch.full(
                (len(output), max_sequence_length), self.pad_token_id
            )
            for i, sequence in enumerate(output):
                tensor_out[i, : len(sequence)] = sequence
            output = tensor_out
        return output

    def decode(self, tokens: List[int]) -> str:
        return "".join([self.ids_to_tokens[idx] for idx in tokens])


class EsmTokenizer(Tokenizer):
    def __init__(self):
        tokens = [
            "<cls>",
            "<pad>",
            "<eos>",
            "<unk>",
            "L",
            "A",
            "G",
            "V",
            "S",
            "E",
            "R",
            "T",
            "I",
            "D",
            "P",
            "K",
            "Q",
            "N",
            "F",
            "Y",
            "M",
            "H",
            "W",
            "C",
            "X",
            "B",
            "U",
            "Z",
            "O",
            ".",
            "-",
            "<null_1>",
            "<mask>",
        ]
        super().__init__(tokens, unk_token_id=3)


class PTMTokenizer(Tokenizer):
    def __init__(self):
        tokens = [
            "<cls>",
            "<pad>",
            "<eos>",
            "<unk>",
            ".",
            "-",
            "<null_1>",
            "<mask>",
            "L",
            "A",
            "G",
            "V",
            "S",
            "E",
            "R",
            "T",
            "I",
            "D",
            "P",
            "K",
            "Q",
            "N",
            "F",
            "Y",
            "M",
            "H",
            "W",
            "C",
            "X",
            "B",
            "U",
            "Z",
            "O",
            "PTM",
            "<N-linked (GlcNAc...) asparagine>",
            "<Pyrrolidone carboxylic acid>",
            "<Phosphoserine>",
            "<Phosphothreonine>",
            "<N-acetylalanine>",
            "<N-acetylmethionine>",
            "<N6-acetyllysine>",
            "<Phosphotyrosine>",
            "<S-diacylglycerol cysteine>",
            "<N6-(pyridoxal phosphate)lysine>",
            "<N-acetylserine>",
            "<N6-carboxylysine>",
            "<N6-succinyllysine>",
            "<S-palmitoyl cysteine>",
            "<O-(pantetheine 4'-phosphoryl)serine>",
            "<Sulfotyrosine>",
            "<O-linked (GalNAc...) threonine>",
            "<Omega-N-methylarginine>",
            "<N-myristoyl glycine>",
            "<4-hydroxyproline>",
            "<Asymmetric dimethylarginine>",
            "<N5-methylglutamine>",
            "<4-aspartylphosphate>",
            "<S-geranylgeranyl cysteine>",
            "<4-carboxyglutamate>",
        ]
        super().__init__(tokens, unk_token_id=3)
        self.ptm_token_start = self.ids_to_tokens.index("PTM")

    def is_ptm_token(self, input_ids: torch.tensor):
        return input_ids > self.ptm_token_start

    def is_special_token(self, input_ids: torch.tensor):
        l_id = self.ids_to_tokens.index("L")
        return input_ids < l_id

    def __len__(self):
        return len(self.ids_to_tokens)

    def get_vocab_size(self):
        return len(self.ids_to_tokens)


class AptTokenizer(Tokenizer):
    def __init__(self):
        # For our own tokenizers, we don't need to explicitly add the <unk> token
        # because it gets added as the last token in the tokens list
        # I've also removed X so that it gets translated to <unk>
        tokens = [
            "<cls>",
            "<pad>",
            "<eos>",
            "L",
            "A",
            "G",
            "V",
            "S",
            "E",
            "R",
            "T",
            "I",
            "D",
            "P",
            "K",
            "Q",
            "N",
            "F",
            "Y",
            "M",
            "H",
            "W",
            "C",
            "B",
            "U",
            "Z",
            "O",
            "<mask>",
        ]
        super().__init__(tokens)
