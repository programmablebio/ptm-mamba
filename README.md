# PTM-Mamba

A PTM-Aware Protein Language Model with Bidirectional Gated Mamba Blocks.

## Install Enviroment

## Docker

Setting up env for mamba could be a pain, alternatively we suggest using docker containers.


#### Run container in interactive and detach mode, and mounte project dir to the container workspace.

```
docker run --gpus all -v $(pwd):/workspace -d -it --name plm_benji nvcr.io/nvidia/pytorch:23.12-py3 /bin/bash && docker attach plm_benji
```

#### Install pkgs in container

```
mkdir /root/.cache/torch/hub/checkpoints/ -p; wget -O /root/.cache/torch/hub/checkpoints/esm2_t33_650M_UR50D.pt https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt
cd protein_lm/modeling/models/libs/ && pip install -e causal-conv1d && pip install -e mamba && cd ../../../../
pip install transformers datasets accelerate evaluate pytest fair-esm biopython deepspeed wandb
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
pip install -e .
pip install hydra-core --upgrade
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
pip install -e protein_lm/tokenizer/rust_trie


```

## Data

We collect protein sequences and their PTM annotations from Uniprot-Swissprot. The PTM annotations are represented as tokens and used to replaced the amino acids. The data can be downloaded from  . Please move the data to  `protein_lm/dataset/`.

## Configs

The training and testing configs are `protein_lm/configs`. We provide a basic training config at `protein_lm/configs/train/base.yaml`.

## Training

##### Single-GPU Training

```
python ./protein_lm/modeling/scripts/train.py +train=base 
```

The commond will use the configs in `protein_lm/configs/train/base.yaml`.

##### Multi-GPU Training

We use [Distributed training with 🤗 Accelerate (huggingface.co)](https://huggingface.co/docs/transformers/main/accelerate).

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_processes=8 --multi_gpu protein_lm/modeling/scripts/train.py +train=base   train.report_to='wandb' train.training_arguments.per_device_train_batch_size=256 train.training_arguments.use_esm=True train.training_arguments.save_dir='ckpt/ptm_mamba' train.model.model_type='bidirectional_mamba' train.training_arguments.max_tokens_per_batch=40000 
```

- `report_to='wandb'`  tracks the training using wandb.
- `training_arguments.per_device_train_batch_size=300` sets the max batch size per device when constructing a batch.
- `training_arguments.max_tokens_per_batch=80000` sets the max num of tokens within a batch. If a batch exceeds the max token limit(depending on the seq len), we will trim the batch. Tune the `per_device_train_batch_size` and ``max_tokens_per_batch`` togather to maximize the memory usage during training. The rule of thumb is setting a large batch size (e.g., 300) while searching for the max num token that fits your GPU memory.
- `training_arguments.use_esm=True` uses the ESM embedding. By default, we use ESM 650M, and set the `model.esm_embed_dim: 1280` in `base.yaml`.  If disabled, the model will use its own embeddings.
- `training_arguments.save_dir='ckpt/bi_directional_mamba-esm'` where the model ckpts will be saved.
- `training_arguments.sample_len_ascending=true` is enable by default, samples sequences from short to long during the training.

##### Multi-GPU training with Deepspeed

Setup deepspeed with

```
accelerate config
```

and answer the questions asked. It will ask whether you want to use a config file for DeepSpeed to which you should answer no. Then answer the following questions to generate a basic DeepSpeed config. Use ZeRo 2 and FP32, which are sufficient for training our ~300M model without introducing overhead. This will generate a config file that will be used automatically to properly set the default options when launching training.

## Inference

The inference example is at `protein_lm/modeling/scripts/infer.py.` The model checkpoints can be downloaded from  . The outputs are:

```
Output = namedtuple("output", ["logits", "hidden_states"])
```

```
from protein_lm.modeling.scripts.infer import PTMMamba

ckpt_path = "ckpt/bi_mamba-esm-ptm_token_input/best.ckpt"
mamba = PTMMamba(ckpt_path,device='cuda:0')
seq = "M<N-acetylalanine>K"
output = mamba(seq)
print(output.logits.shape)
print(output.hidden_states.shape)
```

## Ackonwledgement

This project is based on the  following codebase. Please give them a star if you like our code.

- [OpenBioML/protein-lm-scaling (github.com)](https://github.com/OpenBioML/protein-lm-scaling)
- [state-spaces/mamba (github.com)](https://github.com/state-spaces/mamba)