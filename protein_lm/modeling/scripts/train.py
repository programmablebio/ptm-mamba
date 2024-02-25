import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from transformers.trainer import DataLoader
import torch.nn.functional as F
import os
import accelerate
from accelerate import Accelerator, DistributedType
from accelerate.utils import DistributedDataParallelKwargs

from accelerate.utils import LoggerType
from accelerate.local_sgd import LocalSGD
import esm

from protein_lm.modeling.getters.collate import (
    DataCollatorWithPadding,
    SequenceLengthSampler,
)
from accelerate.utils import set_seed
from protein_lm.modeling.getters.dataset import get_dataset

from protein_lm.modeling.getters.ptm_dataset import get_ptm_dataset
from protein_lm.modeling.getters.log import TrainLogger
from protein_lm.modeling.getters.mask import Masker
from protein_lm.modeling.getters.scheduler import Esm2LRScheduler
from protein_lm.modeling.models.mamba.lm import MambaLMHeadModel
from protein_lm.tokenizer.tokenizer import PTMTokenizer


def mlm_loss(outputs, input_ids, mask):
    return F.cross_entropy(
        outputs[mask],
        input_ids[mask],
    )



@torch.no_grad()
def compute_esm_embedding(tokenizer, esm_model, batch_converter, masked_input_ids):
    device = masked_input_ids.device
    esm_model = esm_model.to(device)
    inputs = [
        (i, tokenizer.decode(input_id.detach().cpu().tolist()))
        for i, input_id in enumerate(masked_input_ids)
    ]
    batch_labels, batch_strs, batch_tokens = batch_converter(inputs)
    batch_tokens = batch_tokens[..., 1:-1].to(
        device
    )  # remove <cls> and <eos> from ESM encoding
    out = esm_model(batch_tokens, repr_layers=[33], return_contacts=False)
    embedding = out["representations"][33]
    return embedding


@hydra.main(
    version_base=None, config_path="../../configs",
)
def main(config_dict: DictConfig):
    config_dict = config_dict['train']
    data_config = config_dict["dataset"]
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    device = accelerator.device
    train_args = config_dict["training_arguments"]
    set_seed(config_dict["seed"])
    if "wandb" in config_dict.report_to:
        import wandb

        if accelerator.is_local_main_process:
            wandb.init(
                project="PTM-Mamba", config=dict(config_dict), name=train_args.save_dir
            )
        logger = wandb
    else:
        logger = TrainLogger()

    tokenizer = PTMTokenizer()
    dataset = get_dataset(
        config_dict=data_config,
        tokenizer=tokenizer,
    )

    if train_args.resume_from_checkpoint:
        model = load_ckpt(train_args.resume_from_checkpoint, tokenizer, device)
        accelerator.print(f"Model loaded from {train_args.resume_from_checkpoint}")
    else:
        config_dict.model.vocab_size = tokenizer.get_vocab_size()
        model = MambaLMHeadModel(config=config_dict.model, device=device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    accelerator.print(f"Number of parameters: {num_params:,}")
    sampler = SequenceLengthSampler(
        dataset["train"], train_args.sort_by_seq, train_args.sample_len_ascending)
    train_loader = DataLoader(
        dataset["train"],
        batch_size=train_args.per_device_train_batch_size,
        sampler=sampler,
        collate_fn=DataCollatorWithPadding(
            max_tokens=train_args.max_tokens_per_batch,
            tokenizer=tokenizer,
            batch_by_tokens=True,
            max_seq_len=train_args.max_sequence_length,
        ),
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        dataset["val"],
        batch_size=train_args.per_device_train_batch_size // 2,
        collate_fn=DataCollatorWithPadding(
            max_tokens=train_args.max_tokens_per_batch,
            tokenizer=tokenizer,
            batch_by_tokens=False,
            max_seq_len=train_args.max_sequence_length,
        ),
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), train_args.lr, betas=(0.9, 0.98), weight_decay=0.01
    )

    scheduler = Esm2LRScheduler(
        optimizer, last_epoch=-1, init_lr=train_args.lr, on_use=False
    )

    masker = Masker(tokenizer)
    model, optimizer, scheduler, train_loader, val_loader = accelerator.prepare(
        model, optimizer, scheduler, train_loader, val_loader
    )

    train(
        config_dict=config_dict,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        tokenizer=tokenizer,
        masker=masker,
        logger=logger,
        accelerator=accelerator,
    )

def load_ckpt(ckpt_path, tokenizer, device):
    ckpt = torch.load(ckpt_path)
    model_state_dict = ckpt["model"]
    model_config = ckpt["config"]
    model_config.vocab_size = tokenizer.get_vocab_size()
    model = MambaLMHeadModel(config=model_config, device=device)
    msg = model.load_state_dict(model_state_dict, strict=True)
    print(msg)
    return model


def train(
    config_dict: DictConfig,
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    tokenizer,
    masker: Masker,
    logger,
    accelerator: Accelerator,
):
    train_args = config_dict["training_arguments"]
    save_dir = train_args.save_dir
    device = accelerator.device
    os.makedirs(save_dir, exist_ok=True)
    best_ckpt_path = os.path.join(save_dir, "best.ckpt")
    last_ckpt_path = os.path.join(save_dir, "last.ckpt")
    best_loss = float("inf")

    if train_args.use_esm:
        esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        batch_converter = alphabet.get_batch_converter()
        esm_model.eval()
        for param in esm_model.parameters():
            param.requires_grad = False
    model_to_save = model if accelerator.distributed_type==DistributedType.NO else model.module
    masking_fn = masker.random_or_random_and_ptm_mask
    total_steps = 0
    for epoch in range(train_args.num_train_epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"]
            pad_mask = batch["pad_mask"]
            esm_input_ids = make_esm_input_ids(input_ids, tokenizer)
            # masking
            masked_input_ids, pred_mask = masking_fn(input_ids)
            masked_esm_input_ids = esm_input_ids.clone()
            masked_esm_input_ids[pred_mask] = tokenizer.mask_token_id
            if train_args.use_esm:
                embedding = compute_esm_embedding(
                    tokenizer, esm_model, batch_converter, masked_esm_input_ids
                )
            else:
                embedding = None
            outputs = model(masked_input_ids, embedding=embedding)
            logits = outputs.logits
            loss = mlm_loss(logits, input_ids, pred_mask)
            accelerator.backward(loss)
            preplexity = torch.exp(loss)
            acc = (logits.argmax(dim=-1) == input_ids)[pred_mask].float().mean()
            ptm_acc = (
                (logits.argmax(dim=-1) == input_ids)[
                    pred_mask & tokenizer.is_ptm_token(input_ids).to(device)
                ]
                .float()
                .mean()
            )
            if accelerator.is_local_main_process:
                logger.log(
                    {
                        "train_loss": loss.item(),
                        "train_preplexity": preplexity.item(),
                        "train_acc": acc.item(),
                        "train_ptm_acc": ptm_acc.item(),
                        "act_bs": input_ids.shape[0],
                        "act_seq_len": input_ids.shape[1],
                    }
                )
            optimizer.step()
            scheduler.step()
            total_steps += 1
            if total_steps % train_args.log_steps == 0:
                model.eval()
                for val_batch in val_loader:
                    with torch.no_grad():
                        input_ids = val_batch["input_ids"]
                        pad_mask = val_batch["pad_mask"]
                        esm_input_ids = make_esm_input_ids(input_ids, tokenizer)
                        masked_input_ids, pred_mask = masking_fn(input_ids)
                        masked_esm_input_ids = esm_input_ids.clone()
                        masked_esm_input_ids[pred_mask] = tokenizer.mask_token_id
                        if train_args.use_esm:
                            embedding = compute_esm_embedding(
                                tokenizer, esm_model, batch_converter, masked_esm_input_ids
                            )
                        else:
                            embedding = None
                        outputs = model(masked_input_ids, embedding=embedding)
                        logits = outputs.logits
                        loss = mlm_loss(logits, input_ids, pred_mask)
                    preplexity = torch.exp(loss)
                    acc = (logits.argmax(dim=-1) == input_ids)[pred_mask].float().mean()
                    ptm_acc = (
                        (logits.argmax(dim=-1) == input_ids)[
                            pred_mask & tokenizer.is_ptm_token(input_ids).to(device)
                        ]
                        .float()
                        .mean()
                    )
                    
                    if accelerator.is_local_main_process:
                        logger.log(
                            {
                                "Epoch": epoch,
                                "val_loss": loss.item(),
                                "val_preplexity": preplexity.item(),
                                "val_acc": acc.item(),
                                "val_ptm_acc": ptm_acc.item(),
                            }
                        )
                        if loss < best_loss:
                            best_loss = loss
                            torch.save(
                                {"model": model_to_save.state_dict(), "config": config_dict.model},
                                best_ckpt_path,
                            )
                if accelerator.is_local_main_process:
                    torch.save(
                        {"model": model_to_save.state_dict(), "config": config_dict.model},
                        last_ckpt_path,
                    )
                    accelerator.print(f"Epoch {epoch}, Step {total_steps} finished!")
                    
    if accelerator.is_local_main_process:
        torch.save(
            {"model": model_to_save.state_dict(), "config": config_dict.model},
            last_ckpt_path,
        )
    accelerator.print(f"Training completed!")

def make_esm_input_ids(input_ids, tokenizer,):
    """
    Replace PTM tokens with mask token for ESM input
    """
    device = input_ids.device
    is_ptm_mask = tokenizer.is_ptm_token(input_ids).to(device)
    esm_input_ids = input_ids.clone()
    esm_input_ids[is_ptm_mask] = tokenizer.mask_token_id
    return esm_input_ids
        
if __name__ == "__main__":
    main()
