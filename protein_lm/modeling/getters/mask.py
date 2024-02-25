import torch


class Masker:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.mask_token_id = self.tokenizer.mask_token_id

    def random_mask(self, input_ids, mask_prob=0.15):
        device = input_ids.device
        mask = (torch.rand(input_ids.shape) < mask_prob).to(device)
        mask = mask & (torch.logical_not(self.tokenizer.is_special_token(input_ids)))
        masked_input_ids = input_ids.clone()
        masked_input_ids[mask] = self.mask_token_id
        return masked_input_ids, mask

    def mask_ptm_tokens(
        self,
        input_ids,
    ):
        device = input_ids.device
        is_ptm_mask = self.tokenizer.is_ptm_token(input_ids).to(device)
        is_ptm_mask = is_ptm_mask & (
            torch.logical_not(self.tokenizer.is_special_token(input_ids))
        )
        masked_input_ids = input_ids.clone()
        masked_input_ids[is_ptm_mask] = self.mask_token_id
        return masked_input_ids, is_ptm_mask

    def random_and_ptm_mask(self, input_ids, mask_prob=0.15):
        device = input_ids.device
        mask = (torch.rand(input_ids.shape) < mask_prob).to(device)
        mask = mask & (torch.logical_not(self.tokenizer.is_special_token(input_ids)))
        is_ptm_mask = self.tokenizer.is_ptm_token(input_ids).to(device)
        is_ptm_mask = is_ptm_mask & (
            torch.logical_not(self.tokenizer.is_special_token(input_ids))
        )
        mask = mask | is_ptm_mask
        masked_input_ids = input_ids.clone()
        masked_input_ids[mask] = self.mask_token_id
        return masked_input_ids, mask

    def random_or_random_and_ptm_mask(
        self, input_ids, ranom_mask_prob=0.15, alternate_prob=0.2
    ):
        """
        alternate between [(1) random mask] and [(2) random mask & ptm mask] by probability alternate_prob
        """
        p = torch.rand(1).item()
        if p < alternate_prob:
            return self.random_mask(input_ids, ranom_mask_prob)
        else:
            return self.random_and_ptm_mask(input_ids, ranom_mask_prob)

