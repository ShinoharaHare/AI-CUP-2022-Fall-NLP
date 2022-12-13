from typing import List

import torch
from transformers import PreTrainedTokenizerFast

from .raw_dataset import RawDataset, RawItem


class PredictionDataset(RawDataset):
    def __init__(self, path: str, tokenizer: PreTrainedTokenizerFast):
        super().__init__(path)
        
        items = {x.id: x for x in self.items}
        self.items = list(items.values())
        self.tokenizer = tokenizer

    def collate_fn(self, batch: List[RawItem]):
        b_q = []
        b_r = []

        for i in range(len(batch)):
            x = batch[i] = batch[i].remove_quote()
            b_q.append(x.q)
            b_r.append(f'[{x.s}] {x.r}')

        encodings = self.tokenizer(b_q, b_r, return_tensors='pt', return_token_type_ids=True, return_special_tokens_mask=True, padding=True)

        return batch, encodings, encodings.__getstate__()
    

class PredictionDatasetForSiamese(PredictionDataset):
    
    def collate_fn(self, batch: List[RawItem]):
        b_q = []
        b_r = []
        b_s = []

        for i in range(len(batch)):
            x = batch[i] = batch[i].remove_quote()
            b_q.append(x.q)
            b_r.append(x.r)
            b_s.append(1 if x.s == 'AGREE' else 0)

        tokenizer_kwargs = dict(return_tensors='pt', return_special_tokens_mask=True, padding=True)
        q = self.tokenizer(b_q, **tokenizer_kwargs)
        r = self.tokenizer(b_r, **tokenizer_kwargs)
        s = torch.tensor(b_s, dtype=torch.long)
        return batch, q, r, s, q.__getstate__(), r.__getstate__()
