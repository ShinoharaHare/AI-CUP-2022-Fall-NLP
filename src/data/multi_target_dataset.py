from typing import Callable, List, Optional

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizerFast

from src.data.data_item import MultiTargetItem

from .utils import Dataset


class MultiTargetDataset(Dataset):
    def __init__(
        self,
        path: str,
        tokenizer: PreTrainedTokenizerFast,
        item_filter: Optional[Callable[[MultiTargetItem], bool]] = None,
        extra_tokenizer_kwargs: Optional[dict] = None
    ):
        super().__init__()

        self.items = []
        self.tokenizer = tokenizer

        with open(path, 'r', encoding='utf-8') as f:
            for l in f:
                x = MultiTargetItem.from_json(l)
                if item_filter is None or item_filter(x):
                    self.items.append(x)
        self.extra_tokenizer_kwargs = extra_tokenizer_kwargs or {}

    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, index: int) -> MultiTargetItem:
        x = self.items[index]
        return x
    
    def collate_fn(self, batch: List[MultiTargetItem]):
        b_q = []
        b_r = []

        for x in batch:
            b_q.append(x.q)
            b_r.append(f'[{x.s}] {x.r}')

        tokenizer_kwargs = dict(return_tensors='pt', return_token_type_ids=True, padding=True)
        tokenizer_kwargs.update(self.extra_tokenizer_kwargs)

        encodings = self.tokenizer(b_q, b_r, **tokenizer_kwargs)

        b_targets = [] # (N, ?, S)
        for i, x in enumerate(batch):
            targets = [] # (?, S)
            for spans in x.spans:
                target = torch.zeros(encodings.input_ids.size(1)) # (S,)
                
                for s, e in spans.q:
                    s = encodings.char_to_token(i, s, 0)
                    e = encodings.char_to_token(i, e, 0)
                    target[s:e + 1] = 1

                for s, e in spans.r:
                    s = encodings.char_to_token(i, s + len(x.s) + 3, 1)
                    e = encodings.char_to_token(i, e + len(x.s) + 3, 1)
                    target[s:e + 1] = 1

                target = target.unsqueeze(0)
                targets.append(target)

            targets = torch.cat(targets)
            b_targets.append(targets)

        b_targets = pad_sequence(b_targets, batch_first=True, padding_value=0)
        b_targets = b_targets.long()

        return encodings, b_targets


class MultiTargetDatasetForSiamese(MultiTargetDataset):

    def collate_fn(self, batch: List[MultiTargetItem]):
        b_q = []
        b_r = []
        b_s = []

        for x in batch:
            b_q.append(x.q)
            b_r.append(x.r)
            b_s.append(1 if x.s == 'AGREE' else 0)

        tokenizer_kwargs = dict(return_tensors='pt', padding=True)
        tokenizer_kwargs.update(self.extra_tokenizer_kwargs)

        q = self.tokenizer(b_q, **tokenizer_kwargs)
        r = self.tokenizer(b_r, **tokenizer_kwargs)
        s = torch.tensor(b_s)

        b_q_targets = [] # (N, ?, Sq)
        b_r_targets = [] # (N, ?, Sr)
        for i, x in enumerate(batch):
            q_targets = [] # (?, Sq)
            r_targets = [] # (?, Sr)
            for spans in x.spans:
                q_t = torch.zeros(q.input_ids.size(1)) # (Sq,)
                r_t = torch.zeros(r.input_ids.size(1)) # (Sr,)
                
                for s, e in spans.q:
                    s = q.char_to_token(i, s)
                    e = q.char_to_token(i, e)
                    q_t[s:e + 1] = 1

                for s, e in spans.r:
                    s = r.char_to_token(i, s)
                    e = r.char_to_token(i, e)
                    r_t[s:e + 1] = 1

                q_t = q_t.unsqueeze(0)
                r_t = r_t.unsqueeze(0)
                q_targets.append(q_t)
                r_targets.append(r_t)

            q_targets = torch.cat(q_targets)
            r_targets = torch.cat(r_targets)

            b_q_targets.append(q_targets)
            b_r_targets.append(r_targets)

        b_q_targets = pad_sequence(b_q_targets, batch_first=True)
        b_r_targets = pad_sequence(b_r_targets, batch_first=True)
        b_q_targets = b_q_targets.long()
        b_r_targets = b_r_targets.long()

        return q, r, s, (b_q_targets, b_r_targets)
