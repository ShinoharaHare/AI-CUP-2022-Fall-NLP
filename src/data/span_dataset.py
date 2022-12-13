import random
from typing import Callable, List, Optional

import torch
from transformers import PreTrainedTokenizerFast

from .data_item import SpanItem
from .utils import Dataset


class SpanDataset(Dataset):

    def __init__(
        self,
        path: str,
        tokenizer: PreTrainedTokenizerFast,
        item_filter: Optional[Callable] = None,
        extra_tokenizer_kwargs: Optional[dict] = None,
        random_span: bool = False
    ):
        super().__init__()

        self.items = []
        self.tokenizer = tokenizer

        with open(path, 'r', encoding='utf-8') as f:
            for l in f:
                x = SpanItem.from_json(l)
                if item_filter is None or item_filter(x):
                    self.items.append(x)

        self.extra_tokenizer_kwargs = extra_tokenizer_kwargs or {}
        self.random_span = random_span

    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, index: int) -> SpanItem:
        x = self.items[index]
        return x
    
    def collate_fn(self, batch: List[SpanItem]):
        b_q = []
        b_r = []

        for x in batch:
            b_q.append(x.q)
            b_r.append(f'[{x.s}] {x.r}')

        tokenizer_kwargs = dict(return_tensors='pt', return_token_type_ids=True, padding=True)
        tokenizer_kwargs.update(self.extra_tokenizer_kwargs)

        encodings = self.tokenizer(b_q, b_r, **tokenizer_kwargs)

        q_start = []
        q_end = []
        r_start = []
        r_end = []
        for i, x in enumerate(batch):
            s, e = random.choice(x.q_spans) if self.random_span else x.q_spans[0]
            s = encodings.char_to_token(i, s, 0)
            e = encodings.char_to_token(i, e, 0)

            q_start.append(s)
            q_end.append(e)

            s, e = random.choice(x.r_spans) if self.random_span else x.r_spans[0]
            s += len(x.s) + 3
            e += len(x.s) + 3

            s = encodings.char_to_token(i, s, 1)
            e = encodings.char_to_token(i, e, 1)

            r_start.append(s)
            r_end.append(e)

        q_start = torch.tensor(q_start)
        q_end = torch.tensor(q_end)
        r_start = torch.tensor(r_start)
        r_end = torch.tensor(r_end)

        return encodings, (q_start, q_end, r_start, r_end)


class SpanDatasetForSiamese(SpanDataset):
        
    def collate_fn(self, batch: List[SpanItem]):
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

        q_start = []
        q_end = []
        r_start = []
        r_end = []
        for i, x in enumerate(batch):
            si, ei = random.choice(x.q_spans) if self.random_span else x.q_spans[0]
            si = q.char_to_token(i, si, 0)
            ei = q.char_to_token(i, ei, 0)

            q_start.append(si)
            q_end.append(ei)

            si, ei = random.choice(x.r_spans) if self.random_span else x.r_spans[0]
            si = r.char_to_token(i, si, 0)
            ei = r.char_to_token(i, ei, 0)

            r_start.append(si)
            r_end.append(ei)

        q_start = torch.tensor(q_start)
        q_end = torch.tensor(q_end)
        r_start = torch.tensor(r_start)
        r_end = torch.tensor(r_end)
        
        return q, r, s, (q_start, q_end, r_start, r_end)
