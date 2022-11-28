from typing import List, Tuple
import torch

from transformers import PreTrainedTokenizerFast
from datasets import load_dataset

from .raw import RawDataset, Dataset


class StanceDataset(RawDataset):
    def __init__(self, path: str, tokenizer: PreTrainedTokenizerFast):
        super().__init__(path)

        self.tokenizer = tokenizer

    def __getitem__(self, index: int) -> Tuple[str, str, int]:
        x = super().__getitem__(index)
        label = 0 if x.s == 'AGREE' else 1
        return x.q, x.r, label
    
    def collate_fn(self, batch: List[Tuple[str, str, float]]):
        qs, rs, labels = zip(*batch)
        
        encodings = self.tokenizer(qs, rs, return_tensors='pt', return_token_type_ids=True, padding=True, truncation=True)
        labels = torch.tensor(labels)
        
        return encodings, labels

    
class RumourEvalDataset(Dataset):
    """
    0 -> support
    1 -> deny
    2 -> query
    3 -> comment
    """

    def __init__(self, tokenizer: PreTrainedTokenizerFast):
        super().__init__()

        dataset = load_dataset('strombergnlp/rumoureval_2019', cache_dir='.cache')
        self.items = [x for s in dataset.values() for x in s]
        self.tokenizer = tokenizer

    def __getitem__(self, index) -> Tuple[str, str, int]:
        x = self.items[index]
        return x['source_text'], x['reply_text'], x['label']
    
    def collate_fn(self, batch: List[Tuple[str, str, float]]):
        qs, rs, labels = zip(*batch)
        
        encodings = self.tokenizer(qs, rs, return_tensors='pt', return_token_type_ids=True, padding=True, truncation=True)
        labels = torch.tensor(labels)
        
        return encodings, labels
