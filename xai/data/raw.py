import csv
from dataclasses import dataclass
from typing import Generator, List, Literal

from transformers import PreTrainedTokenizerFast

from .utils import Dataset


@dataclass
class RawItem:
    id: str
    q: str
    r: str
    s: Literal['AGREE', 'DISAGREE']
    q_prime: str
    r_prime: str

    @classmethod
    def from_dict(cls, d: dict):        
        return cls(
            id=d['id'],
            q=d['q'],
            r=d['r'],
            s=d['s'],
            q_prime=d['q\''],
            r_prime=d['r\''],
        )

    def to_tuple(self):
        return self.id, self.q, self.r, self.s, self.q_prime, self.r_prime


class RawDataset(Dataset):
    def __init__(self, path: str):
        super().__init__()
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            self.items = list(reader)

    def __len__(self) -> int:
        return len(self.items)
    
    def __iter__(self) -> Generator[RawItem, None, None]:
        return super().__iter__()
            
    def __getitem__(self, index: int) -> RawItem:
        x = self.items[index]
        x = RawItem.from_dict(x)
        return x

    
class RawDatasetForTraining(RawDataset):
    def __init__(self, path: str, tokenizer: PreTrainedTokenizerFast):
        super().__init__(path)

        self.tokenizer = tokenizer

    def collate_fn(self, batch: List[RawItem]):
        inputs, labels = zip(*[(f'{x.q}</s>{x.r}</s>{x.s}', f'{x.q_prime}</s>{x.r_prime}') for x in batch])
        encodings = self.tokenizer(inputs, text_target=labels)
        return encodings
