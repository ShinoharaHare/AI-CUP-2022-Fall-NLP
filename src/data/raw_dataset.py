import csv
from typing import Generator, List

from .data_item import RawItem
from .utils import Dataset


class RawDataset(Dataset):
    def __init__(self, path: str):
        super().__init__()

        self.items: List[RawItem] = []
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for d in reader:
                x = RawItem.from_dict(d)
                self.items.append(x)

    def __len__(self) -> int:
        return len(self.items)
    
    def __iter__(self) -> Generator[RawItem, None, None]:
        return super().__iter__()
            
    def __getitem__(self, index: int) -> RawItem:
        x = self.items[index]
        return x
