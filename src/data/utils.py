from itertools import accumulate
from typing import Any, Callable, Generator, Optional, Tuple

from torch.utils.data import DataLoader, Dataset


class Dataset(Dataset):
    worker_init_fn: Optional[Callable] = None
    collate_fn: Optional[Callable] = None

    def __init__(self):
        super().__init__()

    def __iter__(self) -> Generator:
        for i in range(len(self)):
            yield self[i]

            
class CombinedDataset(Dataset):

    @property
    def collate_fn(self):
        return self.datasets[0].collate_fn

    def __init__(self, *datasets: Dataset):
        super().__init__()

        self.datasets = datasets
        self.lengths = [len(x) for x in self.datasets]
        self.accumulation = [0, *accumulate(self.lengths, lambda a, b: a + b)]

    def __len__(self) -> int:
        return sum(self.lengths)
    
    def to_local_index(self, index: int) -> Tuple[int, int]:
        for i, a in enumerate(self.accumulation):
            if index < a:
                break
        return i - 1, index - self.accumulation[i - 1]

    def __getitem__(self, index: int) -> Any:
        di, li = self.to_local_index(index)
        x = self.datasets[di][li]
        return x

    
class DataLoader(DataLoader):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: Optional[bool] = None,
        num_workers: int = 0,
        worker_init_fn: Optional[Callable] = None,
        collate_fn: Optional[Callable] = None,
        prefetch_factor: int = 2,
        pin_memory: bool = False,
        *args,
        **kwargs,
    ):
        worker_init_fn = worker_init_fn or dataset.worker_init_fn
        collate_fn = collate_fn or dataset.collate_fn

        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            worker_init_fn=worker_init_fn,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            *args,
            **kwargs
        )
