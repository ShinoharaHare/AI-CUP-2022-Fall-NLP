import os
from itertools import accumulate
from typing import Any, Callable, Generator, List, Optional, Tuple, Type

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerFast


class Dataset(Dataset):
    worker_init_fn: Optional[Callable] = None
    collate_fn: Optional[Callable] = None

    def __init__(self):
        super().__init__()

    def __iter__(self) -> Generator:
        for i in range(len(self)):
            yield self[i]

            
class ConcatDataset(Dataset):
    def __init__(self, *datasets: List[Dataset]):
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


class DataModule(LightningDataModule):
    def __init__(
        self,
        dataset_cls: Type[Dataset],
        base_dir: str,
        tokenizer: Optional[PreTrainedTokenizerFast] = None,
        batch_size: Optional[int] = 1,
        num_workers: int = 0,
        prefetch_factor: int = 2,
        pin_memory: bool = False,
        extra_dataset_kwargs: Optional[dict] = None,
        extra_dataloader_kwargs: Optional[dict] = None
    ):
        super().__init__()

        self.dataset_cls = dataset_cls
        self.base_dir = base_dir
        self.batch_size = batch_size


        self.datasets = dict(
            train=None,
            val=None,
            test=None
        )

        self.dataset_kwargs = extra_dataset_kwargs or dict()
        if tokenizer is not None:
            self.dataset_kwargs['tokenizer'] = tokenizer

        self.dataloader_kwargs = extra_dataloader_kwargs or dict()
        self.dataloader_kwargs['batch_size'] = batch_size
        self.dataloader_kwargs['num_workers'] = num_workers
        self.dataloader_kwargs['prefetch_factor'] = prefetch_factor
        self.dataloader_kwargs['pin_memory'] = pin_memory

    def setup(self, stage: str = None):
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

        stages = [[None, 'fit'], [None, 'fit', 'validate'], [None, 'test']]
        names = ['train', 'val', 'test']

        for s, n in zip(stages, names):
            path = os.path.join(self.base_dir, f'{n}.csv')
            if stage in s and os.path.exists(path):
                self.datasets[n] = self.dataset_cls(path, **self.dataset_kwargs)

    def get_dataloader(self, t, *args, **kwargs):
        d = self.datasets[t]
        return DataLoader(d, *args, **kwargs, **self.dataloader_kwargs) if d is not None else None

    def train_dataloader(self):
        return self.get_dataloader('train', shuffle=True)

    def val_dataloader(self):
        return self.get_dataloader('val')

    def test_dataloader(self):
        return self.get_dataloader('test')
