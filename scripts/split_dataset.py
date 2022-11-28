import os
import sys
import csv

import torch
from torch.utils.data import random_split

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from xai import RawDataset

if __name__ == '__main__':
    seed = 1669483400
    props = [0.8, 0.1, 0.1] # train, val, test

    dataset = RawDataset('data/train.csv')
    total = len(dataset)
    lengths = map(lambda x: int(x * total), props)
    lengths = list(lengths)
    lengths[0] += total - sum(lengths)

    generator = torch.Generator().manual_seed(seed)
    subsets = random_split(dataset, lengths=lengths, generator=generator)

    names = ['train', 'val', 'test']
    for name, subset in zip(names, subsets):
        with open(f'data/splitted/{name}.csv', 'w', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'q', 'r', 's', 'q\'', 'r\''])
            writer.writerows(map(lambda x: x.to_tuple(), subset))
