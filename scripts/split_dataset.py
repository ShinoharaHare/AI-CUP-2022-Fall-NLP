from collections import defaultdict
import csv
import os
import random
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from xai import RawDataset

if __name__ == '__main__':
    seed = 1669483400
    props = [0.8, 0.1, 0.1] # train, val, test

    dataset = RawDataset('data/train.csv')
    groups = defaultdict(lambda: [])
    for x in dataset:
        groups[x.id].append(x)

    total = len(groups)
    lengths = map(lambda x: int(x * total), props)
    lengths = list(lengths)
    lengths[0] += total - sum(lengths)

    ids = list(groups.keys())
    random.Random(seed).shuffle(ids)

    i = 0
    names = ['train', 'val', 'test']
    for name, length in zip(names, lengths):
        with open(f'data/splitted/{name}.csv', 'w', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'q', 'r', 's', 'q\'', 'r\''])
            for id_ in ids[i:i+length]:
                writer.writerows(map(lambda x: x.to_tuple(), groups[id_]))
            i += length
