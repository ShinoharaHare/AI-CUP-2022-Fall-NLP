from collections import defaultdict
import csv
import os
import random
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src import RawDataset

if __name__ == '__main__':
    seed = 1669483400
    props = [0.8, 0.1, 0.1] # train, val, test

    dataset = RawDataset('data/full.csv')
    groups = defaultdict(lambda: {})
    for x in dataset:
        groups[x.id][x] = x

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
            for id_ in ids[i:i + length]:
                to_tuple = lambda x: (x.id, x.q, x.r, x.s, x.q_prime, x.r_prime)
                writer.writerows(map(to_tuple, groups[id_].values()))
            i += length
