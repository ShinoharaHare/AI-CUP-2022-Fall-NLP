import csv
import os
import sys
from collections import OrderedDict

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from xai import RawDataset

if __name__ == '__main__':
    input_path = 'data/splitted/test.csv'
    output_path = 'data/dedup/test.csv'

    dataset = RawDataset(input_path)
    items = OrderedDict()
    for x in dataset:
        items[x.to_tuple()] = None
    items = items.keys()

    with open(output_path, 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'q', 'r', 's', 'q\'', 'r\''])
        writer.writerows(items)
