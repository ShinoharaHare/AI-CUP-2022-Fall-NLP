import json
import os
import sys
from difflib import SequenceMatcher
from collections import defaultdict
from typing import List, Tuple

from tqdm.auto import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src import RawDataset


def preprocess_str(s: str):
    return s.strip('" ')

def find_spans(s1: str, s2: str) -> List[Tuple[int, int]]:
    s1 = s1.split(' ')
    s2 = s2.split(' ')

    t2c = []
    a = 0
    for t in s1:
        s = a
        e = s + len(t) - 1
        t2c.append((s, e))
        a = e + 2

    sm = SequenceMatcher(a=s1, b=s2, autojunk=False)
    matchs = sm.get_matching_blocks()[:-1]
    spans = [(m.a, m.a + m.size) for m in matchs]
    spans = [(t2c[s][0], t2c[e - 1][1]) for s, e in spans]

    return spans

def check_spans(spans: List[Tuple[int, int]], s1: str, s2: str) -> bool:
    s = [s1[s:e + 1] for s, e in spans]
    s = ' '.join(s)
    return s == s2

if __name__ == '__main__':
    input_path = 'data/splitted/train.csv'
    output_path = 'data/multi_target/train.jsonl'

    dataset = RawDataset(input_path)

    id2item = {x.id: x for x in dataset}
    groups = defaultdict(list)

    for x in tqdm(dataset):
        q = preprocess_str(x.q)
        q_prime = preprocess_str(x.q_prime)
        q_spans = find_spans(q, q_prime)
        
        if not check_spans(q_spans, q, q_prime):
            continue

        r = preprocess_str(x.r)
        r_prime = preprocess_str(x.r_prime)
        r_spans = find_spans(r, r_prime)

        if not check_spans(r_spans, r, r_prime):
            continue

        groups[x.id].append((dict(q=q_prime, r=r_prime), dict(q=q_spans, r=r_spans)))

    with open(output_path, 'w', encoding='utf-8') as f:
        for id, l in groups.items():
            primes, spans = zip(*l)
            x = id2item[id]
            y = dict(
                id=x.id,
                q=x.q,
                r=x.r,
                s=x.s,
                primes=primes,
                spans=spans
            )
            f.write(json.dumps(y) + '\n')
