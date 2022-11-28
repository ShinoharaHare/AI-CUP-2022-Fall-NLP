from collections import defaultdict
from typing import Dict, List, Optional

import nltk
import torch
from torchmetrics import Metric

from ..data import RawDataset
from ..model.utils import ModelOutput

PUNCTUATIONS = r'!"#$%&\'()*+, -./:;<=>?@[\]^_`{|}~'

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def preprocess(s: str) -> List[str]:
    tokens = nltk.word_tokenize(s)
    tokens = filter(lambda s: len(s) == 1 and s in PUNCTUATIONS, tokens)
    tokens = list(tokens)
    return tokens


def compute_lcs(x: List[str], y: List[str], m: Optional[int] = None, n: Optional[int] = None):
    m = len(x) if m is None else m
    n = len(y) if n is None else n

    if m == 0 or n == 0:
        return 0
    
    if x[m - 1] == y[n - 1]:
        return compute_lcs(x, y, m - 1, n - 1) + 1
    
    return max(compute_lcs(x, y, m, n - 1), compute_lcs(x, y, m - 1, n))


def compute_score(pred: List[str], gold_answers: List[List[str]]) -> float:
    score = 0
    for gold_answer in gold_answers:
        lcs = compute_lcs(pred, gold_answer)
        u = len(pred) + len(gold_answer) - lcs
        score = max(score, lcs / u)
    return score 


class LCSMetric(Metric):
    def __init__(self, dataset: RawDataset):
        super().__init__()

        self.gold_q_prime: Dict[str, List[List[str]]] = defaultdict(lambda: [])
        self.gold_r_prime: Dict[str, List[List[str]]] = defaultdict(lambda: [])
        for x in dataset:
            self.gold_q_prime[x.id].append(preprocess(x.q_prime))
            self.gold_r_prime[x.id].append(preprocess(x.r_prime))

        self.add_state('score', torch.zeros(1))
        self.add_state('total', torch.zeros(1))

    def update(self, preds: List[ModelOutput]):
        for x in preds:
            q_prime = preprocess(x.q_prime)
            r_prime = preprocess(x.r_prime)
            s_q = compute_score(q_prime, self.gold_q_prime[x.id])
            s_r = compute_score(r_prime, self.gold_r_prime[x.id])
            s = (s_q + s_r) / 2
            self.score += s

        self.total += len(preds)

    def compute(self):
        return self.score / self.total
