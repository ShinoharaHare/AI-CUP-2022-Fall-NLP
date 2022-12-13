import csv
import json
import os
from typing import Dict, List

import nltk
from tqdm.auto import tqdm

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


PUNCTUATIONS = set(r'!"#$%&\'()*+, -./:;<=>?@[\]^_`{|}~')
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

with open(os.path.join(DATA_DIR, 'q_primes.json'), 'r', encoding='utf-8') as f:
    GOLD_Q_PRIMES: Dict[str, str] = json.load(f)

with open(os.path.join(DATA_DIR, 'r_primes.json'), 'r', encoding='utf-8') as f:
    GOLD_R_PRIMES: Dict[str, str] = json.load(f)


def preprocess(s: str) -> List[str]:
    tokens = nltk.word_tokenize(s)
    tokens = filter(lambda s: len(s) != 1 or s not in PUNCTUATIONS, tokens)
    tokens = list(tokens)
    return tokens


def compute_lcs(s1: List[str], s2: List[str]) -> int:
    m = len(s1)
    n = len(s2)
 
    l = [[None] * (n + 1) for _ in range(m + 1)]
 
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0 :
                l[i][j] = 0
            elif s1[i - 1] == s2[j - 1]:
                l[i][j] = l[i - 1][j - 1] + 1
            else:
                l[i][j] = max(l[i - 1][j], l[i][j - 1])

    return l[m][n]


def compute_score_single(pred: List[str], gold_answers: List[List[str]]) -> float:
    score = 0
    for gold_answer in gold_answers:
        lcs = compute_lcs(pred, gold_answer)
        u = len(pred) + len(gold_answer) - lcs
        score = max(score, lcs / u)
    return score


def compute_score(answers: List[Dict[str, str]]) -> float:
    answers = {x['id']: x for x in answers}
    answers = list(answers.values())
    score = 0
    for x in tqdm(answers):
        q_prime = '"' + x['q'].strip(' "') + '"'
        r_prime = '"' + x['r'].strip(' "') + '"'

        q_prime = preprocess(q_prime)
        r_prime = preprocess(r_prime)

        score_q = compute_score_single(q_prime, GOLD_Q_PRIMES[x['id']])
        score_r = compute_score_single(r_prime, GOLD_R_PRIMES[x['id']])

        score += score_q + score_r
    
    score /= len(answers) * 2
    return score


def compute_csv(path: str) -> float:
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        answers = list(reader)
    return compute_score(answers)
