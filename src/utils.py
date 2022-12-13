from itertools import groupby
from typing import Dict, Iterable, List, Tuple

import torch

Span = Tuple[int, int]
Spans = List[Span]


def write_answers(answers: Iterable[Dict[str, str]], path: str):
    with open(path, 'w', encoding='utf-8') as f:
        f.write('id,q,r\n')
        for a in answers:
            q = a['q'].strip('" ')
            r = a['r'].strip('" ')
            f.write(f'{a["id"]},"""{q}""","""{r}"""\n')


def decode_spans(
    start: torch.Tensor,
    end: torch.Tensor,
    top_k: int,
    max_tokens: int,
    valid_mask: torch.Tensor
) -> List[Spans]:
    outer = start.unsqueeze(-1) @ end.unsqueeze(1)
    candidates = torch.tril(torch.triu(outer), max_tokens - 1)

    scores_flat = candidates.flatten(1)
    indices = scores_flat.topk(top_k).indices
    
    starts, ends = indices // candidates.size(1), indices % candidates.size(1)
    
    valid_indices = torch.arange(valid_mask.size(-1), device=start.device).repeat(valid_mask.size(0), 1)
    valid_indices[~valid_mask] = 0
    valid_indices = valid_indices.unsqueeze(1)
    valid_tokens = (starts.unsqueeze(-1) == valid_indices).any(-1) & (ends.unsqueeze(-1) == valid_indices).any(-1)

    b_spans = []
    for v, s, e, c in zip(valid_tokens, starts, ends, candidates):
        s = s[v]
        e = e[v]

        spans = []
        for si, ei, score in zip(s, e, c[s, e]):
            x = si.item(), ei.item(), score.item()
            spans.append(x)
        b_spans.append(spans)

    return b_spans


def nms(b_spans: List[List[Tuple[int, int, float]]], null_scores: torch.Tensor) -> List[Spans]:
    outputs = []

    for spans, null_score in zip(b_spans, null_scores):
        sorted_spans = sorted(spans, key=lambda x: x[2])
        spans = sorted_spans.copy()
        filtered_spans = []
        while spans:
            highest = spans.pop()
            hs, he, hscore = highest
            
            for i in range(len(spans) - 1, -1, -1):
                s, e, score = spans[i]
                if s <= he and hs <= e:
                    spans.pop(i)
            
            if not filtered_spans or hscore > null_score:
                filtered_spans.append(highest)
        
        filtered_spans = sorted(filtered_spans, key=lambda x: x[0])
        outputs.append(filtered_spans)

    return outputs


def select_starts_ends(
    start: torch.Tensor,
    end: torch.Tensor,
    valid_mask: torch.Tensor,
    top_k: int,
    max_tokens: int
) -> List[Spans]:
    start = torch.where(valid_mask, start, -float('inf'))
    end = torch.where(valid_mask, end, -float('inf'))

    start = torch.softmax(start, -1)
    end = torch.softmax(end, -1)

    null_score = start[:, 0] * end[:, 0]
    start[:, 0] = end[:, 0] = 0.0

    spans = decode_spans(start, end, top_k, max_tokens, valid_mask)
    spans = nms(spans, null_score)
    return spans


def indices_to_spans(indices: List[int], min_length: int = 1) -> Spans:   
    groups = groupby(enumerate(indices), key=lambda x: x[0] - x[1])
    groups = ([i[1] for i in g] for _, g in groups)
    groups = filter(lambda x: len(x) >= min_length, groups)
    groups = [(l[0], l[-1]) for l in groups]
    return groups
