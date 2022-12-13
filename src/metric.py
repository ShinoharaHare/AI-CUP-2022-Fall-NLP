from typing import Dict, List
import torch
from torchmetrics import Metric
from .scoring import compute_score


class ScoreMetric(Metric):
    full_state_update = True

    answers: List[Dict[str, str]]

    def __init__(self):
        super().__init__()

        self.add_state('answers', [])

    def update(self, answers: List[Dict[str, str]]):
        self.answers.extend(answers)

    def compute(self):
        score = compute_score(self.answers)
        score = torch.tensor(score)
        return score
