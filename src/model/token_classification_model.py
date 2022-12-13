from typing import Dict, List, Literal, Tuple

import torch
from pytorch_lightning import LightningModule
from torch import nn, optim
from transformers import (DebertaV2Model, DebertaV2TokenizerFast, DebertaV2Config,
                          get_linear_schedule_with_warmup)
from transformers.tokenization_utils import BatchEncoding

from ..metric import ScoreMetric
from ..utils import indices_to_spans


class BaseTokenClassificationModel(LightningModule):
    plm_name: str = 'deepset/deberta-v3-large-squad2'

    def __init__(self, from_pretrained: bool = False, loss_type: Literal['marginal_likelihood', 'most_likely_likelihood'] = 'marginal_likelihood'):
        super().__init__()

        self.tokenizer: DebertaV2TokenizerFast = DebertaV2TokenizerFast.from_pretrained(self.plm_name, cache_dir='.cache')

        if from_pretrained:
            self.encoder = DebertaV2Model.from_pretrained(self.plm_name, cache_dir='.cache')
        else:
            config = DebertaV2Config.from_pretrained(self.plm_name, cache_dir='.cache')
            self.encoder = DebertaV2Model(config)

        self.loss_type = loss_type

        self.score_metric = ScoreMetric()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-5)
        return {
            'optimizer': optimizer,
            # 'lr_scheduler': {
            #     'scheduler': get_linear_schedule_with_warmup(optimizer, num_warmup_steps=200),
            #     'interval': 'step',
            #     'frequency': 1
            # },
        }

    def marginal_likelihood(self, logp: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logp: (N, S, 2)
        # target: (N, ?, S)

        # (N, ?, S, 2)
        logp = logp.unsqueeze(1).expand(-1, target.size(1), -1, -1)

        # (N, ?, S)
        log_likelihoods = torch.gather(logp, dim=-1, index=target.unsqueeze(-1)).squeeze(-1)

        sequences_log_likelihoods = log_likelihoods.sum(dim=-1)
        sequences_log_likelihoods = sequences_log_likelihoods.masked_fill(target.sum(-1) == 0, -1e7)
        
        max_score = sequences_log_likelihoods.max(-1).values
        stable = sequences_log_likelihoods - max_score.unsqueeze(-1)
        log_marginal_likelihood = max_score + stable.exp().sum(-1).log()

        return log_marginal_likelihood
    
    def most_likely_likelihood(self, logp: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = logp.unsqueeze(1).expand(-1, target.size(1), -1, -1)
        log_likelihoods = torch.gather(logp, dim=-1, index=target.unsqueeze(-1)).squeeze(-1)

        sequences_log_likelihoods = log_likelihoods.sum(dim=-1)
        sequences_log_likelihoods = sequences_log_likelihoods.masked_fill(target.sum(-1) == 0, -1e7)
    
        most_likely_sequence_index = sequences_log_likelihoods.argmax(dim=-1)
        most_likely_sequence_likelihood = sequences_log_likelihoods.gather(dim=1, index=most_likely_sequence_index.unsqueeze(-1)).squeeze(dim=-1)
        return most_likely_sequence_likelihood

    def compute_loss(self, preds: Tuple[torch.Tensor, torch.Tensor], targets: Tuple[torch.Tensor]):
        logp = torch.cat(preds, dim=1)
        target = torch.cat(targets, dim=2)

        if self.loss_type == 'marginal_likelihood':
            likelyhood = self.marginal_likelihood(logp, target)
        elif self.loss_type == 'most_likely_likelihood':
            likelyhood = self.most_likely_likelihood(logp, target)
        
        loss = -likelyhood.mean()
        return loss

    def validation_epoch_end(self, outputs):
        score = self.score_metric.compute()
        self.log('Score/Val', score)
        self.score_metric.reset()


class TokenClassificationModel(BaseTokenClassificationModel):
    def __init__(self, from_pretrained: bool = False, loss_type: Literal['marginal_likelihood', 'most_likely_likelihood'] = 'marginal_likelihood'):
        super().__init__(from_pretrained, loss_type)

        hidden_size = self.encoder.config.hidden_size
        self.outputs = nn.Linear(hidden_size, 2)

    def __call__(self, encodings: BatchEncoding) -> torch.Tensor:
        return super().__call__(encodings)
    
    def forward(self, encodings: BatchEncoding) -> torch.Tensor:
        x = self.encoder(**encodings)[0]
        x = self.outputs(x) # (N, Sq+Sr, 2)
        logp = torch.log_softmax(x, -1)
        pmask = ~encodings.attention_mask.bool()
        logp = logp.masked_fill(pmask.unsqueeze(-1), 0)
        return logp
    
    def training_step(self, batch, batch_idx):
        encodings, targets = batch
        preds = self(encodings)
        loss = self.compute_loss([preds], [targets])
        self.log('Loss/Train', loss, batch_size=encodings.input_ids.size(0))
        return loss

    def decode_answers(self, xs, encodings, special_tokens_mask, logp) -> List[Dict[str, str]]:
        special_tokens_mask = special_tokens_mask.bool()
        logp[special_tokens_mask] = torch.tensor([0.0, -float('inf')])

        answers = []
        for i, x in enumerate(xs):
            tags = logp[i].argmax(-1)
            indices = torch.argwhere(tags).squeeze(1).tolist()

            rs = encodings.char_to_token(i, 0, 1)
            q_indices = []
            r_indices = []

            for idx in indices:
                if idx < rs:
                    q_indices.append(idx)
                else:
                    r_indices.append(idx)

            q_spans = indices_to_spans(q_indices)
            r_spans = indices_to_spans(r_indices)

            q_prime = []
            r_prime = []

            for s, e in q_spans:
                s = encodings.token_to_chars(i, s).start
                e = encodings.token_to_chars(i, e).end
                q_prime.append(x.q[s:e + 1].strip())

            for s, e in r_spans:
                r = f'[{x.s}] {x.r}'
                s = encodings.token_to_chars(i, s).start
                e = encodings.token_to_chars(i, e).end
                r_prime.append(r[s:e + 1].strip())
            
            q_prime = ' '.join(q_prime) or x.q
            r_prime = ' '.join(r_prime) or x.r
            
            answer = dict(id=x.id, q=q_prime, r=r_prime)
            answers.append(answer)
        
        return answers

    def validation_step(self, batch, batch_idx):
        xs, encodings, state = batch
        encodings.__setstate__(state)
        special_tokens_mask = encodings.pop('special_tokens_mask')
        preds = self(encodings)
        answers = self.decode_answers(xs, encodings, special_tokens_mask, preds)
        self.score_metric.update(answers)


class SiameseTokenClassificationModel(BaseTokenClassificationModel):
    def __init__(self, from_pretrained: bool = False, loss_type: Literal['marginal_likelihood', 'most_likely_likelihood'] = 'marginal_likelihood'):
        super().__init__(from_pretrained, loss_type)

        hidden_size = self.encoder.config.hidden_size
        self.qr_attn = nn.MultiheadAttention(hidden_size, 1, batch_first=True, dropout=0.1)
        self.output = nn.Linear(hidden_size, 2)

    def __call__(self, q: BatchEncoding, r: BatchEncoding, s: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        return super().__call__(q, r, s)
    
    def forward(self, q: BatchEncoding, r: BatchEncoding, s: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        q_e = self.encoder(**q)[0]
        r_e = self.encoder(**r)[0]

        q_pmask: torch.Tensor = ~q.attention_mask.bool()
        r_pmask: torch.Tensor = ~r.attention_mask.bool()

        q_h, _ = self.qr_attn(q_e, r_e, r_e, r_pmask)
        r_h, _ = self.qr_attn(r_e, q_e, q_e, q_pmask)

        q_logits = self.output(q_h) # (N, S, 2)
        r_logits = self.output(r_h) # (N, S, 2)

        q_logp = torch.log_softmax(q_logits, -1)
        r_logp = torch.log_softmax(r_logits, -1)

        q_logp = q_logp.masked_fill(q_pmask.unsqueeze(-1), 0)
        r_logp = r_logp.masked_fill(r_pmask.unsqueeze(-1), 0)

        return q_logp, r_logp
    
    def training_step(self, batch, batch_idx):
        q, r, s, targets = batch
        preds = self(q, r, s)
        loss = self.compute_loss(preds, targets)
        self.log('Loss/Train', loss, batch_size=q.input_ids.size(0))
        return loss
    
    def decode_answers(self, xs, q, r, q_special_tokens_mask, r_special_tokens_mask, preds) -> List[Dict[str, str]]:
        q_special_tokens_mask = q_special_tokens_mask.bool()
        r_special_tokens_mask = r_special_tokens_mask.bool()
        q_logp, r_logp = preds

        q_logp[q_special_tokens_mask] = torch.tensor([0.0, -float('inf')])
        r_logp[r_special_tokens_mask] = torch.tensor([0.0, -float('inf')])
        
        answers = []
        for i, x in enumerate(xs):
            q_tag = q_logp[i].argmax(-1)
            r_tag = r_logp[i].argmax(-1)

            q_indices = torch.argwhere(q_tag).squeeze(1).tolist()
            r_indices = torch.argwhere(r_tag).squeeze(1).tolist()

            q_spans = indices_to_spans(q_indices)
            r_spans = indices_to_spans(r_indices)

            q_prime = []
            r_prime = []

            for s, e in q_spans:
                s = q.token_to_chars(i, s).start
                e = q.token_to_chars(i, e).end
                q_prime.append(x.q[s:e + 1].strip())

            for s, e in r_spans:
                s = r.token_to_chars(i, s).start
                e = r.token_to_chars(i, e).end
                r_prime.append(x.r[s:e + 1].strip())
            
            q_prime = ' '.join(q_prime) or x.q
            r_prime = ' '.join(r_prime) or x.r
            
            answer = dict(id=x.id, q=q_prime, r=r_prime)
            answers.append(answer)
        
        return answers

    def validation_step(self, batch, batch_idx):
        xs, q, r, s, q_state, r_state = batch
        q.__setstate__(q_state)
        r.__setstate__(r_state)

        q_special_tokens_mask = q.pop('special_tokens_mask')
        r_special_tokens_mask = r.pop('special_tokens_mask')
        
        preds = self(q, r, s)
        answers = self.decode_answers(xs, q, r, q_special_tokens_mask, r_special_tokens_mask, preds)
        self.score_metric.update(answers)
