from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch import nn, optim
from transformers import (DebertaV2Config, DebertaV2Model,
                          DebertaV2TokenizerFast,
                          get_linear_schedule_with_warmup)
from transformers.tokenization_utils import BatchEncoding

from ..metric import ScoreMetric
from ..utils import select_starts_ends
from .layers import Pooler, ToLSTMHiddenState


class BaseSpanPredictionModel(LightningModule):
    plm_name: str = 'deepset/deberta-v3-large-squad2'

    def __init__(self, from_pretrained: bool = False):
        super().__init__()

        self.tokenizer: DebertaV2TokenizerFast = DebertaV2TokenizerFast.from_pretrained(self.plm_name, cache_dir='.cache')

        if from_pretrained:
            self.encoder = DebertaV2Model.from_pretrained(self.plm_name, cache_dir='.cache')
        else:
            config = DebertaV2Config.from_pretrained(self.plm_name, cache_dir='.cache')
            self.encoder = DebertaV2Model(config)

        self.score_metric = ScoreMetric()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-5)
        return {
            'optimizer': optimizer,
            # 'lr_scheduler': {
            #     'scheduler': get_linear_schedule_with_warmup(optimizer, num_warmup_steps=200, num_training_steps=2000),
            #     'interval': 'step',
            #     'frequency': 1
            # },
        }

    def compute_loss(self, preds: Tuple[torch.Tensor], labels: Tuple[torch.Tensor]):
        loss = 0
        for logits, target in zip(preds, labels):
            loss += F.cross_entropy(logits, target)
        loss /= len(preds)
        return loss

    def validation_epoch_end(self, outputs):
        score = self.score_metric.compute()
        self.log('Score/Val', score)
        self.score_metric.reset()


class SpanPredictionModel(BaseSpanPredictionModel):
    def __init__(self, from_pretrained: bool = False):
        super().__init__(from_pretrained)

        self.tokenizer.add_special_tokens({'additional_special_tokens': ['[AGREE]', '[DISAGREE]']})
        self.encoder.resize_token_embeddings(len(self.tokenizer))

        hidden_size = self.encoder.config.hidden_size
        self.outputs = nn.Linear(hidden_size, 4)

    def __call__(self, encodings: BatchEncoding) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return super().__call__(encodings)

    def forward(self, encodings: BatchEncoding) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x: torch.Tensor = self.encoder(**encodings)[0]

        x = self.outputs(x)
        x = x.transpose(1, 2) # (N, 4, S)
        mask = ~encodings.attention_mask.bool().unsqueeze(1) # (N, 1, S)
        x = x.masked_fill(mask, -float('inf'))
        x = x.transpose(1, 2)

        qs_logits, qe_logits, rs_logits, re_logits = x.chunk(4, dim=-1)
        qs_logits = qs_logits.squeeze(-1).contiguous()
        qe_logits = qe_logits.squeeze(-1).contiguous()
        rs_logits = rs_logits.squeeze(-1).contiguous()
        re_logits = re_logits.squeeze(-1).contiguous()
        return qs_logits, qe_logits, rs_logits, re_logits
    
    def training_step(self, batch, batch_idx):
        encodings, targets = batch
        preds = self(encodings)
        loss = self.compute_loss(preds, targets)
        self.log('Loss/Train', loss)
        return loss

    def decode_answers(self, xs, encodings, valid_mask, preds, top_k: int = 1, max_tokens: int = 10000) -> List[Dict[str, str]]:
        b_qs_logits, b_qe_logits, b_rs_logits, b_re_logits = preds

        answers = []
        for i, x in enumerate(xs):
            qs_logits = b_qs_logits[i].unsqueeze(0)
            qe_logits = b_qe_logits[i].unsqueeze(0)
            rs_logits = b_rs_logits[i].unsqueeze(0)
            re_logits = b_re_logits[i].unsqueeze(0)

            q_valid_mask = valid_mask[i].clone().unsqueeze(0)
            r_valid_mask = valid_mask[i].clone().unsqueeze(0)

            rs = encodings.char_to_token(i, 0, 1)
            
            q_valid_mask[0, rs:] = False
            r_valid_mask[0, :rs] = False
            
            b_q_spans = select_starts_ends(qs_logits, qe_logits, q_valid_mask, top_k=top_k, max_tokens=max_tokens)
            b_r_spans = select_starts_ends(rs_logits, re_logits, r_valid_mask, top_k=top_k, max_tokens=max_tokens)

            q_prime = []
            r_prime = []

            for qs, qe, q_score in b_q_spans[0]:
                qs = encodings.token_to_chars(i, qs).start
                qe = encodings.token_to_chars(i, qe).end
                qp = x.q[qs:qe].strip()
                q_prime.append(qp)

            for rs, re, r_score in b_r_spans[0]:
                r = f'[{x.s}] {x.r}'
                rs = encodings.token_to_chars(i, rs).start
                re = encodings.token_to_chars(i, re).end
                rp = r[rs:re].strip()
                r_prime.append(rp)

            q_prime = ' '.join(q_prime)
            r_prime = ' '.join(r_prime)
            
            answer = dict(id=x.id, q=q_prime, r=r_prime)
            answers.append(answer)
        return answers

    def validation_step(self, batch, batch_idx):
        xs, encodings, state = batch        
        encodings.__setstate__(state)

        special_tokens_mask = encodings.pop('special_tokens_mask')
        valid_mask: torch.Tensor = encodings.attention_mask.bool()
        valid_mask &= ~special_tokens_mask.bool()
        valid_mask[:, 0] = True

        preds = self(encodings)

        answers = self.decode_answers(xs, encodings, valid_mask, preds)
        
        self.score_metric.update(answers)


class BaseSiameseSpanPredictionModel(BaseSpanPredictionModel):
    def __call__(self, q: BatchEncoding, r: BatchEncoding, s: torch.Tensor = None):
        return super().__call__(q, r, s)
    
    def training_step(self, batch, batch_idx):
        q, r, s, labels = batch
        preds = self(q, r, s)
        loss = self.compute_loss(preds, labels)
        self.log('Loss/Train', loss)
        return loss

    def decode_answers(self, xs, q, r, q_valid_mask, r_valid_mask, preds, top_k: int = 1, max_tokens: int = 48):
        qs_logits, qe_logits, rs_logits, re_logits = preds
        
        b_q_spans = select_starts_ends(qs_logits, qe_logits, q_valid_mask, top_k=top_k, max_tokens=max_tokens)
        b_r_spans = select_starts_ends(rs_logits, re_logits, r_valid_mask, top_k=top_k, max_tokens=max_tokens)

        answers = []
        for i, x in enumerate(xs):
            q_prime = []
            r_prime = []
            for q_span, r_span in zip(b_q_spans[i], b_r_spans[i]):
                qs, qe, q_score = q_span
                qs = q.token_to_chars(i, qs).start
                qe = q.token_to_chars(i, qe).end
                
                rs, re, r_score = r_span
                rs = r.token_to_chars(i, rs).start
                re = r.token_to_chars(i, re).end
                
                qp = x.q[qs:qe].strip()
                rp = x.r[rs:re].strip()

                q_prime.append(qp)
                r_prime.append(rp)

            q_prime = ' '.join(q_prime)
            r_prime = ' '.join(r_prime)
            
            answer = dict(id=x.id, q=q_prime, r=r_prime)
            answers.append(answer)
        return answers

    def validation_step(self, batch, batch_idx):
        xs, q, r, s, q_state, r_state = batch
        
        q.__setstate__(q_state)
        r.__setstate__(r_state)

        q_special_tokens_mask = q.pop('special_tokens_mask')
        r_special_tokens_mask = r.pop('special_tokens_mask')

        q_valid_mask: torch.Tensor = q.attention_mask.bool()
        q_valid_mask &= ~q_special_tokens_mask.bool()
        q_valid_mask[:, 0] = True

        r_valid_mask: torch.Tensor = r.attention_mask.bool()
        r_valid_mask &= ~r_special_tokens_mask.bool()
        r_valid_mask[:, 0] = True

        preds = self(q, r, s)

        answers = self.decode_answers(xs, q, r, q_valid_mask, r_valid_mask, preds)
        
        self.score_metric.update(answers)
    

class SiameseSpanPredictionModel(BaseSiameseSpanPredictionModel):
    def __init__(self, from_pretrained: bool = False):
        super().__init__(from_pretrained)

        hidden_size = self.encoder.config.hidden_size
        self.qr_attn = nn.MultiheadAttention(hidden_size, 1, batch_first=True, dropout=0.1)
        self.outputs = nn.Linear(hidden_size, 2)

    def forward(self, q: BatchEncoding, r: BatchEncoding, s: torch.Tensor = None):
        q_e = self.encoder(**q)[0]
        r_e = self.encoder(**r)[0]

        q_h, _ = self.qr_attn(q_e, r_e, r_e)
        r_h, _ = self.qr_attn(r_e, q_e, q_e)

        x = self.outputs(q_h)
        qs_logits, qe_logits = x.split(1, dim=-1)
        qs_logits = qs_logits.squeeze(-1).contiguous()
        qe_logits = qe_logits.squeeze(-1).contiguous()

        x = self.outputs(r_h)
        rs_logits, re_logits = x.split(1, dim=-1)
        rs_logits = rs_logits.squeeze(-1).contiguous()
        re_logits = re_logits.squeeze(-1).contiguous()

        return qs_logits, qe_logits, rs_logits, re_logits


class SiameseSpanPredictionModelWithLSTM(BaseSiameseSpanPredictionModel):
    def __init__(self, from_pretrained: bool = False):
        super().__init__(from_pretrained)

        hidden_size = self.encoder.config.hidden_size
        self.pooler = Pooler(hidden_size, self.encoder.config.pooler_dropout)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.to_hidden = ToLSTMHiddenState(self.lstm)
        self.to_context = ToLSTMHiddenState(self.lstm)
        self.lstm_dropout = nn.Dropout(0.1)

        self.attn = nn.MultiheadAttention(hidden_size * 2, 1, batch_first=True, dropout=0.1)

        self.to_span = nn.Linear(hidden_size * 2, 2)
    
    def forward(self, q: BatchEncoding, r: BatchEncoding, s: torch.Tensor = None):
        q_embeddings = self.encoder(**q)[0]
        r_embeddings = self.encoder(**r)[0]

        q_pooled = self.pooler(q_embeddings)
        r_pooled = self.pooler(r_embeddings)
        
        q_hc0 = self.to_hidden(r_pooled), self.to_context(r_pooled)
        r_hc0 = self.to_hidden(q_pooled), self.to_context(q_pooled)

        q_lstm_out, _ = self.lstm(q_embeddings, q_hc0)
        q_lstm_out = self.lstm_dropout(q_lstm_out)

        r_lstm_out, _ = self.lstm(r_embeddings, r_hc0)
        r_lstm_out = self.lstm_dropout(r_lstm_out)

        x, _ = self.attn(q_lstm_out, r_lstm_out, r_lstm_out)
        x = self.to_span(x)
        qs_logits, qe_logits = torch.split(x, 1, dim=-1)
        qs_logits = qs_logits.squeeze(-1).contiguous()
        qe_logits = qe_logits.squeeze(-1).contiguous()

        x, _ = self.attn(r_lstm_out, q_lstm_out, q_lstm_out)
        x = self.to_span(x)
        rs_logits, re_logits = torch.split(x, 1, dim=-1)
        rs_logits = rs_logits.squeeze(-1).contiguous()
        re_logits = re_logits.squeeze(-1).contiguous()

        return qs_logits, qe_logits, rs_logits, re_logits
