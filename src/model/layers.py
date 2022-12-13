import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.deberta_v2.modeling_deberta_v2 import StableDropout


class Pooler(nn.Module):
    def __init__(self, hidden_size: int, dropout: float):
        super().__init__()

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = StableDropout(dropout)

    def forward(self, hidden_states):
        context_token = hidden_states[:, 0]
        context_token = self.dropout(context_token)
        pooled_output = self.dense(context_token)
        pooled_output = F.gelu(pooled_output)
        return pooled_output


class ToLSTMHiddenState(nn.Module):
    def __init__(self, lstm: nn.LSTM):
        super().__init__()

        self.num_layers = lstm.num_layers
        self.to_forward = nn.Linear(lstm.input_size, lstm.hidden_size * lstm.num_layers)
        self.to_backward = nn.Linear(lstm.input_size, lstm.hidden_size * lstm.num_layers)

    def forward(self, x: torch.Tensor):
        f = self.to_forward(x)
        f = f.unsqueeze(0).chunk(self.num_layers, -1)
        f = torch.cat(f)

        b = self.to_backward(x)
        b = b.unsqueeze(0).chunk(self.num_layers, -1)
        b = torch.cat(b)

        x = torch.cat([f, b])
        return x
