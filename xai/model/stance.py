from typing import Optional

from pytorch_lightning import LightningModule
from torch import nn, optim
from torchmetrics import Accuracy, F1Score
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification


class StanceModel(LightningModule):
    PLM: str = 'deepset/roberta-base-squad2'

    def __init__(self, base_model: Optional[nn.Module] = None):
        super().__init__()

        if base_model is None:
            baseline = AutoModel.from_pretrained(self.PLM)
            baseline.config.num_labels = 1
            base_model = AutoModelForSequenceClassification.from_config(baseline.config)
            setattr(base_model, base_model.base_model_prefix, baseline)

        self.base_model: nn.Module = base_model

        self.metrics = nn.ModuleDict({
            'Accuracy/Train': Accuracy(num_classes=1),
            'Accuracy/Val': Accuracy(num_classes=1),
            'Accuracy/Test': Accuracy(num_classes=1),
            'F1-Score/Train': F1Score(num_classes=1),
            'F1-Score/Val': F1Score(num_classes=1),
            'F1-Score/Test': F1Score(num_classes=1),
        })

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=1e-5)
    
    def __call__(self, *args, **kwargs):
        x = self.base_model(*args, **kwargs)
        return x
        
    def training_step(self, batch, batch_idx):
        encodings, labels = batch
        output = self(**encodings, labels=labels.float())
        loss = output.loss
        preds = output.logits.squeeze()

        self.metrics['Accuracy/Train'](preds, labels)
        self.metrics['F1-Score/Train'](preds, labels)

        self.log('Loss/Train', loss)
        self.log('Accuracy/Train', self.metrics['Accuracy/Train'])
        self.log('F1-Score/Train', self.metrics['F1-Score/Train'])
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        encodings, labels = batch
        output = self(**encodings, labels=labels)
        loss = output.loss
        preds = output.logits.squeeze()

        self.metrics['Accuracy/Val'](preds, labels)
        self.metrics['F1-Score/Val'](preds, labels)

        self.log('Loss/Val', loss)
        self.log('Accuracy/Val', self.metrics['Accuracy/Val'])
        self.log('F1-Score/Val', self.metrics['F1-Score/Val'])
        
        return loss

    def test_step(self, batch, batch_idx):
        encodings, labels = batch
        output = self(**encodings, labels=labels)
        loss = output.loss
        preds = output.logits.squeeze()

        self.metrics['Accuracy/Test'](preds, labels)
        self.metrics['F1-Score/Test'](preds, labels)

        self.log('Loss/Test', loss)
        self.log('Accuracy/Test', self.metrics['Accuracy/Test'])
        self.log('F1-Score/Test', self.metrics['F1-Score/Test'])
        
        return loss
    
    @classmethod
    def get_tokenizer(cls):
        return AutoTokenizer.from_pretrained(cls.PLM)
