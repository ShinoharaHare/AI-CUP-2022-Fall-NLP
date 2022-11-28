from pytorch_lightning import LightningModule
from transformers import LongT5ForConditionalGeneration, T5TokenizerFast
from torch import optim

from .stance import *

class TestModel(LightningModule):
    def __init__(self):
        super().__init__()

        self.t5 = LongT5ForConditionalGeneration.from_pretrained('pszemraj/long-t5-tglobal-base-16384-book-summary', cache_dir='.cache')

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=5e-4)
        
    def training_step(self, batch, batch_idx):
        output = self.t5(**batch)
        return output.loss
    
    @staticmethod
    def get_tokenizer():
        return T5TokenizerFast.from_pretrained('pszemraj/long-t5-tglobal-base-16384-book-summary', cache_dir='.cache')
