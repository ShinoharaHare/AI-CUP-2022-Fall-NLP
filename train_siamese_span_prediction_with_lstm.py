import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src import *


@dataclass
class Config:
    name: str
    batch_size: int

    max_epochs: int = -1
    log_every_n_steps: Optional[int] = 50
    save_every_n_steps: Optional[int] = None
    val_check_interval: Optional[int] = None
    num_workers: int = 8
    accumulate_grad_batches: int = 1
    save_dir: str = 'lightning_logs'
    ckpt_path: Optional[str] = None

    def __post_init__(self):
        if not self.ckpt_path:
            ckpt_dir = Path(self.save_dir).joinpath(self.name, 'checkpoints')
            ckpts = sorted(ckpt_dir.glob('e*.ckpt'))
            if ckpts:
                last_path = ckpts[-1]
                self.ckpt_path = str(last_path) if last_path.exists() else None


def main():
    config = Config(
        name='SiameseSpanPredictionModelWithLSTM',
        batch_size=4,
        accumulate_grad_batches=8,
        save_every_n_steps=100,
        val_check_interval=500,
        max_epochs=1,
        # ckpt_path='',
    )

    model = SiameseSpanPredictionModelWithLSTM(from_pretrained=True)
    tokenizer = model.tokenizer
    
    with open('data/length/deberta-v3.json', 'r') as f:
        lengths = json.load(f)

    def item_filter(x):
        qn, rn = lengths[x.id]
        return qn <= 512 and rn <= 512
    
    extra_tokenizer_kwargs = dict(
        # max_length=512,
        # padding='max_length'
    )
    
    train_dataset = SpanDatasetForSiamese('data/span/train.jsonl', tokenizer, item_filter, extra_tokenizer_kwargs)
    # train_dataset = CombinedDataset(
    #     SpanDatasetForSiamese('data/span/val.jsonl', tokenizer, item_filter, extra_tokenizer_kwargs),
    #     SpanDatasetForSiamese('data/span/test.jsonl', tokenizer, item_filter, extra_tokenizer_kwargs),
    # )
    # train_dataset = SpanDatasetForSiamese('data/span/full.jsonl', tokenizer, item_filter, extra_tokenizer_kwargs)
    
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=config.num_workers, pin_memory=True, shuffle=True)

    val_dataset = PredictionDatasetForSiamese('data/splitted/val.csv', tokenizer)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=config.num_workers, pin_memory=True)

    trainer = Trainer(
        accelerator='gpu',
        accumulate_grad_batches=config.accumulate_grad_batches,
        logger=TensorBoardLogger(config.save_dir, name=config.name, version=''),
        log_every_n_steps=config.log_every_n_steps,
        callbacks=[
            LearningRateMonitor(),
            ModelCheckpoint(
                monitor='Score/Val',
                mode='max',
                filename='score{Score/Val:.4f}',
                auto_insert_metric_name=False,
            ),
            ModelCheckpoint(
                monitor='step',
                mode='max',
                filename='s{step:06d}',
                auto_insert_metric_name=False,
                every_n_train_steps=config.save_every_n_steps,
            ),
            ModelCheckpoint(
                monitor='epoch',
                mode='max',
                filename='e{epoch:02d}',
                auto_insert_metric_name=False,
                save_on_train_epoch_end=True,
                save_top_k=5,
            ),
        ],
        max_epochs=config.max_epochs,
        val_check_interval=config.val_check_interval
    )
    
    trainer.fit(
        model,
        train_dataloader,
        val_dataloader,
        ckpt_path=config.ckpt_path
    )

if __name__ == '__main__':
    main()
