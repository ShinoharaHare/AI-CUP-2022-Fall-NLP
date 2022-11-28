import os
from dataclasses import dataclass

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from xai import *

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


@dataclass
class Config:
    name: str
    data: str
    batch_size: int
    save_every_n_steps: int

    max_epochs: Optional[int] = None
    num_workers: int = 8
    accumulate_grad_batches: int = 1
    save_dir: str = 'lightning_logs'
    ckpt_path: Optional[str] = None


def main():
    config = Config(
        name='test',
        data='data/train.csv',
        batch_size=1,
        save_every_n_steps=100,
        max_epochs=5,
        ckpt_path=''
    )
    
    model = TestModel()
    tokenizer = TestModel.get_tokenizer()
    train_dataset = RawDatasetForTraining(config.data, tokenizer)
    train_dataloaders = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=True)

    trainer = Trainer(
        accelerator='gpu',
        accumulate_grad_batches=config.accumulate_grad_batches,
        logger=TensorBoardLogger(config.save_dir, name=config.name, version=''),
        callbacks=[
            ModelCheckpoint(
                monitor='step',
                mode='max',
                filename='s{step}',
                auto_insert_metric_name=False,
                every_n_train_steps=config.save_every_n_steps,
                save_top_k=2,
            ),
            ModelCheckpoint(
                monitor='epoch',
                mode='max',
                filename='e{epoch}',
                auto_insert_metric_name=False,
                save_on_train_epoch_end=True,
                save_top_k=5,
            )
        ],
        max_epochs=config.max_epochs,
        val_check_interval=config.save_every_n_steps
    )
    
    trainer.fit(model, train_dataloaders=train_dataloaders, ckpt_path=config.ckpt_path)
