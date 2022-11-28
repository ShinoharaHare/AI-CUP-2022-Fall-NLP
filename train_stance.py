import os
from dataclasses import dataclass

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from xai import *


@dataclass
class Config:
    name: str
    data: str
    batch_size: int

    max_epochs: int = -1
    val_check_interval: Optional[int] = None
    save_every_n_steps: Optional[int] = None
    num_workers: int = 8
    accumulate_grad_batches: int = 1
    save_dir: str = 'lightning_logs'
    ckpt_path: Optional[str] = None

    def __post_init__(self):
        if self.ckpt_path is None:
            last_path = os.path.join(self.save_dir, self.name, 'checkpoints/last.ckpt')
            self.ckpt_path = last_path if os.path.exists(last_path) else None


def main():
    config = Config(
        name='roberta-stance',
        data='data/splitted_by_id',
        batch_size=32,
        accumulate_grad_batches=4,
        val_check_interval=500,
        save_every_n_steps=100,
        # ckpt_path=''
    )
    
    model = StanceModel()
    tokenizer = model.get_tokenizer()
    datamodule = DataModule(StanceDataset, config.data, tokenizer, batch_size=config.batch_size, num_workers=config.num_workers)

    trainer = Trainer(
        accelerator='gpu',
        accumulate_grad_batches=config.accumulate_grad_batches,
        logger=TensorBoardLogger(config.save_dir, name=config.name, version=''),
        callbacks=[
            ModelCheckpoint(
                monitor='Loss/Val',
                filename='loss{Loss/Val:.3f}',
                auto_insert_metric_name=False,
            ),
            ModelCheckpoint(
                monitor='step',
                mode='max',
                filename='s{step}',
                auto_insert_metric_name=False,
                every_n_train_steps=config.save_every_n_steps,
                save_last=True,
            ),
            # EarlyStopping('Loss/Val')
        ],
        max_epochs=config.max_epochs,
        val_check_interval=config.val_check_interval,
    )
    
    trainer.fit(model, datamodule=datamodule, ckpt_path=config.ckpt_path)
    trainer.test(model, datamodule=datamodule)


if __name__ == '__main__':
    main()
