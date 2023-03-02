import os
import shutil

import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import ModelCheckpoint


from model import LitResnet
from dataset import IntelDataModule

DEVICE = "gpu"
num_cpus = os.cpu_count()


def run_training(datamodule):

    model_name = 'vit_base_patch16_224'
    module = LitResnet(model_name, 0.02, 'Adam', num_classes=10)
    module.model = torch.load('model.pt')
    
    trainer = pl.Trainer(
        accelerator=DEVICE,
        devices=1,
        num_nodes=1,
        enable_model_summary=False,
    )
    trainer.test(module, datamodule, ckpt_path="best.ckpt")


if __name__ == "__main__":
    datamodule = IntelDataModule(num_workers=num_cpus, batch_size=32)
    datamodule.setup()

    run_training(datamodule)


