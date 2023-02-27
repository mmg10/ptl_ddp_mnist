import os
import shutil

import pytorch_lightning as pl
from pytorch_lightning import loggers


from model import LitResnet
from dataset import IntelDataModule

DEVICE = "gpu"
EPOCHS = 1
num_cpus = os.cpu_count()



def run_training(datamodule):

    tb_logger = loggers.TensorBoardLogger(save_dir='./tensorboard/')
    
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator=DEVICE,
        strategy='ddp_find_unused_parameters_false',
        devices=1,
        num_nodes=2,
        logger=[tb_logger],
        num_sanity_val_steps=0,
        enable_model_summary=False,
        enable_checkpointing=False,
        log_every_n_steps=1,
        # fast_dev_run=True
    )
    module = LitResnet(0.02, 'Adam', num_classes=10)
    trainer.fit(module, datamodule)



if __name__ == "__main__":
    datamodule = IntelDataModule(num_workers=num_cpus, batch_size=32)
    datamodule.setup()

    run_training(datamodule)


