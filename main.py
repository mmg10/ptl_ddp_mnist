import os
import shutil

import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import ModelCheckpoint


from model import LitResnet
from dataset import IntelDataModule

DEVICE = "gpu"
EPOCHS = 10
num_cpus = os.cpu_count()
from pytorch_lightning import seed_everything
seed_everything(42, workers=True)

NUM_DEVICES = torch.cuda.device_count()
# WORLD_SIZE = int(os.getenv("WORLD_SIZE", "1"))
# NODE_RANK = int(os.getenv("NODE_RANK", "0"))
def run_training(datamodule):

    tb_logger = loggers.TensorBoardLogger(save_dir='./tensorboard/')
    
    checkpoint_callback = ModelCheckpoint(
                    monitor='val_acc',
                    mode='max',
                    filename='{epoch}-{val_acc:.2f}',
                    save_on_train_epoch_end=True)

    
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator=DEVICE,
        strategy='fsdp_native',
        # devices=NUM_DEVICES,
        num_nodes=2,
        # num_nodes=WORLD_SIZE,
        logger=[tb_logger],
        num_sanity_val_steps=0,
        enable_model_summary=False,
        log_every_n_steps=1,
        callbacks=[checkpoint_callback],
        # fast_dev_run=True
    )
    model_name = 'vit_base_patch16_224'
    module = LitResnet(model_name, 0.02, 'Adam', num_classes=10)
    trainer.fit(module, datamodule)
    
    # if NODE_RANK==0 and trainer.local_rank==0:
    #     print(checkpoint_callback.best_model_path)
    #     print(checkpoint_callback.best_model_score)
    #     print('copying checkpoint')
    #     shutil.copyfile(checkpoint_callback.best_model_path, 'best.ckpt')

    



if __name__ == "__main__":
    datamodule = IntelDataModule(num_workers=num_cpus, batch_size=32)
    datamodule.setup()

    run_training(datamodule)


