# main.py
# ! pip install torchvision
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision as tv
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

import argparse
import os
import os.path as osp

from tools.sys_utils import set_logger


# --------------------------------
# Step 1: Define a LightningModule
# --------------------------------
# A LightningModule (nn.Module subclass) defines a full *system*
# (ie: an LLM, diffusion model, autoencoder, or simple image classifier).

class CustomLossLogger(Callback):
    def __init__(self, log_interval=100, log_file="loss_log.txt"):
        self.log_interval = log_interval
        self.log_file = log_file
        self.custom_logger = set_logger(log_file)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # 每隔固定迭代打印损失并记录到文件
        if batch_idx % self.log_interval == 0:
            loss = outputs['loss'].item()
            log_message = f"Epoch: {trainer.current_epoch}, [{batch_idx}/{trainer.num_training_batches}], Iteration: {trainer.global_step}, Loss: {loss:.4f}"
            self.custom_logger.info(log_message)
            


class LitAutoEncoder(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 3))
        self.decoder = nn.Sequential(nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 28 * 28))

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--gpus", type=int, default=0)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--dump_model_every_n_steps", type=int, default=1000)
    args = parser.parse_args()
    # -------------------
    output_dir = "outputs"
    checkpoint_dir = osp.join(output_dir, "checkpoints")
    tensorboard_dir = osp.join(output_dir, "tensorboard")
    log_dir = osp.join(output_dir, "logs")

    # -------------------
    # Step 2: Define data
    # -------------------
    dataset = tv.datasets.MNIST(".", download=True, transform=tv.transforms.ToTensor())
    train, val = data.random_split(dataset, [55000, 5000])

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="autoencoder-{epoch:02d}-{step:04d}-{val_loss:.2f}",
        every_n_train_steps=args.dump_model_every_n_steps,
        save_top_k=-1
    )

    logger = TensorBoardLogger(save_dir=tensorboard_dir)

    custom_logger = CustomLossLogger(log_interval=100, log_file=log_dir)

    # -------------------
    # Step 3: Train
    # -------------------
    autoencoder = LitAutoEncoder()
    trainer = L.Trainer(
        logger=logger,
        callbacks=[
            custom_logger,
            checkpoint_callback
        ],
        enable_progress_bar=False
    )
    trainer.fit(autoencoder, data.DataLoader(train, batch_size=args.batch_size), data.DataLoader(val))