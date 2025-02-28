
import lightning as L
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor

import argparse
import os.path as osp

from logger import MnistAETaskLogger as Logger
from module import LitVAE as LitModule
from dataset.mnist_dataset import get_mnist_data as get_dataset
from miscs.config_parser import parse_config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--gpus", type=int, default=0)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--model_save_freq", type=int, default=1000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--valid_freq", type=int, default=1000)

    parser.add_argument("--output_dir", type=str, default='outputs_vae')
    parser.add_argument("--task_config", type=str, default='configs/mnist_ae.yaml')
    parser.add_argument("--@task_mnist_ae.model.name", type=str, default='vae')
    args = parser.parse_args()
    # -------------------
    config = parse_config(args.task_config, args)

    output_dir = args.output_dir
    checkpoint_dir = osp.join(output_dir, "checkpoints")
    tensorboard_dir = osp.join(output_dir, "tensorboard")
    log_dir = osp.join(output_dir, "logs.txt")

    # -------------------
    # Step 1: Define callbacks
    # -------------------

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="autoencoder-{epoch:02d}-{step:04d}-{val_loss:.2f}",
        every_n_train_steps=args.model_save_freq,
        save_top_k=-1
    )

    logger = TensorBoardLogger(save_dir=tensorboard_dir)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    custom_logger = Logger(log_interval=args.log_interval, log_file=log_dir, img_dump_dir=osp.join(output_dir, 'images'))

    # -------------------
    # Step 2: Define data
    # -------------------
    train_loader, val_loader = get_dataset(args)

    # -------------------
    # Step 3: Train
    # -------------------
    model = LitModule(config)
    trainer = L.Trainer(
        logger=logger,
        callbacks=[
            custom_logger,
            checkpoint_callback,
            lr_monitor
        ],
        enable_progress_bar=False,
        enable_model_summary=False,
        check_val_every_n_epoch=1,
        # val_check_interval=args.valid_freq,
        # max_steps=args.max_steps,
        max_epochs=args.max_epochs,
    )
    trainer.fit(model, train_loader, val_loader)