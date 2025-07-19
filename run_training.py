import hydra
from omegaconf import DictConfig, OmegaConf

import os.path as osp

import lightning as L
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor

from infrastructure.component_builder import build_dataloader, build_logger, build_module


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    checkpoint_dir = osp.join(cfg.output_dir, "checkpoints")
    tensorboard_dir = osp.join(cfg.output_dir, "tensorboard")
    log_dir = osp.join(cfg.output_dir, "logs.txt")

    # -------------------
    # Step 1: Define callbacks
    # -------------------

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=cfg.model_save_name,
        every_n_train_steps=cfg.model_save_freq,
        save_top_k=-1
    )

    logger = TensorBoardLogger(save_dir=tensorboard_dir)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    custom_logger = build_logger(log_dir, cfg)

    # -------------------
    # Step 2: Define data
    # -------------------
    train_loader, val_loader = build_dataloader(cfg=cfg)

    # -------------------
    # Step 3: Train
    # -------------------
    model = build_module(cfg)

    trainer = L.Trainer(
        logger=logger,
        callbacks=[
            custom_logger,
            checkpoint_callback,
            lr_monitor
        ],
        enable_progress_bar=False,
        enable_model_summary=False,
        check_val_every_n_epoch=100,
        # val_check_interval=args.valid_freq,
        # max_steps=args.max_steps,
        max_epochs=cfg.max_epochs,
    )
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()