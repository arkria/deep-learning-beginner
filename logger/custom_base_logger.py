import logging
from typing import Any, Union

from typing_extensions import override

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback, ModelSummary
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities.model_summary.model_summary import _format_summary_table

import os
import os.path as osp
from collections import defaultdict
import numpy as np


def set_logger(log_dir):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S"
    )

    fh = logging.FileHandler(log_dir)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


class CustomLogger(ModelSummary):
    def __init__(self, log_interval=100, log_file="loss_log.txt", **kwargs):
        super().__init__(**kwargs)
        if osp.exists(log_file):
            os.remove(log_file)
        self.log_interval = log_interval
        self.log_file = log_file
        self.custom_logger = set_logger(log_file)

        self.val_dict = defaultdict(list)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # 每隔固定迭代打印损失并记录到文件
        if batch_idx % self.log_interval == 0:
            loss = outputs['loss'].item()
            log_message = f"[{trainer.current_epoch}/{trainer.max_epochs}, {batch_idx}/{trainer.num_training_batches}], Iter: {trainer.global_step}, Total Loss: {loss:.4f}"
            for k, v in outputs.items():
                if k != 'loss' and k.endswith('loss'):
                    log_message += f", {k}: {v.item():.4f}"
            self.custom_logger.info(log_message)

    
    def on_validation_epoch_start(self, trainer, pl_module):
        self.val_dict.clear()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx = 0):
        for k, v in outputs.items():
            self.val_dict[k].append(v.item())
    
    def on_validation_epoch_end(self, trainer, pl_module):
        # 每隔固定迭代打印损失并记录到文件
        log_message = f"Valid: [{trainer.current_epoch}/{trainer.max_epochs}], Iter: {trainer.global_step}"
        self.custom_logger.info(log_message)
        for k, v in self.val_dict.items():
            self.custom_logger.info(f"{k}: {np.mean(v) / len(v):.4f}")
   
    @override
    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        super().on_fit_start(trainer, pl_module)
        if trainer.train_dataloader is not None:
            self.custom_logger.info(f"training data size: {len(trainer.train_dataloader.dataset)}")
        # if trainer.val_dataloader is not None:
        #     self.custom_logger.info(f"validation data size: {len(trainer.val_dataloader.dataset)}")

    def summarize(
        self,
        summary_data: list[tuple[str, list[str]]],
        total_parameters: int,
        trainable_parameters: int,
        model_size: float,
        total_training_modes: dict[str, int],
        **summarize_kwargs: Any,
    ) -> None:
        summary_table = _format_summary_table(
            total_parameters,
            trainable_parameters,
            model_size,
            total_training_modes,
            *summary_data,
        )
        self.custom_logger.info("\n" + summary_table)