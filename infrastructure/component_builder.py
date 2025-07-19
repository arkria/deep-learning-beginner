import torch.utils.data as data
import os.path as osp
from dataset import *
from logger import *
from module import *





def build_dataloader(cfg):
    dataname = cfg.dataset.name
    train, val = globals()[f'{dataname}'](cfg)
    train_loader = data.DataLoader(train, batch_size=cfg.dataset.batch_size, num_workers=cfg.dataset.num_workers)
    val_loader = data.DataLoader(val, batch_size=cfg.dataset.batch_size, num_workers=cfg.dataset.num_workers)
    return train_loader, val_loader

def build_logger(log_dir, configs):
    logger_name = configs.logger.name
    logger = globals()[f'{logger_name}'](
        configs,
        log_file=log_dir,
        log_interval=configs.log_interval,
    )
    return logger

def build_module(cfg):
    model_name = cfg.model.name
    model = globals()[f'{model_name}'](cfg)
    return model