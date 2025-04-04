import torch.utils.data as data
from dataset import *


def build_dataloader(args, configs):
    dataname = configs.dataset.name
    train, val = globals()[f'{dataname}'](args, configs)
    train_loader = data.DataLoader(train, batch_size=args.batch_size, num_workers=args.num_workers)
    val_loader = data.DataLoader(val, batch_size=args.batch_size, num_workers=args.num_workers)
    return train_loader, val_loader