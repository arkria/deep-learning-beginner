import math
import os
import time
import torch
import torch.nn as nn
import argparse
import pandas as pd
from torch.utils.data import DataLoader
from collections import defaultdict
import numpy as np
import random
# from sklearn.datasets import make_blobs

from models.vae_mse import VAE
from tools.sys_utils import set_logger


def generate_data(seed, data_num=1000):

    data = np.random.rand(data_num, 129, 256) - 0.5

    labels = np.zeros((data_num,), dtype=int)

    # 为每个数据点分配类别
    for i, data_point in enumerate(data):
        x = np.mean(data_point)
        if np.sin(x) > 0:
            labels[i] = 1
        else:
            labels[i] = 0

    train_size = math.floor(0.8 * data.shape[0])
    train_data = data[:train_size, :]
    valid_data = data[train_size:, :]

    train_label = labels[:train_size]
    valid_label = labels[train_size:]
    print(f"data num {data_num}, train: {len(train_data)}, label {np.sum(train_label)}, valid: {len(valid_data)}, label {np.sum(valid_label)}")
    return train_data, train_label, valid_data, valid_label


def loss_fn(recon_x, x, mean, log_var):
    BCE = torch.nn.functional.mse_loss(
        recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return (BCE + KLD) / x.size(0), BCE / x.size(0), KLD / x.size(0)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, label):
        self.data = data.astype('float32')
        self.label = label.astype(bool)

    def __getitem__(self, item):
        return self.data[item, :], self.label[item]

    def __len__(self):
        return self.data.shape[0]


def train(tr_loader, device, vae, optimizer, epoch, logs, logger, log_dir):
    vae.train()
    tracker_epoch = defaultdict(list)
    logger.info(f"========== train E{epoch} ==========")
    for iteration, (x, y) in enumerate(tr_loader):
        x, y = x.to(device), y.to(device)
        if args.conditional:
            recon_x, mean, log_var, z = vae(x, y)
        else:
            recon_x, mean, log_var, z = vae(x)

        loss, loss_bce, loss_kld = loss_fn(recon_x, x, mean, log_var)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tracker_epoch['dim0'].extend(z.detach().cpu().numpy()[:, 0].tolist())
        tracker_epoch['dim1'].extend(z.detach().cpu().numpy()[:, 1].tolist())
        tracker_epoch['label'].extend(y.detach().cpu().numpy()[:].tolist())

        logs['loss'].append(loss.item())
        logs['loss_bce'].append(loss_bce.item())
        logs['loss_kld'].append(loss_kld.item())

        if iteration % args.print_every == 0 or iteration == len(tr_loader) - 1:
            logger.info("Epoch {:02d}/{:02d} iteration {:04d}/{:d}, Loss {:9.4f}, BCE {:9.4f}, KLD {:9.4f}".format(
                epoch, args.epochs, iteration, len(tr_loader) - 1, loss.item(), loss_bce.item(), loss_kld.item()))

            if args.conditional:
                c = torch.arange(0, 10).long().unsqueeze(1).to(device)
                z = torch.randn([c.size(0), args.latent_size]).to(device)
                x = vae.inference(z, c=c)
            else:
                z = torch.randn([1000, args.latent_size]).to(device)
                x = vae.inference(z)

    df = pd.DataFrame(dict([(key, tracker_epoch[key]) for key in ['dim0', 'dim1', 'label']]))
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.scatter(tracker_epoch['dim0'], tracker_epoch['dim1'], s=50, c=tracker_epoch['label'], alpha=1 / 5)
    plt.title(f'E{epoch}-train-z.png')

    plt.subplot(2, 2, 2)
    df['dim0'].hist(bins=100)
    plt.title(f"E{epoch}-train-dim0")

    plt.subplot(2, 2, 3)
    df['dim1'].hist(bins=100)
    plt.title(f'E{epoch}-train-dim1')
    plt.savefig(os.path.join(log_dir, f"E{epoch}-train.png"))


def valid(tt_loader, device, vae, optimizer, epoch, logs, logger, log_dir, model_dir, checkpoint):
    vae.eval()
    with torch.no_grad():
        tracker_epoch = defaultdict(list)
        logger.info(f"========== valid E{epoch} ==========")
        for iteration, (x, y) in enumerate(tt_loader):
            x, y = x.to(device), y.to(device)
            if args.conditional:
                recon_x, mean, log_var, z = vae(x, y)
            else:
                recon_x, mean, log_var, z = vae(x)
            tracker_epoch['x'].append(x.detach().cpu())
            tracker_epoch['recon_x'].append(recon_x.detach().cpu())
            tracker_epoch['y'].append(y.detach().cpu())
            tracker_epoch['mean'].append(mean.detach().cpu())
            tracker_epoch['log_var'].append(log_var.detach().cpu())

            tracker_epoch['dim0'].extend(z.detach().cpu().numpy()[:, 0].tolist())
            tracker_epoch['dim1'].extend(z.detach().cpu().numpy()[:, 1].tolist())
            tracker_epoch['label'].extend(y.detach().cpu().numpy()[:].tolist())

        x_total = torch.cat(tracker_epoch['x'], dim=0)
        recon_x_total = torch.cat(tracker_epoch['recon_x'], dim=0)
        y_total = torch.cat(tracker_epoch['y'], dim=0)
        mean_total = torch.cat(tracker_epoch['mean'], dim=0)
        log_var_total = torch.cat(tracker_epoch['log_var'], dim=0)

        loss, loss_bce, loss_kld = loss_fn(recon_x_total, x_total, mean_total, log_var_total)
        mse = torch.sum(torch.nn.functional.mse_loss(x_total, recon_x_total, reduction='none'), axis=1)
        mse_max, _ = torch.max(mse, dim=0)
        mse_min, _ = torch.min(mse, dim=0)
        logger.info(f"loss {loss.item()}, BCE {loss_bce.item()}, KLD {loss_kld.item()}, "
                    f"MSE max {mse_max}, min {mse_min}")

        checkpoint['model_state_dict'] = vae.state_dict()
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        torch.save(checkpoint, os.path.join(model_dir, f"epoch_{epoch}.pkl"))

        df = pd.DataFrame(dict([(key, tracker_epoch[key]) for key in ['dim0', 'dim1', 'label']]))
        plt.figure()
        plt.subplot(2, 2, 1)
        plt.scatter(tracker_epoch['dim0'], tracker_epoch['dim1'], s=50, c=tracker_epoch['label'], alpha=1/5)
        plt.title(f'E{epoch}-valid-z.png')

        plt.subplot(2, 2, 2)
        df['dim0'].hist(bins=100)
        plt.title(f"E{epoch}-valid-dim0")

        plt.subplot(2, 2, 3)
        df['dim1'].hist(bins=100)
        plt.title(f'E{epoch}-valid-dim1')

        recon_data = np.concatenate([recon_x_total.numpy()[:, :2], y_total.numpy().reshape(-1, 1)], axis=1)
        plt.subplot(2, 2, 4)
        plt.scatter(recon_data[:, 0], recon_data[:, 1], s=50, c=recon_data[:, 2], alpha=1/5)
        plt.axis('equal')
        plt.title(f'E{epoch}-valid.png')
        plt.savefig(os.path.join(log_dir, f"E{epoch}-valid.png"))


def ood(tt_loader, device, vae, optimizer, epoch, logs, logger, log_dir, model_dir, checkpoint):
    vae.eval()
    with torch.no_grad():
        tracker_epoch = defaultdict(list)
        logger.info(f"========== ood E{epoch} ==========")
        for iteration, (x, y) in enumerate(tt_loader):
            x, y = x.to(device), y.to(device)
            if args.conditional:
                recon_x, mean, log_var, z = vae(x, y)
            else:
                recon_x, mean, log_var, z = vae(x)
            tracker_epoch['x'].append(x.detach().cpu())
            tracker_epoch['recon_x'].append(recon_x.detach().cpu())
            tracker_epoch['y'].append(y.detach().cpu())
            tracker_epoch['mean'].append(mean.detach().cpu())
            tracker_epoch['log_var'].append(log_var.detach().cpu())

            tracker_epoch['dim0'].extend(z.detach().cpu().numpy()[:, 0].tolist())
            tracker_epoch['dim1'].extend(z.detach().cpu().numpy()[:, 1].tolist())
            tracker_epoch['label'].extend(y.detach().cpu().numpy()[:].tolist())

        x_total = torch.cat(tracker_epoch['x'], dim=0)
        recon_x_total = torch.cat(tracker_epoch['recon_x'], dim=0)
        y_total = torch.cat(tracker_epoch['y'], dim=0)
        mean_total = torch.cat(tracker_epoch['mean'], dim=0)
        log_var_total = torch.cat(tracker_epoch['log_var'], dim=0)

        loss, loss_bce, loss_kld = loss_fn(recon_x_total, x_total, mean_total, log_var_total)
        mse = torch.sum(torch.nn.functional.mse_loss(x_total, recon_x_total, reduction='none'), axis=1)
        mse_max, _ = torch.max(mse, dim=0)
        mse_min, _ = torch.min(mse, dim=0)
        logger.info(f"loss {loss.item()}, BCE {loss_bce.item()}, KLD {loss_kld.item()}, "
                    f"MSE max {mse_max}, min {mse_min}")

        df = pd.DataFrame(dict([(key, tracker_epoch[key]) for key in ['dim0', 'dim1', 'label']]))
        plt.figure()
        plt.subplot(2, 2, 1)
        plt.scatter(tracker_epoch['dim0'], tracker_epoch['dim1'], s=50, c=tracker_epoch['label'], alpha=1/5)
        plt.title(f'E{epoch}-ood-z.png')

        plt.subplot(2, 2, 2)
        df['dim0'].hist(bins=100)
        plt.title(f"E{epoch}-ood-dim0")

        plt.subplot(2, 2, 3)
        df['dim1'].hist(bins=100)
        plt.title(f'E{epoch}-ood-dim1')

        recon_data = np.concatenate([recon_x_total.numpy()[:, :2], y_total.numpy().reshape(-1, 1)], axis=1)
        plt.subplot(2, 2, 4)
        plt.scatter(recon_data[:, 0], recon_data[:, 1], s=50, c=recon_data[:, 2], alpha=1/5)
        plt.axis('equal')
        plt.title(f'E{epoch}-ood.png')
        plt.savefig(os.path.join(log_dir, f"E{epoch}-ood.png"))


def main(args):
    # env set
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # generate data
    train_data, train_label, valid_data, valid_label = generate_data(seed)
    train_ds = Dataset(train_data, train_label)
    valid_ds = Dataset(valid_data, valid_label)
    train_loader = DataLoader(dataset=train_ds, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_ds, batch_size=args.batch_size, shuffle=True)

    # logger init
    ts = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    ts = '-'.join((ts, args.post_fix)) if args.post_fix else ts
    if not (os.path.exists(os.path.join(args.log_root))):
        os.mkdir(os.path.join(args.log_root))
    if not (os.path.exists(os.path.join(args.model_root))):
        os.mkdir(os.path.join(args.model_root))

    log_dir = os.path.join(args.log_root, ts)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    model_dir = os.path.join(args.model_root, ts)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    logger = set_logger(os.path.join(log_dir, ts+'.txt'))
    logs = defaultdict(list)

    checkpoint = {'model_state_dict': None,
                  'optimizer_state_dict': None,
                  'epoch': -1}

    # model
    vae = VAE(encoder_layer_sizes=args.encoder_layer_sizes,
              latent_size=args.latent_size,
              decoder_layer_sizes=args.decoder_layer_sizes,
              conditional=args.conditional,
              num_labels=10 if args.conditional else 0).to(device)

    # optimizer = torch.optim.Adam(vae.parameters(), lr=args.learning_rate)

    # # train procedure
    # for epoch in range(args.epochs):
    #     train(tr_loader, device, vae, optimizer, epoch, logs, logger, log_dir)
    #     valid(tt_loader, device, vae, optimizer, epoch, logs, logger, log_dir, model_dir, checkpoint)
    #     ood(od_loader, device, vae, optimizer, epoch, logs, logger, log_dir, model_dir, checkpoint)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=0.001)

    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--model_root", type=str, default='model_dump')
    parser.add_argument("--log_root", type=str, default='log_dump')
    parser.add_argument("--post_fix", type=str, default='')
    args = parser.parse_args()

    main(args)
