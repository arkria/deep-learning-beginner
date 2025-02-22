from .custom_base_logger import CustomLogger
import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt


class MnistAETaskLogger(CustomLogger):
    def __init__(self, log_interval=100, log_file="loss_log.txt", **kwargs):
        super().__init__(log_interval, log_file, **kwargs)
        self.img_x = []
        self.img_x_hat = []
        self.img_dump_dir = kwargs.get('img_dump_dir', 'output_images')
        os.makedirs(self.img_dump_dir, exist_ok=True)

    def on_validation_epoch_start(self, trainer, pl_module):
        self.img_x = []
        self.img_x_hat = []
        super().on_validation_epoch_start(trainer, pl_module)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx = 0):
        x = outputs.pop('x')
        x_hat = outputs.pop('x_hat')
        self.img_x.extend(x)
        self.img_x_hat.extend(x_hat)
        super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
    
    def on_validation_epoch_end(self, trainer, pl_module):
        dump_dir = osp.join(self.img_dump_dir, f'{trainer.global_step}')
        os.makedirs(dump_dir, exist_ok=True)
        pick_idx = np.random.choice(len(self.img_x), 6)
        for idx in pick_idx:
            x = self.img_x[idx]
            x_hat = self.img_x_hat[idx]
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(x, cmap='gray')
            ax[1].imshow(x_hat, cmap='gray')
            plt.savefig(osp.join(dump_dir, f'{idx}.png'))
            plt.close()

        super().on_validation_epoch_end(trainer, pl_module)
        