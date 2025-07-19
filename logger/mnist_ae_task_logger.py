from .custom_base_logger import CustomLogger
import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt


class MnistAETaskLogger(CustomLogger):
    def __init__(self, configs, log_interval=100, log_file="loss_log.txt", **kwargs):
        super().__init__(configs, log_interval, log_file, **kwargs)
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
        dump_dir = osp.join(self.img_dump_dir, f'E{trainer.current_epoch}-Iter{trainer.global_step}')
        os.makedirs(dump_dir, exist_ok=True)
        pick_idx = np.random.choice(len(self.img_x), 6)        
        fig, axes = plt.subplots(2, 3, figsize=(12, 4))  # figsize 可以调整整个图的大小
        # 遍历每个子图
        for i, (idx, ax) in enumerate(zip(pick_idx, axes.flatten())):  # 将 axes 展平为一维数组
            # 在每个子图上绘制两张灰度图像
            x = self.img_x[idx]
            x_hat = self.img_x_hat[idx]
            ax1 = ax.inset_axes([0, 0, 0.45, 1])  # 左侧子图
            ax2 = ax.inset_axes([0.45, 0, 0.45, 1])  # 右侧子图
            ax1.imshow(x, cmap='gray')
            ax2.imshow(x_hat, cmap='gray')
            # ax.set_title(f'Subplot {idx}')  # 设置子图标题
            ax.axis('off')  # 关闭坐标轴
        plt.tight_layout()
        plt.savefig(osp.join(dump_dir, f'img.png'))
        plt.close()

        super().on_validation_epoch_end(trainer, pl_module)
        