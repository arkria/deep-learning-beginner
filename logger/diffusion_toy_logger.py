from .custom_base_logger import CustomLogger
import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt


class DiffusionToyLogger(CustomLogger):
    def __init__(self,configs, log_interval=100, log_file="loss_log.txt", **kwargs):
        super().__init__(configs, log_interval, log_file, **kwargs)
        self.noise = []
        self.target = []
        self.trajectory = []
        self.img_dump_dir = osp.join(osp.dirname(log_file), configs.logger.params.img_dump_dir)
        os.makedirs(self.img_dump_dir, exist_ok=True)

    def on_validation_epoch_start(self, trainer, pl_module):
        self.noise = []
        self.target = []
        self.trajectory = []
        super().on_validation_epoch_start(trainer, pl_module)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx = 0):
        noise = outputs.pop('noise')
        target = outputs.pop('target')
        trajectory = outputs.pop('trajectory')
        self.noise.extend(noise)
        self.target.extend(target)
        self.trajectory.extend(trajectory)
        super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
    
    def on_validation_epoch_end(self, trainer, pl_module):
        dump_dir = osp.join(self.img_dump_dir, f'E{trainer.current_epoch}-Iter{trainer.global_step}')
        os.makedirs(dump_dir, exist_ok=True)
        plot_num = min(len(self.trajectory), 5)
        for i in range(plot_num):
            trajectory = self.trajectory[i]
            noise_data = self.noise[i]
            target_data = self.target[i]
            plt.figure(figsize=(10, 5))
            x = np.arange(0, 4 * np.pi, 0.1)
            y = np.sin(x)
            plt.plot(x, y, 'k.', label='baseline (sin(x))')
            plt.scatter(target_data[0], target_data[1], c='blue', label='Target (sin(x))')
            plt.scatter(noise_data[0], noise_data[1], c='red', alpha=0.3, label='Noise')
            plt.plot(trajectory[:,0], trajectory[:,1], 'g-', linewidth=2, label='Generated Path')
            plt.legend()
            plt.title("Flow Matching: From Noise to Target Distribution")
            plt.savefig(osp.join(dump_dir, f'img_{i}.png'))
            plt.close()
        
        plt.figure(figsize=(10, 5))
        x = np.arange(0, 4 * np.pi, 0.1)
        y = np.sin(x)
        plt.plot(x, y, 'k.', label='baseline (sin(x))')
        for i in range(len(self.trajectory)):
            noise_data = self.noise[i]
            target_data = self.target[i]
            plt.scatter(target_data[0], target_data[1], c='blue', label='Target (sin(x))')
            plt.scatter(noise_data[0], noise_data[1], c='red', alpha=0.3, label='Noise')
        plt.title("Total distribution")
        plt.savefig(osp.join(dump_dir, f'Total.png'))
        plt.close()


        super().on_validation_epoch_end(trainer, pl_module)
        