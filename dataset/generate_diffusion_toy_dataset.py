import torch
import torch.utils.data as data
import numpy as np

class DiffusionToyDataset(data.Dataset):
    def __init__(self, target_data, noise_data=None):
        super().__init__()
        self.target_data = target_data
        self.noise_data = noise_data
    
    def __len__(self):
        return len(self.target_data)
    
    def __getitem__(self, idx):
        if self.noise_data is not None:
            return self.target_data[idx], self.noise_data[idx]
        else:
            return self.target_data[idx]
    

def get_diffusion_toy_data(args, configs):
    """
    生成一个简单的正弦波数据集
    :param args: 参数
    :param configs: 配置文件
    :return: 训练集和验证集
    """
    num_samples = 10000
    dim = 2

    x1_samples = np.random.rand(num_samples, 1) * 4 * np.pi  # 0到4π
    y1_samples = np.sin(x1_samples)                  # y=sin(x)
    target_data = np.concatenate([x1_samples, y1_samples], axis=1).astype(np.float32)
    train_data = DiffusionToyDataset(target_data)
    val_data = DiffusionToyDataset(target_data[:100])
    return train_data, val_data