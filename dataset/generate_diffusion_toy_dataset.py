import torch

def get_diffusion_toy_data(args, configs):
    """
    生成一个简单的正弦波数据集
    :param args: 参数
    :param configs: 配置文件
    :return: 训练集和验证集
    """
    num_samples = 1000
    # 生成正弦波数据

    x1_samples = torch.rand(num_samples, 1) * 4 * torch.pi  # 0到4π
    y1_samples = torch.sin(x1_samples)                      # y=sin(x)
    target_data = torch.cat([x1_samples, y1_samples], dim=1)
    return target_data, target_data