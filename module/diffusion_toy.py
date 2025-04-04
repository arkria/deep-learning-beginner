import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import numpy as np
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR
from models.attention_model import TransformersModule


class DiffusionToy(L.LightningModule):
    def __init__(self, configs):
        super().__init__()
        self.model = VectorField(dim=2)
        self.warmup_steps = 1000  # Warmup 的步数
        self.total_steps = 10000  # 总训练步数
        self.num_steps = 50

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        return embedding
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        # 定义 Linear Warmup 调度器
        warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=self.warmup_steps)

        # 定义 Cosine Annealing 调度器
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=self.total_steps - self.warmup_steps, eta_min=1e-6)

        # 组合调度器
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[self.warmup_steps]
            ),
            "interval": "step",  # 每个 step 调整学习率
            "frequency": 1
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x1, x0 = batch
        # x0 = torch.randn(x1.size(0), x1.size(1), dtype=torch.float32).to(x1.device)  # 生成随机噪声
        t = torch.rand(x0.size(0), 1, dtype=torch.float32).to(x1.device)  # 例如：shape (1000, 1)
  
        # 线性插值生成中间点
        xt = (1 - t) * x0 + t * x1
        vt_pred = self.model(xt, t)  # t的维度保持不变
  
        # 目标向量场：x1 - x0
        vt_target = x1 - x0
    
        # 损失函数
        loss = torch.mean((vt_pred - vt_target)**2)

        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x1 = batch
        x0 = torch.randn(x1.size(0), x1.size(1)) * 2
        t = 0
        delta_t = 1 / self.num_steps
        x = x0.clone()
        trajectory = []
        for i in range(self.num_steps):
            vt = self.model(x, torch.tensor([[t]], dtype=torch.float32).to(x.device).repeat(x.shape[0], 1))  # t的维度保持不变
            t += delta_t
            x = x + vt * delta_t  # x(t+Δt) = x(t) + v(t)Δt
            trajectory.append(x.detach().numpy())
    
        # 损失函数
        val_error = torch.mean(torch.sin(x[:, 0]) - x[:, 1])
        self.log_dict({'val_error': val_error})
        trajectory = np.concatenate(np.expand_dims(trajectory, 0), axis=0).transpose(1, 0, 2)
        return {'val_loss': val_error, 
                'noise': [v for v in x0.detach().cpu().numpy()], 
                'target': [v for v in x.detach().cpu().numpy()],
                'trajectory': [v for v in trajectory]}


class VectorField(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, 64),  # 输入维度: x (2) + t (1) = 3
            nn.ReLU(),
            nn.Linear(64, dim)
        )
  
    def forward(self, x, t):
        # 直接拼接x和t（t的形状需为(batch_size, 1)）
        return self.net(torch.cat([x, t], dim=1))

    
