import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR
from models.attention_model import TransformersModule


class LitAutoEncoder(L.LightningModule):
    def __init__(self, configs):
        super().__init__()
        # self.encoder = nn.Sequential(nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 3))
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            *[TransformersModule(d_model=128, nhead=8, dropout=0.1) for _ in range(2)],
            nn.Linear(128, 3)
        )
        self.decoder = nn.Sequential(nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 28 * 28))
        self.warmup_steps = 1000  # Warmup 的步数
        self.total_steps = 10000  # 总训练步数

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
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        # implement your own
        z = self.forward(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log_dict({'val_loss': loss})
        return {'val_loss': loss, 
                'x': [v.reshape(28, 28) for v in x.detach().cpu().numpy()], 
                'x_hat': [v.reshape(28, 28) for v in x_hat.detach().cpu().numpy()]}

    

    
