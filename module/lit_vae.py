import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR

from models.vae import VAE


class LitVAE(L.LightningModule):
    def __init__(self, configs):
        super().__init__()
        encoder_layer_sizes = configs.model.params.get('encoder_layer_sizes', [784, 128])
        decoder_layer_sizes = configs.model.params.get('decoder_layer_sizes', [3, 128, 784])
        latent_size = configs.model.params.get('latent_size', 3)
        self.vae = VAE(encoder_layer_sizes=encoder_layer_sizes,
                        latent_size=latent_size,
                        decoder_layer_sizes=decoder_layer_sizes,
                        conditional=False,
                        num_labels=0) # VAE 模型
        
        self.config = configs

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        
        x_hat, *_ = self.vae(x)
        return x_hat
    
    def configure_optimizers(self):
        self.warmup_steps = self.config.optimizer.params.warmup_steps
        self.start_factor = self.config.optimizer.params.start_factor
        self.total_steps = self.config.optimizer.params.total_steps
        self.eta_min = self.config.optimizer.params.eta_min
        lr = self.config.optimizer.params.lr
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        # 定义 Linear Warmup 调度器
        warmup_scheduler = LinearLR(optimizer, start_factor=self.start_factor, total_iters=self.total_steps)

        # 定义 Cosine Annealing 调度器
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=self.total_steps - self.warmup_steps, eta_min=self.eta_min)

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
        x_hat, mean, log_var, z = self.vae(x)
        loss_bce = torch.nn.functional.mse_loss(x_hat, x, reduction='mean')
        loss_kld = -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())
        
        loss = loss_bce + 0.01 * loss_kld
        loss_dict = {'loss': loss, 'bce_loss': loss_bce, 'kld_loss': loss_kld}
        self.log_dict(loss_dict)
        return loss_dict
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        # implement your own
        x_hat = self.forward(x)
        loss = F.mse_loss(x_hat, x)
        self.log_dict({'val_loss': loss})
        return {'val_loss': loss, 
                'x': [v.reshape(28, 28) for v in x.detach().cpu().numpy()], 
                'x_hat': [v.reshape(28, 28) for v in x_hat.detach().cpu().numpy()]}

    

    
