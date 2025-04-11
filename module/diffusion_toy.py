import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import numpy as np


class DiffusionToy(L.LightningModule):
    def __init__(self, configs):
        super().__init__()
        self.diffusion_alg = configs.model.params.diffusion_alg
        self.lr = configs.optimizer.params.lr
        self.model = SimpleMlp(dim=2)
        self.T = configs.model.params.denoise_steps
        if self.diffusion_alg in ['ddpm', 'ddim']:
            beta_start = 1e-4
            beta_end = 0.02
            self.betas = torch.linspace(beta_start, beta_end, self.T)
            self.alphas = 1. - self.betas
            self.alpha_hat = torch.cumprod(self.alphas, dim=0)
        

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        return embedding
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {"optimizer": optimizer}

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        if self.diffusion_alg == 'flow_matching':
            loss = self.flow_matching_train(batch)
        elif self.diffusion_alg in ['ddpm', 'ddim']:
            loss = self.ddpm_train(batch)
        else:
            raise ValueError(f"Unknown diffusion algorithm: {self.diffusion_alg}")

        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        if self.diffusion_alg == 'flow_matching':
            val_error, trajectory, noise, pred = self.flow_matching_inference(batch)
        elif self.diffusion_alg == 'ddpm':
            val_error, trajectory, noise, pred = self.ddpm_inference(batch)
        elif self.diffusion_alg == 'ddim':
            val_error, trajectory, noise, pred = self.ddim_inference(batch)
        else:  
            raise ValueError(f"Unknown diffusion algorithm: {self.diffusion_alg}")

        self.log_dict({'val_error': val_error})
        trajectory = np.concatenate(np.expand_dims(trajectory, 0), axis=0).transpose(1, 0, 2)
        return {'val_loss': val_error, 
                'noise': [v for v in noise.detach().cpu().numpy()], 
                'target': [v for v in pred.detach().cpu().numpy()],
                'trajectory': [v for v in trajectory]}
    
    def generate_noise(self, x_target):
        return torch.randn_like(x_target)
    
    def flow_matching_train(self, batch):
        x1 = batch
        x0 = self.generate_noise(x1)  # 生成随机噪声
        t = torch.rand(x0.size(0), 1, dtype=torch.float32).to(x1.device)  # 例如：shape (1000, 1)
  
        # 线性插值生成中间点
        xt = (1 - t) * x0 + t * x1
        vt_pred = self.model(xt, t)  # t的维度保持不变
  
        # 目标向量场：x1 - x0
        vt_target = x1 - x0
    
        # 损失函数
        loss = torch.mean((vt_pred - vt_target)**2)
        return loss
    
    def flow_matching_inference(self, batch):
        x1 = batch
        x0 = self.generate_noise(x1)
        t = 0
        delta_t = 1 / self.T
        x = x0.clone()
        trajectory = []
        for i in range(self.T):
            vt = self.model(x, torch.tensor([[t]], dtype=torch.float32).to(x.device).repeat(x.shape[0], 1))  # t的维度保持不变
            t += delta_t
            x = x + vt * delta_t  # x(t+Δt) = x(t) + v(t)Δt
            trajectory.append(x.detach().numpy())
    
        # 损失函数
        val_error = torch.mean(torch.abs(torch.sin(x[:, 0]) - x[:, 1]))
        return val_error, trajectory, x0, x
    
    def ddpm_train(self, batch):
        x0 = batch
        t = torch.randint(0, self.T, (x0.size(0),1), device=x0.device).long()
        noise = self.generate_noise(x0)  # 生成随机噪声
        x_t = self.q_sample(x0, t, noise)
        predicted_noise = self.model(x_t, t)
    
        loss = F.mse_loss(predicted_noise, noise)
        return loss
    
    def ddpm_inference(self, batch):
        x0 = batch
        noise = self.generate_noise(x0)  # 初始噪声点
        x = noise.clone()
        trajectory = []
        for t in reversed(range(self.T)):
            t_batch = torch.tensor([[t]], dtype=torch.float32).to(x.device).repeat(x.shape[0], 1)
            z = torch.randn_like(x) if t > 0 else 0
            predicted_noise = self.model(x, t_batch)
            alpha_t = self.alphas[t].to(x.device)
            alpha_hat_t = self.alpha_hat[t].to(x.device)
            x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_hat_t).sqrt() * predicted_noise) + self.betas[t].sqrt() * z
            trajectory.append(x.detach().numpy())
    
        # 损失函数
        val_error = torch.mean(torch.abs(torch.sin(x[:, 0]) - x[:, 1]))
        return val_error, trajectory, noise, x
    
    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha_hat_t = self.alpha_hat[t].sqrt().to(x0.device)
        sqrt_one_minus_alpha_hat_t = (1 - self.alpha_hat[t]).sqrt().to(x0.device)
        return sqrt_alpha_hat_t * x0 + sqrt_one_minus_alpha_hat_t * noise


class SimpleMlp(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, 64),  # 输入维度: x (2) + t (1) = 3
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, dim)
        )
  
    def forward(self, x, t):
        # 直接拼接x和t（t的形状需为(batch_size, 1)）
        return self.net(torch.cat([x, t], dim=1))

    
