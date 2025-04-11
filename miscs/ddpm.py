import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 超参数
T = 1000  # Diffusion steps
beta_start = 1e-4
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, T)

alphas = 1. - betas
alpha_hat = torch.cumprod(alphas, dim=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 正弦分布数据
def generate_sine_data(n_samples=10000, n_points=100):
    x = np.linspace(0, 2*np.pi, n_points)
    data = np.sin(x) + 0.1 * np.random.randn(n_samples, n_points)
    return torch.tensor(data, dtype=torch.float32)

# 网络结构（简单的MLP）
class SimpleMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, dim)
        )

    def forward(self, x, t):
        t_embed = t.unsqueeze(1).float() / T
        t_embed = t_embed.repeat(1, x.shape[1])
        x = torch.cat([x, t_embed], dim=1)
        return self.net(x)

# 前向加噪
def q_sample(x0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x0)
    sqrt_alpha_hat_t = alpha_hat[t].sqrt().unsqueeze(1).to(device)
    sqrt_one_minus_alpha_hat_t = (1 - alpha_hat[t]).sqrt().unsqueeze(1).to(device)
    return sqrt_alpha_hat_t * x0 + sqrt_one_minus_alpha_hat_t * noise

# 训练
def train(model, data, epochs=100, batch_size=128, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):
        idxs = torch.randperm(data.size(0))
        total_loss = 0
        for i in range(0, len(data), batch_size):
            batch = data[idxs[i:i+batch_size]].to(device)
            t = torch.randint(0, T, (batch.size(0),), device=device).long()
            noise = torch.randn_like(batch)
            x_t = q_sample(batch, t, noise)
            predicted_noise = model(x_t, t)

            loss = F.mse_loss(predicted_noise, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# 采样
@torch.no_grad()
def sample(model, n_samples=16, n_points=100):
    model.eval()
    x = torch.randn(n_samples, n_points).to(device)

    for t in reversed(range(T)):
        t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)
        z = torch.randn_like(x) if t > 0 else 0
        alpha_t = alphas[t].to(device)
        alpha_hat_t = alpha_hat[t].to(device)

        predicted_noise = model(x, t_batch)
        x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_hat_t).sqrt() * predicted_noise) + betas[t].sqrt() * z

    return x.cpu()

# 主程序
if __name__ == "__main__":
    n_points = 100
    data = generate_sine_data(n_samples=5000, n_points=n_points)
    model = SimpleMLP(n_points).to(device)

    train(model, data, epochs=50)

    samples = sample(model, n_samples=10, n_points=n_points)

    # 绘图
    x_axis = np.linspace(0, 2 * np.pi, n_points)
    for i in range(samples.shape[0]):
        plt.plot(x_axis, samples[i], label=f"Sample {i+1}")
    plt.title("Generated Sine Waves")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    plt.show()
