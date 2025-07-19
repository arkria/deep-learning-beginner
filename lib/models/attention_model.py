import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformersModule(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
     
    def forward(self, tgt, tgt_mask=None, tgt_key_padding_mask=None):
        tgt2 = self.norm1(tgt)
        tgt2 = self.self_attn(tgt2, tgt2, tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)

        tgt2 = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt2))))
        tgt = tgt + self.dropout2(tgt2)

        return tgt
    

class MlpModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, head_num, dropout=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embed_dim = embed_dim
        self.head_num = head_num
        self.head_dim = embed_dim // head_num
        assert self.head_dim * head_num == embed_dim, "embed_dim must be divisible by head_num"
        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(dropout)
        self.w_q = nn.Linear(embed_dim, embed_dim)
        self.w_k = nn.Linear(embed_dim, embed_dim)
        self.w_v = nn.Linear(embed_dim, embed_dim)

        self.w_o = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, mask=None):
        q = self.w_q(query)
        k = self.w_k(key)
        v = self.w_v(value)
        q = q.view(q.size(0), -1, self.head_num, self.head_dim).transpose(1, 2)
        k = k.view(k.size(0), -1, self.head_num, self.head_dim).transpose(1, 2)
        v = v.view(v.size(0), -1, self.head_num, self.head_dim).transpose(1, 2)
        attention_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attention_weights = attention_weights.masked_fill(mask == 0, float('-inf'))
        attention_weights = F.softmax(attention_weights, dim=-1)
        attention_weights = self.dropout(attention_weights)
        output = torch.matmul(attention_weights, v)
        output = output.transpose(1, 2).contiguous().view(output.size(0), -1, self.embed_dim)
        output = self.w_o(output)   
        return output


class DeformableAttention(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, num_points=4):
        """
        Args:
            embed_dim: 特征维度
            num_heads: 注意力头数
            num_points: 每个注意力头的采样点数
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_points = num_points
        self.head_dim = embed_dim // num_heads

        # 生成采样偏移和注意力权重的线性层
        self.offset_generator = nn.Linear(embed_dim, num_heads * num_points * 2)  # 2表示xy偏移
        self.attention_weights_generator = nn.Linear(embed_dim, num_heads * num_points)
        
        # 输出投影层
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        
        # 初始化参数
        self._reset_parameters()

    def _reset_parameters(self):
        # 偏移量初始化较小值
        nn.init.constant_(self.offset_generator.weight, 0.)
        nn.init.constant_(self.offset_generator.bias, 0.)
        # 注意力权重初始化
        nn.init.xavier_uniform_(self.attention_weights_generator.weight)
        nn.init.constant_(self.attention_weights_generator.bias, 0.)

    def forward(self, query, reference_points, value, value_spatial_shapes):
        """
        Args:
            query: (B, N, C) 查询向量
            reference_points: (B, N, 2) 参考点坐标（归一化到[0,1]）
            value: (B, C, H, W) 输入特征图
            value_spatial_shapes: 特征图的原始空间尺寸(H, W)
        Returns:
            output: (B, N, C) 注意力输出
        """
        B, N, _ = query.shape
        H, W = value_spatial_shapes
        
        # 生成采样偏移量 (B, N, num_heads*num_points*2)
        offset = self.offset_generator(query)  # [B, N,  num_heads*num_points*2]
        offset = offset.view(B, N, self.num_heads, self.num_points, 2)  # [B, N,  num_heads, num_points, 2]
        
        # 生成注意力权重 (B, N, num_heads*num_points)
        attention_weights = self.attention_weights_generator(query)
        attention_weights = attention_weights.view(B, N, self.num_heads, self.num_points)
        attention_weights = F.softmax(attention_weights, dim=-1)  # 在采样点维度归一化
        
        # 归一化参考点坐标到[-1, 1]范围（grid_sample要求）
        reference_points = reference_points * 2 - 1  # [B, N, 2]
        reference_points = reference_points.unsqueeze(2).unsqueeze(2)  # [B, N, 1, 1, 2]
        
        # 计算采样位置 = 参考点 + 偏移量
        sampling_locations = reference_points + offset  # [B, N,  num_heads, num_points, 2]
        
        # 调整value形状为多头形式 (B*num_heads, C//num_heads, H, W)
        value = value.view(B, self.num_heads, self.head_dim, H, W)  # [B, num_heads, C/num_heads, H, W]
        value = value.permute(0, 1, 3, 4, 2).contiguous().view(B*self.num_heads, H, W, self.head_dim)
        value = value.permute(0, 3, 1, 2)  # [B*num_heads, C/num_heads, H, W]
        
        # 采样特征 (B*H, C/H, N, K)
        sampled_features = F.grid_sample(
            value,
            sampling_locations.view(B*self.num_heads, N*self.num_points, 1, 2),
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )  # [B*H, C/H, N*K, 1]
        
        # 调整形状 (B, H, N, K, C/H)
        sampled_features = sampled_features.view(B, self.num_heads, self.head_dim, N, self.num_points)
        sampled_features = sampled_features.permute(0, 3, 1, 4, 2)  # [B, N, num_heads, num_points, C/H]
        
        # 加权求和
        output = torch.einsum('bnhk,bnhkc->bnhc', attention_weights, sampled_features)
        output = output.reshape(B, N, self.embed_dim)  # [B, N, C]
        
        # 输出投影
        output = self.output_proj(output)
        return output

# 使用示例
if __name__ == "__main__":
    # 输入参数
    B, N, C = 2, 100, 256  # batch_size=2, 100个查询点
    H, W = 32, 32  # 特征图尺寸
    
    # 创建模块
    deform_attn = DeformableAttention(embed_dim=C)
    
    # 生成虚拟输入
    query = torch.randn(B, N, C)
    reference_points = torch.rand(B, N, 2)  # 归一化坐标[0,1]
    value = torch.randn(B, C, H, W)
    
    # 前向传播
    output = deform_attn(query, reference_points, value, (H, W))
    print(output.shape)  # 应该输出 torch.Size([2, 100, 256])


# # 示例用法
# if __name__ == "__main__":
#     embed_dim = 256
#     num_heads = 8
#     latent_dim = 64
#     seq_len = 10
#     batch_size = 32

#     query = torch.randn(batch_size, seq_len, embed_dim)
#     key = torch.randn(batch_size, seq_len, embed_dim)
#     value = torch.randn(batch_size, seq_len, embed_dim)

#     mha = MultiHeadAttention(embed_dim, num_heads)
#     output = mha(query, key, value)
#     print("Output shape:", output.shape)  # 应该是 (batch_size, seq_len, embed_dim)
