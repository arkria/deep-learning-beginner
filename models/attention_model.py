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


# 示例用法
if __name__ == "__main__":
    embed_dim = 256
    num_heads = 8
    latent_dim = 64
    seq_len = 10
    batch_size = 32

    query = torch.randn(batch_size, seq_len, embed_dim)
    key = torch.randn(batch_size, seq_len, embed_dim)
    value = torch.randn(batch_size, seq_len, embed_dim)

    mha = MultiHeadAttention(embed_dim, num_heads)
    output = mha(query, key, value)
    print("Output shape:", output.shape)  # 应该是 (batch_size, seq_len, embed_dim)
