import torch
import torch.nn as nn
import torch.nn.functional as F


class FewshotSideMemory(nn.Module):
    def __init__(self, config):
        super().__init__()
        agent_num = config.get('agent_num', 129)
        feat_dim = config.get('feature_dim', 256)
        tf_nhead = config.get('num_heads', 8)
        dropout = config.get('dropout_ratio', 0.1)

        self.fewshot_detector = FewshotDetector(
            agent_num=agent_num,
            feat_dim=feat_dim,
            tf_nhead=tf_nhead,
            tf_num_layers=2,
            mlp_num_layers=2,
            dropout=dropout
        )
        self.side_memory = SideMemory(
            feat_dim=feat_dim,
            tf_nhead=tf_nhead,
            tf_num_layers=2,
            dropout=dropout
        )

    def forward(self, *inputs):
        feature = inputs[0]
        logits = self.fewshot_detector(feature)
        if self.training:
            flag = inputs[1]['is_fewshot_data']  
        else:
            flag = logits[:, 1] > logits[:, 0]
        feat_out = self.side_memory(feature, flag)
        return logits, feat_out


class FewshotDetector(nn.Module):
    def __init__(self, agent_num, feat_dim, tf_nhead, tf_num_layers=2, mlp_num_layers=2, dropout=0.1):
        super().__init__()
        tf_layer = TransformersModule(
            d_model=feat_dim, 
            nhead=tf_nhead, 
            dropout=dropout
        )
        self.fusion_layers = nn.ModuleList(
            [tf_layer for _ in range(tf_num_layers)]
        )

        self.classifier = MlpModule(
            input_dim=agent_num*feat_dim,
            hidden_dim=512,
            output_dim=2,
            num_layers=mlp_num_layers,
            dropout=dropout
        )
    
    def forward(self, feat):
        output = feat
        for layer in self.fusion_layers:
            output = layer(output)
        logits = self.classifier(output.view(output.size(0), -1))
        return logits


class SideMemory(nn.Module):
    def __init__(self, feat_dim, tf_nhead, tf_num_layers=2, dropout=0.1):
        super().__init__()
        tf_layer = TransformersModule(
            d_model=feat_dim, 
            nhead=tf_nhead, 
            dropout=dropout
        )
        self.fusion_layers = nn.ModuleList(
            [tf_layer for _ in range(tf_num_layers)]
        )
    
    def forward(self, feat, flag):
        output = feat
        true_indices = flag.nonzero(as_tuple=True)[0]

        if true_indices.numel() > 0:
            for layer in self.fusion_layers:
                output[true_indices] = layer(output[true_indices])            
        return output


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


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, preds, labels):
        """
        preds: 模型的输出，未经过 Softmax，形状为 [N, C]
        labels: 标签，形状为 [N]，取值为类别的索引
        """
        eps = 1e-7
        preds_logsoft = F.log_softmax(preds, dim=1)
        preds_softmax = torch.exp(preds_logsoft)
        preds_softmax = preds_softmax.gather(1, labels.unsqueeze(1)).squeeze(1)
        preds_logsoft = preds_logsoft.gather(1, labels.unsqueeze(1)).squeeze(1)

        if self.alpha is not None:
            if isinstance(self.alpha, (list, torch.Tensor)):
                alpha = self.alpha[labels]
            else:
                alpha = self.alpha
        else:
            alpha = 1.0

        loss = -alpha * torch.pow(1 - preds_softmax, self.gamma) * preds_logsoft

        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss.sum()


if __name__ == '__main__':
    data = torch.rand(4, 129, 256)
    label = torch.tensor([True, False, True, False])

    model = FewshotSideMemory({})
    criterion = FocalLoss(alpha=0.25, gamma=2)

    infos = {
        'is_fewshot_data': label
    }

    logits, feat_out = model(data, infos)

    loss = criterion(logits, label.to(torch.int64))
    print(loss)

