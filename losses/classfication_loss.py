import torch


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