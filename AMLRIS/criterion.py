
import torch.nn as nn
import torch
from torch.nn import functional as F

class HybridSegLoss(nn.Module):
    def __init__(self):
        super(HybridSegLoss, self).__init__()

        self.ce_weight = torch.FloatTensor([0.9, 1.1]).cuda()
        self.ce_loss = nn.CrossEntropyLoss(weight=self.ce_weight)
        

        self.smooth = 1e-5  # 防止除零的小常数

    def dice_loss(self, pred, target):
        """专用Dice Loss计算（仅对前景类别）"""
        pred_foreground = pred[:, 1].sigmoid()  # 只取前景通道[B,H,W]
        target_flat = target.float().view(-1)   # 展平GT
        pred_flat = pred_foreground.view(-1)     # 展平预测
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        return 1 - (2. * intersection + self.smooth) / (union + self.smooth)

    def forward(self, pred, targets):
        '''
        输入:
            pred: [B, 2, H, W] 
            targets['mask']: [B, H, W] (值为0或1)
        输出:
            包含混合损失的字典
        '''
        target = targets['mask']
        
        # 1. 分辨率对齐
        if pred.shape[-2:] != target.shape[-2:]:
            pred = F.interpolate(pred, size=target.shape[-2:], 
                               mode='bilinear', align_corners=True)
        

        ce_loss = self.ce_loss(pred, target)      
        

        total_loss = ce_loss
        return {
            'total_loss': total_loss,
        }

# 更新损失函数字典
criterion_dict = {
    'caris': HybridSegLoss,  # 替换为新的混合损失
}


