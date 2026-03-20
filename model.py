import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ResNet50(nn.Module):
    def __init__(self, num_landmarks=9):
        super(ResNet50, self).__init__()
        # 使用预训练的 ResNet50
        self.backbone = models.resnet50(pretrained=True)
        num_ftrs = self.backbone.fc.in_features
        
        # 移除原有的 FC 层，替换为 Identity (不做任何处理)
        self.backbone.fc = nn.Identity() 

        # 新增的头部结构：Dropout + FC
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(num_ftrs, num_landmarks * 2)

    def forward(self, x):
        x = self.backbone(x) # 输出 (Batch, 2048)
        x = self.dropout(x)  # 正则化
        x = self.fc(x)       # 输出 18 个坐标值
        return torch.sigmoid(x)  # 归一化到 [0, 1] 范围内