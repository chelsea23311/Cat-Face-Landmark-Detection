import torch
import matplotlib
matplotlib.use('Agg') # 服务器无界面后端，放在 plt 之前
import matplotlib.pyplot as plt

import torch.nn as nn
from tqdm import tqdm
import numpy as np
import os
torch.manual_seed(123)
from dataset import train_loader, val_loader, test_loader
from model import ResNet50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet50(num_landmarks=9)  
if torch.cuda.device_count() > 1:
    print(f"使用 {torch.cuda.device_count()} 个 GPU")
    model = nn.DataParallel(model)
model = model.to(device)

# 1. 设置初始指标（用于保存最优模型）
best_pck = 0.0 
# 2. 训练超参数
num_epochs = 70
lr = 0.001           # ResNet50 预训练模型建议学习率稍低，这里使用 0.001 或 0.0001
weight_decay = 1e-4  # 权重衰减

# 3. 设置损失函数：选用 Smooth L1 Loss
loss_fn = nn.SmoothL1Loss()

# 4. 设置优化器：
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5, weight_decay=weight_decay)

# 5. 设置学习率调度器：StepLR
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

# 6. PCK 评估函数 (同时计算Loss)
def evaluate_pck(model, loader, device, loss_fn, alpha=0.2): 
    model.eval()
    total_correct = 0
    total_points = 0
    running_val_loss = 0.0
    
    # 定义当前的图像尺寸，用于还原坐标
    IMG_SIZE = 224 

    with torch.no_grad():
        for inputs, labels, _, _ in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            # 1. 计算验证集 Loss (此时 outputs 和 labels 都是 0-1 归一化的)
            loss = loss_fn(outputs, labels)
            running_val_loss += loss.item()
            
            # 2. 定义 preds 和 gts，并调整形状为 (Batch, 9, 2)
            preds = outputs.view(-1, 9, 2)
            gts = labels.view(-1, 9, 2) #真实值
            
            # 3. 遍历 Batch 中的每一张图
            for i in range(preds.size(0)):
                # 还原回 224 像素单位进行物理距离计算
                pred_pixels = preds[i] * IMG_SIZE  
                gt_pixels = gts[i] * IMG_SIZE      

                # 计算参考距离 L：两眼间的像素距离 (点0和点1)
                L = torch.dist(gt_pixels[0], gt_pixels[1]) 
                
                # 如果这张图两眼距离太小（数据异常），跳过不计入统计
                if L < 1e-6: 
                    continue 

                # 计算 9 个预测点与真实点之间的欧氏距离
                dists = torch.norm(pred_pixels - gt_pixels, dim=1)
                
                # 判定：误差像素 < alpha * L 像素
                correct = (dists < (alpha * L)).sum().item()
                total_correct += correct
                total_points += 9
                
    avg_pck = total_correct / total_points if total_points > 0 else 0
    avg_val_loss = running_val_loss / len(loader)
    
    return avg_pck, avg_val_loss

# 记录训练过程中的数据
train_losses = []
val_losses = []
val_pcks = []

print('开始训练...')
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
    for inputs, labels, _, _ in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    scheduler.step()

    # 验证阶段 
    train_loss = running_loss / len(train_loader)
    current_pck, val_loss = evaluate_pck(model, val_loader, device, loss_fn)
    
    # 记录数据
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_pcks.append(current_pck)

    print(f'Epoch {epoch+1} 结束 - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val PCK: {current_pck:.4f}')

    # --- 保存最优模型 ---
    if current_pck > best_pck:
        best_pck = current_pck
        torch.save(model.state_dict(), 'best_cat_model.pth')
        print(f'*** 找到新的最优模型 (Val PCK: {best_pck:.4f}) ***')

print('训练结束 - 最优 PCK:', best_pck)

# --- 绘制训练曲线 ---
plt.figure(figsize=(10, 5))

# 第一张图：Train Loss vs Val Loss
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs+1), val_losses, label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)

# 第二张图：Val PCK
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs+1), val_pcks, label='Val PCK', color='orange')
plt.xlabel('Epochs')
plt.ylabel('PCK')
plt.title('Validation PCK')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_results.png') # 保存图片
print("训练曲线已保存为 training_results.png")


