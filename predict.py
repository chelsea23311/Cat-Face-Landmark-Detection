import torch
import torch.nn as nn
import matplotlib
# matplotlib.use('Agg') # 如果在服务器生成文件不显示，可以取消注释
import matplotlib.pyplot as plt
import numpy as np
import os
from torchvision import transforms



from dataset import CatLandmarksDataset, prepare_data
from model import ResNet50

def visualize_results(model, dataset, device, num_samples=5):
    """
    可视化模型预测结果
    绿色点 (GT): 真实标签
    红色点 (Pred): 模型预测
    """
    model.eval()
    
    # 随机选择索引 (确保不超出数据集范围)
    if len(dataset) < num_samples:
        indices = np.arange(len(dataset))
        num_samples = len(dataset)
    else:
        indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    fig, axes = plt.subplots(1, num_samples, figsize=(4 * num_samples, 5))
    if num_samples == 1: axes = [axes] # 兼容单张图情况
    
    print(f"正在可视化 {num_samples} 张图片...")

    IMG_SIZE = 224
    
    for i, idx in enumerate(indices):
        # 注意：dataset.__getitem__ 现在返回 4 个值 (img, label, path, size)
        img_tensor, label, _, _ = dataset[idx]
        
        # 1. 模型预测
        input_tensor = img_tensor.unsqueeze(0).to(device) # [1, 3, 224, 224]
        with torch.no_grad():
            output = model(input_tensor)
            pred = output.cpu().numpy().flatten().reshape(9, 2)
        
        # 2. 获取真实标签 (GT)
        gt = label.numpy().flatten().reshape(9, 2)
        
        # 3. 反缩放坐标 (从 0-1 还原回 0-224)
        pred = pred * IMG_SIZE
        gt = gt * IMG_SIZE
        
        # 4. 反归一化图片 (Normalize -> 0-1)
        # 这里的 mean/std 必须和 dataset.py 中保持一致
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        img_np = img_tensor.permute(1, 2, 0).numpy() # CHW -> HWC
        img_np = img_np * std + mean 
        img_np = np.clip(img_np, 0, 1) # 限制在 0-1 之间
        
        # 5. 绘图
        axes[i].imshow(img_np)
        axes[i].scatter(gt[:, 0], gt[:, 1], c='g', s=20, marker='o', label='Ground Truth' if i==0 else "")
        axes[i].scatter(pred[:, 0], pred[:, 1], c='r', s=20, marker='x', label='Prediction' if i==0 else "")
        # 设置显示范围
        axes[i].set_xlim(0, IMG_SIZE)
        axes[i].set_ylim(IMG_SIZE, 0)
        axes[i].set_title(f"Sample {idx}")
        axes[i].axis('off')

    # 添加图例 (只显示第一个)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    
    plt.tight_layout()
    save_path = 'visualization_results.png'
    plt.savefig(save_path)
    print(f"可视化结果已保存至: {save_path}")
    # plt.show() # 服务器上通常不需要 show

def main():
    # --- 1. 配置参数 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
   
    MODEL_PATH = '/home/user1/cats/cats/best_cat_model.pth' 
    DATA_ROOT = '/home/user1/cats/cats' 
    
    # 检查路径是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"找不到模型文件: {MODEL_PATH}")
        # 尝试看看当前目录下有没有 (兼容本地测试)
        if os.path.exists('best_cat_model.pth'):
            MODEL_PATH = 'best_cat_model.pth'
            print(f"-> 切换使用当前目录下的模型: {MODEL_PATH}")
        else:
            return

    if not os.path.exists(DATA_ROOT):
        # 尝试本地路径
        local_root = r"C:\Users\15342\Downloads\cats\cats"
        if os.path.exists(local_root):
            DATA_ROOT = local_root
        else:
            print(f"找不到数据集路径: {DATA_ROOT}")
            return

    # --- 2. 加载数据 ---
    print(f"正在加载数据集... ({DATA_ROOT})")
    _, _, test_list = prepare_data(DATA_ROOT)
    
    # 定义必须一致的 transform
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = CatLandmarksDataset(test_list, transform=data_transform)
    print(f"测试集大小: {len(test_dataset)}")

    # --- 3. 加载模型 ---
    print(f"正在加载模型... ({MODEL_PATH})")
    model = ResNet50(num_landmarks=9)
    
    # --- 关键：解决 DataParallel 加载权重的问题 ---
    # 如果训练时使用了 DataParallel (多卡)，state_dict 的 key 会带有 "module." 前缀
    # 如果现在是单卡推理，需要去掉这个前缀；如果是多卡推理，需要保持 DataParallel 结构
    
    state_dict = torch.load(MODEL_PATH, map_location=device)
    
    # 检查是否包含 "module." 前缀
    if list(state_dict.keys())[0].startswith('module.'):
        # 方法 A: 创建一个新的字典，去掉 module.
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # 去掉 `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        # 方法 B: 直接加载
        model.load_state_dict(state_dict)
    
    model = model.to(device)
    
    # --- 4. 执行可视化 ---
    visualize_results(model, test_dataset, device, num_samples=5)

if __name__ == "__main__":
    main()
