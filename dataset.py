import os
import glob
import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split

def rotate_landmarks(landmarks, angle, center=(0.5, 0.5)):
    """ 绕中心点旋转坐标 (归一化坐标系) """
    if angle == 0: return landmarks
    angle_rad = math.radians(-angle) # PIL旋转方向与数学坐标系相反
    ox, oy = center
    
    # 向量化计算旋转: 避免循环，速度更快
    # 旋转矩阵 R = [[cos, -sin], [sin, cos]]
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    
    # 平移到原点
    landmarks_centered = landmarks - np.array([ox, oy])
    
    # 旋转 (x' = x*cos - y*sin, y' = x*sin + y*cos)
    new_landmarks = np.zeros_like(landmarks)
    new_landmarks[:, 0] = landmarks_centered[:, 0] * cos_a - landmarks_centered[:, 1] * sin_a
    new_landmarks[:, 1] = landmarks_centered[:, 0] * sin_a + landmarks_centered[:, 1] * cos_a
    
    # 平移回中心
    return new_landmarks + np.array([ox, oy])

class CatLandmarksDataset(Dataset):
    def __init__(self, data_list, target_size=(224, 224), transform=None, mode='train'):
        self.data_list = data_list
        self.target_size = target_size
        self.transform = transform
        self.mode = mode # 'train', 'val', or 'test'

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_path, txt_path = self.data_list[idx]

        # 1. 加载图片并记录原始尺寸
        img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img.size

        # 2. 读取txt文件
        with open(txt_path, 'r') as f:
            # 读取所有数字（第一个是9，后面18个是坐标）
            content = f.read().split()
            coords = [float(x) for x in content]
        
        # 提取坐标部分 (18个数字)
        landmarks = np.array(coords[1:]).reshape(-1, 2) # 形状变为 (9, 2)

        # 3. 计算缩放比例
        ratio_w = self.target_size[0] / orig_w
        ratio_h = self.target_size[1] / orig_h

        # 4. 调整图片尺寸
        img = img.resize(self.target_size, Image.BILINEAR)

        # 5. 坐标映射到 [0, 224]
        landmarks[:, 0] *= ratio_w
        landmarks[:, 1] *= ratio_h
        
        # 6. 坐标归一化到 [0, 1] (方便旋转计算)
        landmarks[:, 0] /= self.target_size[0]
        landmarks[:, 1] /= self.target_size[1]
        
        # --- 数据增强：随机旋转 ---
        # 仅在训练模式下，且以 50% 概率和随机角度执行
        if self.mode == 'train' and np.random.rand() < 0.5:
            angle = np.random.uniform(-30, 30) # 随机旋转 -30 到 30 度
            # PIL 旋转图片 (默认中心也是 (w/2, h/2))
            img = img.rotate(angle, resample=Image.BILINEAR)
            # 旋转关键点
            landmarks = rotate_landmarks(landmarks, angle)
            
            

        # 转换为 Tensor (将坐标压扁回18个数字)
        landmarks = torch.from_numpy(landmarks.flatten()).float()
        
        if self.transform:
            img = self.transform(img)

        return img, landmarks, img_path, np.array([orig_w, orig_h])

def prepare_data(root_dir):
    all_pairs = []
    # 扫描 CAT_00 到 CAT_05
    for i in range(6):
        folder_name = f"CAT_{i:02d}"
        folder_path = os.path.join(root_dir, folder_name)
        
        if not os.path.exists(folder_path):
            continue
            
        # 查找所有图片（假设后缀是 .jpg）
        img_files = glob.glob(os.path.join(folder_path, "*.jpg"))
        for img_path in img_files:
            target_txt = img_path + ".txt"
            # 只有当对应的 txt 存在时，才配对
            if os.path.exists(target_txt):
                all_pairs.append((img_path, target_txt))
    
    # 划分训练集、验证集、测试集 (8:1:1)
    # 1. 先分出 20% 作为 (验证/测试)
    train_data, temp_data = train_test_split(all_pairs, test_size=0.2, random_state=42)
    # 2. 再将这 20% 对半分为 验证集 和 测试集
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    
    return train_data, val_data, test_data

# --- 使用示例 / 全局加载器配置 ---

root = r"/home/user1/cats/cats"  
if not os.path.exists(root):
    print(f"警告: 数据集路径 {root} 不存在，请在 dataset.py 中修改 root 变量。")

train_list, val_list, test_list = prepare_data(root)

# 定义图像转换 (转换为Tensor并归一化)
from torchvision import transforms
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 实例化 Dataset 和 DataLoader (供 train.py 调用)
# 训练集开启数据增强 (mode='train')
train_dataset = CatLandmarksDataset(train_list, transform=data_transform, mode='train')
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 验证集和测试集不开启数据增强 (mode='val' 或 'test')
val_dataset = CatLandmarksDataset(val_list, transform=data_transform, mode='val')
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

test_dataset = CatLandmarksDataset(test_list, transform=data_transform, mode='test')
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


if __name__ == "__main__":
    print(f"训练集数量: {len(train_list)}")
    print(f"验证集数量: {len(val_list)}")
    print(f"测试集数量: {len(test_list)}")

    if len(train_dataset) > 0:
        # 测试读取一笔数据
        img, label, _, _ = train_dataset[0]
        print(f"图片维度: {img.shape}") # torch.Size([3, 224, 224])
        print(f"坐标维度: {label.shape}") # torch.Size([18])

        data_iter = iter(train_loader)
        batch_imgs, batch_labels, _, _ = next(data_iter)
        print(f"Batch 图片形状: {batch_imgs.shape}") 
        print(f"Batch 标签形状: {batch_labels.shape}")
