import torch
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
import os
from model import ResNet50

# 根据评估结果填入的 Top 5 失败样本路径
# 如果您是在 Windows 本地运行且路径不同，请在此处修改根目录映射
failure_paths = [
    "/home/user1/cats/cats/CAT_03/00000833_027.jpg",
    "/home/user1/cats/cats/CAT_02/00000619_025.jpg",
    "/home/user1/cats/cats/CAT_01/00000108_005.jpg",
    "/home/user1/cats/cats/CAT_00/00000429_024.jpg",
    "/home/user1/cats/cats/CAT_00/00000437_004.jpg"
]

def process_and_visualize(paths):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 加载模型
    print("Loading model...")
    model = ResNet50(num_landmarks=9)
    model_path = 'best_cat_model.pth'
    
    # 路径兼容性检查
    if not os.path.exists(model_path):
         if os.path.exists('/home/user1/cats/cats/best_cat_model.pth'):
             model_path = '/home/user1/cats/cats/best_cat_model.pth'
    
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        if list(state_dict.keys())[0].startswith('module.'):
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_state_dict[k[7:]] = v
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(state_dict)
    else:
        print("Model not found. Please check path.")
        return

    model.to(device)
    model.eval()

    # 2. 定义预处理 (必须与训练时一致)
    IMG_SIZE = 224
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 3. 绘图准备
    valid_paths = [p for p in paths if os.path.exists(p)]
    num_samples = len(valid_paths)
    if num_samples == 0:
        print("No image files found from the list. Check paths.")
        # 尝试打印出当前目录以帮助调试
        print("Current Working Directory:", os.getcwd())
        return

    fig, axes = plt.subplots(1, num_samples, figsize=(4 * num_samples, 5))
    if num_samples == 1: axes = [axes]

    print(f"Visualizing {num_samples} failure cases...")

    for i, img_path in enumerate(valid_paths):
        ax = axes[i]
        
        # --- 数据加载与处理 ---
        try:
            # A. 图片处理
            img = Image.open(img_path).convert('RGB')
            orig_w, orig_h = img.size
            img_resized = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            
            # 转换为输入 Tensor
            input_tensor = transform(img_resized).unsqueeze(0).to(device)
            
            # B. 标签处理 (GT)
            txt_path = img_path + ".txt"
            gt_landmarks = None
            if os.path.exists(txt_path):
                with open(txt_path, 'r') as f:
                    content = f.read().split()
                    coords = [float(x) for x in content][1:] # 跳过第一个数字9
                    gt_landmarks = np.array(coords).reshape(-1, 2)
                    
                    # 缩放 GT 到 224x224
                    gt_landmarks[:, 0] *= (IMG_SIZE / orig_w)
                    gt_landmarks[:, 1] *= (IMG_SIZE / orig_h)
            
            # --- 模型推理 ---
            with torch.no_grad():
                output = model(input_tensor)
                # 反归一化预测值 (模型输出 0-1)
                pred_landmarks = output.cpu().numpy().flatten().reshape(9, 2)
                pred_landmarks = pred_landmarks * IMG_SIZE
                
            # --- 可视化 ---
            # 为了显示图片，需反归一化 (Tensor -> Image)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            
            disp_img = input_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
            disp_img = disp_img * std + mean
            disp_img = np.clip(disp_img, 0, 1)
            
            ax.imshow(disp_img)
            
            # 画点
            if gt_landmarks is not None:
                # 连线辅助观察 (可选，这里只画点)
                ax.scatter(gt_landmarks[:, 0], gt_landmarks[:, 1], c='lime', s=40, marker='o', label='GT')
                # 标记两眼 (假设前两点是眼睛)
                ax.plot(gt_landmarks[0:2, 0], gt_landmarks[0:2, 1], 'lime', linestyle='--', alpha=0.5)

            ax.scatter(pred_landmarks[:, 0], pred_landmarks[:, 1], c='red', s=40, marker='x', label='Pred')
            
            # 计算这一张图的 NME 显示在标题
            if gt_landmarks is not None:
                L = np.linalg.norm(gt_landmarks[0] - gt_landmarks[1])
                if L > 0:
                    errors = np.linalg.norm(pred_landmarks - gt_landmarks, axis=1)
                    nme = np.mean(errors / L)
                    ax.set_title(f"NME: {nme:.4f}\n{os.path.basename(img_path)}")
                else:
                    ax.set_title(f"Invalid Eye Dist\n{os.path.basename(img_path)}")
            else:
                ax.set_title(f"No GT File\n{os.path.basename(img_path)}")
                
            ax.axis('off')
            if i == 0:
                ax.legend()
                
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    plt.tight_layout()
    save_file = 'failure_analysis_top5.png'
    plt.savefig(save_file)
    print(f"Failure analysis saved to: {save_file}")

if __name__ == "__main__":
    process_and_visualize(failure_paths)
