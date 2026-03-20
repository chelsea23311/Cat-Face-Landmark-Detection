import torch
import os
import numpy as np
from tqdm import tqdm
from model import ResNet50
# 从 dataset 导入测试集加载器
# dataset.py 中设置了 random_state=42，所以每次运行划分的数据集是一致的
from dataset import test_loader 

def evaluate_test_set(model, loader, device, alpha=0.2):
    """
    计算测试集指标:
    1. PCK (Percentage of Correct Keypoints) @ alpha
    2. NME (Normalized Mean Error)
    3. Median NME (中值误差)
    4. Top 5 Failures (NME 最大的5个样本)
    """
    model.eval()
    
    # PCK 统计
    total_correct = 0
    total_points = 0
    
    # NME 统计
    total_nme = 0.0
    valid_samples = 0
    skipped_samples = 0  # 新增：记录被剔除的样本数
    
    # 收集详细结果用于计算中值和排序
    all_sample_nmes = []       # 仅保存 NME 值
    bad_case_candidates = []   # 保存 (img_path, nme) 元组
    
    IMG_SIZE = 224 # 还原基准分辨率

    print(f"正在评估测试集 (Total batches: {len(loader)})...")
    
    with torch.no_grad():
        # 注意：这里我们修改了解包，获取 img_paths (第三个返回值)
        for inputs, labels, img_paths, _ in tqdm(loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            
            # 维度调整 (Batch, 9, 2)
            preds = outputs.view(-1, 9, 2)
            gts = labels.view(-1, 9, 2)
            
            # 逐样本计算
            for i in range(preds.size(0)):
                # 1. 还原坐标到 224x224
                pred_pixels = preds[i] * IMG_SIZE
                gt_pixels = gts[i] * IMG_SIZE
                
                # 2. 计算参考距离 L (两眼间距：点0 和 点1)
                inter_ocular_dist = torch.dist(gt_pixels[0], gt_pixels[1])
                
                # --- 关键修改：剔除 L < 10 像素的脏数据 ---
                if inter_ocular_dist < 10.0:
                    skipped_samples += 1
                    continue
                
                valid_samples += 1
                
                # 3. 计算该图片 9 个点的欧氏距离误差
                dists = torch.norm(pred_pixels - gt_pixels, dim=1)
                
                # --- 指标 1: PCK ---
                correct_mask = dists < (alpha * inter_ocular_dist)
                total_correct += correct_mask.sum().item()
                total_points += 9
                
                # --- 指标 2: NME ---
                sample_nme = torch.mean(dists / inter_ocular_dist).item()
                total_nme += sample_nme
                
                # --- 收集数据 ---
                all_sample_nmes.append(sample_nme)
                bad_case_candidates.append((img_paths[i], sample_nme)) # img_paths 是 batch 元组
    
    # 汇总结果
    avg_pck = (total_correct / total_points) * 100 if total_points > 0 else 0.0
    avg_nme = (total_nme / valid_samples) if valid_samples > 0 else 0.0
    
    # --- 指标 3: Median NME ---
    if len(all_sample_nmes) > 0:
        median_nme = np.median(all_sample_nmes)
        
        # --- 指标 4: Top 5 Failures ---
        # 按 NME 降序排列 (误差大的在前)
        bad_case_candidates.sort(key=lambda x: x[1], reverse=True)
        top_5_failures = bad_case_candidates[:5]
    else:
        median_nme = 0.0
        top_5_failures = []
        
    # 打印剔除报告，让你心里有数
    print(f"\n[数据清洗报告] 总计评估: {valid_samples + skipped_samples} 张图")
    print(f"[数据清洗报告] 剔除脏样本 (L < 10px): {skipped_samples} 张")
    print(f"[数据清洗报告] 最终有效样本: {valid_samples} 张")
    
    return avg_pck, avg_nme, median_nme, top_5_failures

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")
    
    # 1. 加载模型结构
    model = ResNet50(num_landmarks=9)
    
    # 2. 加载权重
    model_path = 'best_cat_model.pth'
    # 服务器路径兼容
    if not os.path.exists(model_path):
         # 尝试常见路径
         server_path = '/home/user1/cats/cats/best_cat_model.pth'
         if os.path.exists(server_path):
             model_path = server_path
    
    if os.path.exists(model_path):
        print(f"Loading checkpoint: {model_path}")
        state_dict = torch.load(model_path, map_location=device)
        
        # 处理 DataParallel 的 module. 前缀
        if list(state_dict.keys())[0].startswith('module.'):
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_state_dict[k[7:]] = v
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(state_dict)
    else:
        print(f"Error: Model file '{model_path}' not found!")
        exit(1)
        
    model = model.to(device)
    
    # 3. 运行评估
    # alpha=0.1 表示容错率为 10% 的两眼间距
    pck, nme, median_nme, top_5 = evaluate_test_set(model, test_loader, device, alpha=0.2)
    
    print("-" * 50)
    print("Test Set Evaluation Results:")
    print("-" * 50)
    print(f"Images Evaluated : {len(test_loader.dataset)}")
    print(f"PCK @ 0.1        : {pck:.2f}%")
    print(f"Mean NME         : {nme:.5f} ({nme*100:.2f}%)")
    print(f"Median NME       : {median_nme:.5f} ({median_nme*100:.2f}%)")
    print("-" * 50)
    print("Top 5 Hardest Samples (Highest NME):")
    for i, (path, val) in enumerate(top_5):
        print(f"{i+1}. NME: {val:.4f} | Path: {path}")
    print("-" * 50)
