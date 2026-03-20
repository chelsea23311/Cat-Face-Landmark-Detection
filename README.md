#  猫脸关键点检测 (Cat Face Landmark Detection)

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

一个基于 **ResNet-50** 的深度学习项目，用于检测猫脸上的 **9 个关键点**（眼睛、鼻子、耳朵、嘴巴）。该项目包含完整的数据预处理、增强、训练、PCK/NME 评估以及错误案例分析流程。

## 📊 模型性能 (Model Performance)

我们使用两个核心指标来评估模型的关键点定位能力：**PCK (关键点检出率)** 和 **NME (归一化平均误差)**。所有指标均在测试集上计算得出，并且剔除了两眼间距过小 (<10px) 的脏数据。

| 指标 (Metric) | 得分 (Score) | 描述 (Description) |
| :--- | :--- | :--- |
| **PCK @ 0.1** | **76.97%** | 误差小于两眼间距 10% 的关键点比例 (越高越好) |
| **Mean NME** | **16.76%** | 所有点的平均误差占两眼间距的百分比 (越低越好) |
| **Median NME** | **12.71%** | 中位数误差，更能反映绝大多数样本的表现 (越低越好) |

### 📈 训练过程 (Training Process)

我们在训练过程中监控了 Loss 和 PCK 的变化。
*   **Final Train Loss**: 0.0004
*   **Final Val Loss**: 0.0006
*   **Best Val PCK**: 0.7340 (Found new best model)

![Training Results](training_results_4.png)

### 🖼️ 预测结果可视化 (Visualization)

模型在测试集上的随机预测结果展示（绿色为真实点，红色为预测点）：

![Visualization Results](visualization_results_test.png)

---

### 📏 评估指标详解

#### 1. 基础物理距离 ($d_i$)
在计算任何指标之前，首先计算预测点与真实点之间的欧氏距离（像素单位）：
$$d_{i,j} = \sqrt{(x_{i,j} - \hat{x}_{i,j})^2 + (y_{i,j} - \hat{y}_{i,j})^2}$$

#### 2. PCK (Percentage of Correct Keypoints)
PCK 衡量误差在容忍范围内的关键点占总点数的百分比。
$$PCK@\alpha = \frac{\sum_{i=1}^{M} \sum_{j=1}^{9} \mathbb{I}(d_{i,j} < \alpha \cdot L_i)}{M \times 9}$$
*   $\mathbb{I}(\cdot)$：指示函数，误差小于阈值记为 1，否则为 0。
*   $L_i$：归一化因子（第 $i$ 张图的两眼间距）。
*   $\alpha$：容忍度阈值（本项目取 0.2，即 10% 的两眼间距）。

#### 3. NME (Normalized Mean Error)
NME 衡量平均每个关键点偏离真实位置的距离（相对于归一化因子的比例）。
$$NME = \frac{1}{M} \sum_{i=1}^{M} \left( \frac{1}{9} \sum_{j=1}^{9} \frac{d_{i,j}}{L_i} \right)$$
*   分母 $L_i$：两眼间距。
*   该指标反映了模型预测偏离真实的平均程度。

---

### 💾 数据预处理与增强 (Data Preprocessing)

为了适配 ResNet-50 模型输入并增强模型的泛化能力，我们在 `dataset.py` 中实现了完整的数据处理流程：

1.  **尺寸标准化 (Resizing)**:
    *   将所有不同尺寸的原始图片统一缩放到 **224x224** 像素。
    *   这是为了匹配 ResNet-50 预训练模型的标准输入尺寸。

2.  **标签同步缩放 (Label Scaling)**:
    *   关键点坐标 $(x, y)$ 会根据图像宽高的缩放比例进行对应调整，确保坐标位置依然准确。
    *   最终将坐标值归一化到 $[0, 1]$ 区间，加速模型收敛。

3.  **在线数据增强 (On-the-fly Augmentation)**:
    *   **随机旋转**: 在训练时，以 50% 的概率对图像进行 **-30° 到 +30°** 的随机旋转。
    *   关键点坐标也会应用相同的旋转矩阵进行变换，使模型对头部倾斜姿态更加鲁棒。

---

## 🛠️ 主要功能

*   **ResNet-50 主干网络**: 相比 ResNet-18 具有更强的特征提取能力。
*   **自定义回归头**: 将原始的全连接层替换为  Dropout + FC 结构，提升训练稳定性。
*   **评估指标**:
    *   **PCK**: 基于两眼间距的准确率指标。
    *   **NME**: 用于精确测量偏差的归一化平均误差。
    *   **Failure Analysis**: 自动识别并可视化 Top-5 最差预测结果，便于调试。

## 🚀 快速开始

### 1. 安装

克隆仓库并安装依赖：

```bash
git clone https://github.com/chelsea23311/Cat-Face-Landmark-Detection.git
cd Cat-Face-Landmark-Detection
pip install -r requirements.txt
```

### 2. 下载数据和权重

请下载数据集和训练好的模型权重，然后放入项目根目录（或者更新 `dataset.py` / `train.py` 中的路径）。

*   **🐱 猫脸数据集**: [通过网盘分享的文件：cats.zip
链接: https://pan.baidu.com/s/1fr7NDvKNruEYi_kUapW92Q?pwd=kcwi 提取码: kcwi]
    *   *目录结构要求: 包含 `.jpg` 图片和对应 `.txt`标注文件的文件夹（例如 `CAT_00`）。*
*   **⚖️ 模型权重 (`best_cat_model.pth`)**: [https://github.com/chelsea23311/Cat-Face-Landmark-Detection/releases/download/v1.0/best_cat_model.pth]

### 3. 使用方法

#### 训练 (Training)
从头开始训练模型：
```bash
python train.py
```

#### 评估 (Evaluation)
计算测试集的 PCK 和 NME 指标：
```bash
python test_model.py
```

#### 可视化与推理 (Visualization & Inference)
可视化随机验证样本的预测结果：
```bash
python predict.py
```

分析“最难”案例 (Top-5 失败样本)：
```bash
python visualize_failures.py
```

## 📖 探索过程

记录了完整的开发历程，包括架构决策（ResNet-18 vs ResNet-50）、分辨率权衡（224x224 vs 448x448）以及调试复杂的 PyTorch 错误的经验。

👉 **[阅读我的探索日志与实验](./EXPLORATION.md)**

## 📂 项目结构

```
├── dataset.py             # 数据集类、数据划分 (8:1:1) 和增强
├── model.py               # ResNet-50 模型定义
├── train.py               # 包含可视化的主训练循环
├── test_model.py          # 高级评估脚本 (PCK, NME, Median NME)
├── predict.py             # 推理和可视化脚本
├── visualize_failures.py  # 坏例诊断工具
└── requirements.txt       # 依赖列表
```

## 许可证

本项目采用 MIT 许可证。
