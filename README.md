Torch-based 4K-DMD-Net
================================================================================================================

一、项目简介
----------------------------------------------------------------------------------------------------------------
本项目旨在复现论文《4K-DMDNet: diffraction model-driven network for 4K computer-generated holography》的结果。

二、项目背景
----------------------------------------------------------------------------------------------------------------
在此项目之前，GitHub 上已有部分使用 MATLAB 实现的开源代码，但这些代码既不完整，也受 MATLAB 生态限制，使用起来并不方便。
因此，本项目在前人代码的基础上，基于 PyTorch 对论文中的模型进行了完整复现，希望为正在学习全息成像和深度学习技术的学生、开发者提供帮助。

三、功能特性
----------------------------------------------------------------------------------------------------------------
完整实现“训练模型 -> 保存权重 -> 执行预测”的全过程。

四、技术栈
----------------------------------------------------------------------------------------------------------------
- 语言/运行时：Python 3.13
- 深度学习框架：PyTorch 2.11（CUDA 13.0 构建）
- 核心库：NumPy、Pillow、torch.nn / torch.fft / torch.optim
- 物理模型：Fresnel 衍射、角谱 FFT 过采样传播，4K 计算全息（CGH）
- 网络结构：Sub-pixel Convolution U-Net（RGB 三通道独立模型）
- 数据集：DIV2K
- 训练/优化：NPCC 损失（可选 MSE）、Adam（lr=1e-3）

五、快速开始
----------------------------------------------------------------------------------------------------------------

1. 环境要求
- Python 3.13+
- PyTorch 2.11.0（CUDA 13.0 构建，+cu130）
- NVIDIA GPU + CUDA 13.0 驱动（推荐；推理可回退 CPU）
- 内存/显存：单通道模型约 1456 万参数 ×3（RGB），权重每个约 61.5 MB
- 磁盘：DIV2K 数据集 + 权重

2. 安装依赖  

创建虚拟环境（可选但推荐）

python -m venv venv

Windows：

venv\Scripts\activate

Linux/macOS：

source venv/bin/activate

安装 PyTorch（按你的 CUDA 版本替换 +cu130）

pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130

pip install numpy Pillow

3. 准备数据与权重

在项目根目录放置：

DIV2K_train_HR/      # 训练集（800 张 .png）

DIV2K_valid_HR/      # 验证集（100 张 .png）

checkpoints/         # 模型权重：model_ch0.pt / model_ch1.pt / model_ch2.pt

4. 训练

python train.py

5. 推理

python predict.py \
  --data-dir DIV2K_valid_HR \
  --checkpoints checkpoints \
  --out-dir predictions \
  --size 1024 \
  --device cuda

注意：--size 必须是 64 的整数倍（网络 6 级下采样），默认 1024。

六、许可证
----------------------------------------------------------------------------------------------------------------
MIT
