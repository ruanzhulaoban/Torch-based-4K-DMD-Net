"""预测脚本：加载训练好的三个单通道模型，对 DIV2K_valid_HR 中的图片逐通道预测并合成 RGB 彩色图。

与 main.py 的训练流程保持一致：每张 RGB 图拆成 R/G/B 三个 (H, W, 1, 1) 张量，
各自送入对应的 sub_pixel_convolution 得到相位图，再过 oversampling 得到衍射强度，
中心裁剪回 (H, W) 后逐通道 min-max 归一化，最后合成 RGB 图片保存。

运行方式（项目根目录下）：
    python predict.py

可选参数：
    --data-dir      验证图片目录（默认 DIV2K_valid_HR）
    --checkpoints   模型权重目录（默认 checkpoints，内含 model_ch{0,1,2}.pt）
    --out-dir       合成图输出目录（默认 predictions，每张存为 <原名>_pred.png）
    --size          预测边长（正方形，需能被 64 整除；默认 1024）
    --device        计算设备（默认自动：有 CUDA 用 cuda，否则 cpu）
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import sub_pixel_convolution as spc
import oversampling


def _read_image(path):
    """读取一张图片为 (H, W, 3, 1) 的 float32 张量，像素归一化到 [0, 1]。"""
    im = Image.open(path).convert('RGB')
    arr = np.asarray(im, dtype=np.float32) / 255.0
    return torch.from_numpy(arr)[..., None]     # (H, W, 3) -> (H, W, 3, 1)


def _resize(x, H, W, device):
    """x: (H0, W0, 3, 1) -> 双线性插值到 (H, W, 3, 1)。"""
    x = x.permute(3, 2, 0, 1)                   # -> (1, 3, H0, W0)
    x = torch.nn.functional.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
    return x.permute(2, 3, 1, 0).to(device)     # -> (H, W, 3, 1)


def _center_crop(x, out_h, out_w):
    """从 (2H, 2W, C, B) 裁出中心 (out_h, out_w) 区域，返回 (out_h, out_w, C, B)。"""
    H, W = x.shape[0], x.shape[1]
    top = (H - out_h) // 2
    left = (W - out_w) // 2
    return x[top:top + out_h, left:left + out_w]


def load_models(checkpoint_dir, device):
    """加载三个单通道模型（置为 eval）与共享的 oversampling。"""
    checkpoint_dir = Path(checkpoint_dir)
    models = []
    for c in range(3):
        m = spc.sub_pixel_convolution().to(device)
        m.load_state_dict(torch.load(checkpoint_dir / f'model_ch{c}.pt', map_location=device))
        m.eval()
        models.append(m)
    return models, oversampling.oversampling().to(device)


def predict_rgb(x, models, oversample, H, W):
    """x: (H, W, 3, 1) -> 三通道分别预测并合成归一化的 RGB PIL.Image。"""
    preds = []
    for c in range(3):
        with torch.no_grad():
            poh = models[c](x[:, :, c:c + 1, :])     # (H, W, 1, 1) 相位图
            y = _center_crop(oversample(poh), H, W)   # (H, W, 1, 1) 衍射强度
        preds.append(y)

    rgb = torch.cat(preds, dim=2)[..., 0].cpu().numpy()   # (H, W, 3)

    # 逐通道 min-max 归一化到 [0, 1]，再转 uint8
    lo = rgb.min(axis=(0, 1), keepdims=True)
    hi = rgb.max(axis=(0, 1), keepdims=True)
    rgb = (rgb - lo) / (hi - lo + 1e-8)
    return Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8), 'RGB')


def main():
    parser = argparse.ArgumentParser(description='用训练好的模型预测 DIV2K 验证图并合成 RGB 彩色图')
    parser.add_argument('--data-dir', default='DIV2K_valid_HR', help='验证图片目录')
    parser.add_argument('--checkpoints', default='checkpoints', help='模型权重目录')
    parser.add_argument('--out-dir', default='predictions', help='合成图输出目录')
    parser.add_argument('--size', type=int, default=1024, help='预测边长（正方形，需能被 64 整除）')
    parser.add_argument('--device', default=None, help='计算设备（默认自动检测）')
    args = parser.parse_args()

    if args.size % 64 != 0:
        raise ValueError(f'size 需能被 64 整除（网络 6 级下采样），当前为 {args.size}')

    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))

    data_dir = Path(args.data_dir)
    paths = sorted(data_dir.glob('*.png'))
    if not paths:
        raise FileNotFoundError(f'在 {data_dir} 下未找到 .png 图片')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models, oversample = load_models(args.checkpoints, device)

    H = W = args.size
    for p in paths:
        x = _resize(_read_image(p), H, W, device)          # (H, W, 3, 1)
        img = predict_rgb(x, models, oversample, H, W)
        img.save(out_dir / f'{p.stem}_pred.png')

    print(f'已完成 {len(paths)} 张图片预测，结果保存到 {out_dir}（分辨率 {H}×{W}，设备 {device}）')


if __name__ == '__main__':
    main()
