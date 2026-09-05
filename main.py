import torch
import numpy as np
from pathlib import Path

from PIL import Image

import sub_pixel_convolution as spc
import oversampling


# 训练 / 验证图片目录（相对项目根目录）
TRAIN_DATA_DIR = 'DIV2K_train_HR'
VALID_DATA_DIR = 'DIV2K_valid_HR'


def loss_function(I_hat, I):
    """NPCC 损失（单通道版）：预测 I_hat 与参考 I 的负余弦相似度（逐 batch 取平均）。

    输入均为 (H, W, 1, B)（单通道）。对空间维 (H, W) 先减均值（逐 batch），再算余弦：

        L = - Σ_i (Î_i - Î̄)(I_i - Ī) / sqrt( Σ_i (Î_i - Î̄)² · Σ_i (I_i - Ī)² )

    其中 i 遍历 H*W 个空间位置，Î̄、Ī 为对应空间均值；最后对 B 取平均。
    对应 MATLAB training_code.m 的 npccLoss：mean 沿 [1 2]，loss = mean(npcc,'all')。
    该损失对整体的均值平移不变（减均值后再归一化）。
    """

    # 去掉单通道维，(H, W, 1, B) -> (H, W, B)
    I_hat = I_hat.squeeze(2)
    I = I.squeeze(2)

    X0 = I_hat - I_hat.mean(dim=(0, 1), keepdim=True)
    Y0 = I - I.mean(dim=(0, 1), keepdim=True)

    numerator = (X0 * Y0).sum(dim=(0, 1))
    denominator = (X0.pow(2).sum(dim=(0, 1))).sqrt() * (Y0.pow(2).sum(dim=(0, 1))).sqrt()

    npcc = -numerator / denominator          # 逐 B
    return npcc.mean()                        # 对 B 取平均


def image_read_layer(img, normalize=True):
    """读取 RGB 图片并转为 (H, W, 3, B) 张量。

    参数:
        img: 文件路径(str/Path)、PIL.Image 或 numpy.ndarray。
             单张输入得到 B=1；传入 list/tuple 时沿 batch 维堆叠得到 B>1（需同尺寸）。
             灰度图（2D）会自动复制为 3 通道。
        normalize: uint8 像素值是否归一化到 [0, 1]（默认 True）。
    返回:
        (H, W, 3, B) 的 float32 张量。
    """

    def _to_array(x):
        if isinstance(x, (str, Path)):
            x = Image.open(x)
        if isinstance(x, Image.Image):
            x = x.convert('RGB')              # 统一为 RGB（处理灰度/RGBA）
        x = np.asarray(x)
        if x.ndim == 2:                       # 灰度 -> 复制为 3 通道
            x = np.stack([x, x, x], axis=-1)
        return x

    if isinstance(img, (list, tuple)):
        arr = np.stack([_to_array(i) for i in img], axis=-1)   # (H, W, 3, B)
    else:
        arr = _to_array(img)[..., None]                        # (H, W, 3, 1)

    t = torch.from_numpy(np.array(arr, copy=True))   # copy 保证可写，避免只读数组告警
    if t.dtype == torch.uint8:
        t = t.float()
        if normalize:
            t = t / 255.0
    else:
        t = t.float()
    return t


def _center_crop(x, out_h, out_w):
    """从 (2H, 2W, C, B) 中裁出中心 (out_h, out_w) 区域，返回 (out_h, out_w, C, B)。"""
    H, W = x.shape[0], x.shape[1]
    top = (H - out_h) // 2
    left = (W - out_w) // 2
    return x[top:top + out_h, left:left + out_w]


def _read_batch(paths, H, W, device):
    """读取一批 RGB 图片并统一 resize 到 (H, W)，返回 (H, W, 3, B)。"""
    batch = []
    for p in paths:
        im = image_read_layer(p)                      # (H0, W0, 3, 1)
        im = im.permute(3, 2, 0, 1)                   # (1, 3, H0, W0)
        im = torch.nn.functional.interpolate(im, size=(H, W), mode='bilinear', align_corners=False)
        batch.append(im)                              # (1, 3, H, W)
    x = torch.cat(batch, dim=0)                       # (B, 3, H, W)
    return x.permute(2, 3, 1, 0).to(device)           # (H, W, 3, B)


def train(data_dir=TRAIN_DATA_DIR, H=256, W=256, batch_size=2, epochs=10, lr=1e-1, device='cuda',
          save_dir='checkpoints'):
    """完整训练流程：读取 DIV2K 图片 -> 拆分 RGB 三通道 -> 独立训练三个单通道模型。

    每张 RGB 图拆成 R/G/B 三个 (H, W, 1, B) 张量，各自独立训练一个 sub_pixel_convolution：
      通道 -> sub_pixel_convolution(POH) -> oversampling(衍射强度) -> 中心裁剪 -> 与原通道算 NPCC
    H、W 需能被 64 整除（网络 6 级下采样）。oversampling 无参数，三个模型共享。
    """
    data_dir = Path(data_dir)
    paths = sorted(data_dir.glob('*.png'))
    if not paths:
        raise FileNotFoundError(f'在 {data_dir} 下未找到 .png 图片')

    device = torch.device(device)

    models = [spc.sub_pixel_convolution().to(device) for _ in range(3)]
    optimizers = [torch.optim.Adam(m.parameters(), lr=lr) for m in models]
    oversample = oversampling.oversampling().to(device)

    num_batches = len(paths) // batch_size
    for epoch in range(epochs):
        for b in range(num_batches):
            batch_paths = paths[b * batch_size:(b + 1) * batch_size]
            x = _read_batch(batch_paths, H, W, device)   # (H, W, 3, B)

            losses = []
            for c in range(3):                            # R/G/B 三通道独立训练
                xc = x[:, :, c:c + 1, :]                  # (H, W, 1, B)
                optimizers[c].zero_grad()
                poh = models[c](xc)                       # (H, W, 1, B) 相位图
                y = oversample(poh)                       # (2H, 2W, 1, B) 衍射强度
                y = _center_crop(y, H, W)                 # (H, W, 1, B)
                loss = loss_function(y, xc)
                loss.backward()
                optimizers[c].step()
                losses.append(loss.item())

            if b % 10 == 0:
                print(f'epoch {epoch:3d}/{epochs}  batch {b:4d}/{num_batches}  '
                      f'loss R/G/B = {losses[0]:.4f}/{losses[1]:.4f}/{losses[2]:.4f}')

    save_models(models, save_dir)
    return models


def save_models(models, save_dir='checkpoints'):
    """保存三个单通道模型的权重，每个通道一个 .pt 文件。"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for c, m in enumerate(models):
        torch.save(m.state_dict(), save_dir / f'model_ch{c}.pt')
    print(f'已保存 {len(models)} 个模型权重到 {save_dir}')


def _load_models(save_dir, device):
    """加载三个单通道模型与共享的 oversampling（均置为 eval）。"""
    models = []
    for c in range(3):
        m = spc.sub_pixel_convolution().to(device)
        m.load_state_dict(torch.load(Path(save_dir) / f'model_ch{c}.pt', map_location=device))
        m.eval()
        models.append(m)
    return models, oversampling.oversampling().to(device)


def _resize_to(x, H, W, device):
    """x: (H0, W0, 3, 1) -> resize 到 (H, W, 3, 1)。"""
    x = x.permute(3, 2, 0, 1)                        # (1, 3, H0, W0)
    x = torch.nn.functional.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
    return x.permute(2, 3, 1, 0).to(device)          # (H, W, 3, 1)


def _predict_rgb(x, models, oversample, H, W):
    """x: (H, W, 3, 1) -> 三通道分别预测并合成归一化的 RGB PIL.Image。"""
    preds = []
    for c in range(3):
        with torch.no_grad():
            poh = models[c](x[:, :, c:c + 1, :])          # (H, W, 1, 1) 相位图
            y = _center_crop(oversample(poh), H, W)       # (H, W, 1, 1) 衍射强度
        preds.append(y)

    rgb = torch.cat(preds, dim=2)                          # (H, W, 3, 1)

    # 逐通道 min-max 归一化到 [0,1]，再转 uint8
    arr = rgb[..., 0].cpu().numpy()                        # (H, W, 3)
    lo = arr.min(axis=(0, 1), keepdims=True)
    hi = arr.max(axis=(0, 1), keepdims=True)
    arr = (arr - lo) / (hi - lo + 1e-8)
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), 'RGB')


def predict(image_path, save_dir='checkpoints', H=256, W=256, device='cpu', out_path=None):
    """读取指定图片 -> 三个模型分别预测 -> 合成预测 RGB 图片。

    参数:
        image_path: 输入 RGB 图片路径（str/Path/PIL/np.ndarray）。
        save_dir: 三个模型权重所在目录（save_models 保存的位置）。
        H, W: 预测分辨率（需能被 64 整除）。
        device: 计算设备。
        out_path: 合成图保存路径（None 则只返回不落盘）。
    返回:
        PIL.Image（RGB，尺寸 H×W）。
    """
    device = torch.device(device)
    x = _resize_to(image_read_layer(image_path), H, W, device)   # (H, W, 3, 1)
    models, oversample = _load_models(save_dir, device)
    img = _predict_rgb(x, models, oversample, H, W)
    if out_path is not None:
        img.save(out_path)
    return img


def predict_all(data_dir=VALID_DATA_DIR, save_dir='checkpoints', H=256, W=256, device='cpu',
                out_dir='predictions'):
    """批处理验证目录下所有图片：逐个用三个模型预测并合成 RGB，保存到 out_dir。

    参数:
        data_dir: 验证图片目录（默认 VALID_DATA_DIR = 'DIV2K_valid_HR'）。
        save_dir: 三个模型权重所在目录。
        out_dir: 合成图保存目录（每张存为 `<原文件名>_pred.png`）。
    """
    data_dir = Path(data_dir)
    paths = sorted(data_dir.glob('*.png'))
    if not paths:
        raise FileNotFoundError(f'在 {data_dir} 下未找到 .png 图片')

    device = torch.device(device)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models, oversample = _load_models(save_dir, device)

    for p in paths:
        x = _resize_to(image_read_layer(p), H, W, device)        # (H, W, 3, 1)
        _predict_rgb(x, models, oversample, H, W).save(out_dir / f'{p.stem}_pred.png')

    print(f'已批处理 {len(paths)} 张验证图，合成结果保存到 {out_dir}')


if __name__ == '__main__':
    train()
   
