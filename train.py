import torch
import numpy as np
from pathlib import Path

from PIL import Image

import sub_pixel_convolution as spc
import oversampling


# 训练 / 验证图片目录
TRAIN_DATA_DIR = 'DIV2K_train_HR'
VALID_DATA_DIR = 'DIV2K_valid_HR'


def loss_function(I_hat, I, euclidean_weight=0):
        # 去掉单通道维，(H, W, 1, B) -> (H, W, B)
    I_hat = I_hat.squeeze(2)
    I = I.squeeze(2)

    # 1) NPCC
    X0 = I_hat - I_hat.mean(dim=(0, 1), keepdim=True)
    Y0 = I - I.mean(dim=(0, 1), keepdim=True)

    numerator = (X0 * Y0).sum(dim=(0, 1))
    denominator = (X0.pow(2).sum(dim=(0, 1))).sqrt() * (Y0.pow(2).sum(dim=(0, 1))).sqrt()

    npcc = -numerator / denominator          
    cos_loss = npcc.mean()                    

    # 2) MSE
    diff = I_hat - I
    mse = diff.pow(2).mean(dim=(0, 1))        
    mse_loss = mse.mean()                     

    return cos_loss + euclidean_weight * mse_loss


def image_read_layer(img, normalize=True):
    #读取 RGB 图片并转为 (H, W, 3, B) 张量。
    def _to_array(x):
        if isinstance(x, (str, Path)):
            x = Image.open(x)
        if isinstance(x, Image.Image):
            x = x.convert('RGB')              # 统一为 RGB
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


def train(data_dir=TRAIN_DATA_DIR, H=256, W=256, batch_size=2, epochs=2, lr=1e-3, device='cuda',
          save_dir='checkpoints', euclidean_weight=0.0):
    """完整训练流程：读取 DIV2K 图片 -> 拆分 RGB 三通道 -> 独立训练三个单通道模型。"""

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
                loss = loss_function(y, xc, euclidean_weight)
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


if __name__ == '__main__':
    train()
   
