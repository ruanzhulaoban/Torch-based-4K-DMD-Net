import torch


class down_sampling_block(torch.nn.Module):
    """下采样块：输入 (H, W, C, B) -> 输出 (H/2, W/2, k*C, B)。

    先过 ReLU，再过 3x3、stride=2 的卷积，通道数变为 k*C（k 可调）。
    输入为 (H, W, C, B) 布局，而 nn.Conv2d 要求 (B, C, H, W)，
    故用 permute 前后转置。所有算子均可微。
    """

    def __init__(self, C, k=2):
        super().__init__()
        # padding=1 保证 H 为偶数时输出恰好 H/2
        self.conv = torch.nn.Conv2d(C, int(k * C), kernel_size=3, stride=2, padding=1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # x: (H, W, C, B)
        x = self.relu(x)
        x = x.permute(3, 2, 0, 1)          # -> (B, C, H, W)
        x = self.conv(x)                    # -> (B, k*C, H/2, W/2)
        x = x.permute(2, 3, 1, 0)          # -> (H/2, W/2, k*C, B)
        return x


def _pixel_shuffle_shi(x, upscale_factor=2):
    """Shi et al. 2016 子像素周期重排（通道序与 nn.PixelShuffle 不同）。

    输入 x: (B, C_in, H, W)，C_in = C_out * r^2；输出: (B, C_out, r*H, r*W)。
    映射：PS(x)[h,w,c] = x[h//r, w//r, C_out*r*(w%r) + C_out*(h%r) + c]。
    对比 nn.PixelShuffle 的映射 x[h//r, w//r, c*r^2 + r*(h%r) + (w%r)]，两者通道序不同。
    """
    r = upscale_factor
    B, C_in, H, W = x.shape
    C_out = C_in // (r * r)
    # 通道维 C_in 拆成 (w%r, h%r, c)，与公式 c_in = (w%r)*r*C_out + (h%r)*C_out + c 对齐
    x = x.view(B, r, r, C_out, H, W)      # (B, w%r, h%r, c, H, W)
    x = x.permute(0, 3, 4, 2, 5, 1)       # (B, c, H, h%r, W, w%r)
    x = x.reshape(B, C_out, r * H, r * W)  # (B, C_out, r*H, r*W)
    return x


class up_sampling_block(torch.nn.Module):
    """上采样块（sub-pixel 残差结构）：输入 (H, W, C, B) -> 输出 (2H, 2W, C/2, B)。

    计算流程（2C 为中间通道，输出 C/2）：
      主路 A  = BN -> Conv(3x3, C->2C) -> ReLU -> BN -> Conv(3x3, 2C->2C) -> ReLU
      短路 B  = Conv(3x3, C->2C)
      C     = A + B                                                  # (H, W, 2C)
      D     = BN -> Conv(3x3, 2C->2C) -> ReLU -> BN -> Conv(3x3, 2C->2C) -> ReLU
      E     = PS(D), F = PS(C)                                       # (2H, 2W, C/2)
      输出   = E + F                                                  # (2H, 2W, C/2)

    PS 为 Shi et al. 2016 周期重排（见 _pixel_shuffle_shi）。按题面字面顺序，
    BN 位于 Conv 之前（BN -> Conv -> ReLU）。要求 C 为偶数（输出 C/2）。
    """

    def __init__(self, C):
        super().__init__()
        # 主路 A：BN -> Conv(C->2C) -> ReLU -> BN -> Conv(2C->2C) -> ReLU
        self.bn_a1 = torch.nn.BatchNorm2d(C)
        self.conv_a1 = torch.nn.Conv2d(C, 2 * C, kernel_size=3, stride=1, padding=1)
        self.bn_a2 = torch.nn.BatchNorm2d(2 * C)
        self.conv_a2 = torch.nn.Conv2d(2 * C, 2 * C, kernel_size=3, stride=1, padding=1)
        # 短路 B
        self.conv_b = torch.nn.Conv2d(C, 2 * C, kernel_size=3, stride=1, padding=1)
        # 主路 D：BN -> Conv(2C->2C) -> ReLU -> BN -> Conv(2C->2C) -> ReLU
        self.bn_d1 = torch.nn.BatchNorm2d(2 * C)
        self.conv_d1 = torch.nn.Conv2d(2 * C, 2 * C, kernel_size=3, stride=1, padding=1)
        self.bn_d2 = torch.nn.BatchNorm2d(2 * C)
        self.conv_d2 = torch.nn.Conv2d(2 * C, 2 * C, kernel_size=3, stride=1, padding=1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # x: (H, W, C, B)
        x = x.permute(3, 2, 0, 1)          # -> (B, C, H, W)

        # 主路 A：BN -> Conv -> ReLU -> BN -> Conv -> ReLU
        a = self.conv_a1(self.bn_a1(x))
        a = self.relu(a)
        a = self.conv_a2(self.bn_a2(a))
        a = self.relu(a)

        # 短路 B
        b = self.conv_b(x)

        # 残差求和 C
        c = a + b                          # (B, 2C, H, W)

        # 主路 D：BN -> Conv -> ReLU -> BN -> Conv -> ReLU
        d = self.conv_d1(self.bn_d1(c))
        d = self.relu(d)
        d = self.conv_d2(self.bn_d2(d))
        d = self.relu(d)

        # 子像素重排 E、F，再相加
        e = _pixel_shuffle_shi(d)          # (B, C/2, 2H, 2W)
        f = _pixel_shuffle_shi(c)          # (B, C/2, 2H, 2W)
        out = e + f

        return out.permute(2, 3, 1, 0)     # -> (2H, 2W, C/2, B)


class addition_layer(torch.nn.Module):
    """逐元素相加：两个 (H, W, C, B) 同型张量 -> (H, W, C, B)。

    加法本身可导，无需参数，直接返回 x + y 即可。
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return x + y


class residual_block(torch.nn.Module):
    """标准残差块：两个 3x3 卷积 + ReLU，恒等短路，通道不变。输入/输出 (H, W, C, B)。"""

    def __init__(self, C):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # x: (H, W, C, B)
        y = x.permute(3, 2, 0, 1)          # -> (B, C, H, W)
        out = self.relu(self.conv1(y))
        out = self.conv2(out)
        out = self.relu(out + y)           # 残差短路
        return out.permute(2, 3, 1, 0)     # -> (H, W, C, B)


class tanh_layer(torch.nn.Module):
    """逐元素 tanh，可导。"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.pi * torch.tanh(x)


class sub_pixel_convolution(torch.nn.Module):
    """U-Net 式子像素卷积网络：输入 (H, W, 1, B) -> 输出 (H, W, 1, B)。

    先 3x3/stride2 卷积把 1 通道映射到 8 通道并下采样到 (H/2, W/2)，接残差块；
    再编码器 5 级 k=2 下采样（8 -> 16 -> 32 -> 64 -> 128 -> 256），每级接残差块；
    解码器 6 级上采样（256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4），前 5 级接镜像跳跃连接（加法），
    末级 1x1 卷积把 4 通道压回 1 通道，最后残差块 + tanh。要求 H、W 能被 2^6=64 整除。
    """

    def __init__(self):
        super().__init__()
        # 输入 (H, W, 1, B)：3x3/stride2 卷积把 1 通道映射到 8 通道并下采样到 (H/2, W/2)，
        # 接残差块作为第一级；随后 5 级 k=2 下采样（8 -> 16 -> 32 -> 64 -> 128 -> 256）。
        self.input_conv = torch.nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1)
        self.input_res = residual_block(8)

        self.downs = torch.nn.ModuleList()
        self.res_downs = torch.nn.ModuleList()
        C = 8
        for _ in range(5):
            self.downs.append(down_sampling_block(C, 2))
            C = 2 * C
            self.res_downs.append(residual_block(C))

        # 解码器：6 级上采样，每级通道减半（256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4）
        self.ups = torch.nn.ModuleList()
        for _ in range(6):
            self.ups.append(up_sampling_block(C))
            C = C // 2

        # 末级 4 -> 1 通道对齐输入（1x1 卷积，空间尺寸不变）
        self.final_conv = torch.nn.Conv2d(C, 1, kernel_size=1)

        self.add = addition_layer()
        self.res_out = residual_block(1)   # C == 1
        self.tanh = tanh_layer()

    def forward(self, x):
        # x: (H, W, 1, B)
        x = x.permute(3, 2, 0, 1)          # -> (B, 1, H, W)
        x = self.input_conv(x)             # -> (B, 8, H/2, W/2)
        x = x.permute(2, 3, 1, 0)          # -> (H/2, W/2, 8, B)
        x = self.input_res(x)              # -> (H/2, W/2, 8, B)

        skips = [x]                        # r1 = 8
        for i in range(len(self.downs)):
            x = self.downs[i](x)
            x = self.res_downs[i](x)
            skips.append(x)                # r2..r6

        for i in range(5):                 # 前 5 个 up，镜像加 r5..r1
            x = self.ups[i](x)
            x = self.add(x, skips[4 - i])

        x = self.ups[5](x)                 # 最后一个 up -> 4 通道
        x = x.permute(3, 2, 0, 1)          # -> (B, 4, H, W)
        x = self.final_conv(x)             # -> (B, 1, H, W)
        x = x.permute(2, 3, 1, 0)          # -> (H, W, 1, B)
        x = self.res_out(x)
        x = self.tanh(x)
        return x
