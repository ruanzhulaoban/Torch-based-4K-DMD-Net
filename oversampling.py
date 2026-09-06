import torch


class sin_layer(torch.nn.Module):
    """逐元素 sin：sin_layer(X) = sin(X)。可导。"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)


class cos_layer(torch.nn.Module):
    """逐元素 cos：cos_layer(X) = cos(X)。可导。"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.cos(x)


class zero_padding_2d(torch.nn.Module):
    """2× 零填充：输入 (H, W, C, B) -> 输出 (2H, 2W, C, B)。"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x: (H, W, C, B)
        H, W, C, B = x.shape
        top = H // 2
        bottom = H - top
        left = W // 2
        right = W - left
        out = x.new_zeros((2 * H, 2 * W, C, B))
        out[top:top + H, left:left + W] = x
        return out


class fftshift_layer(torch.nn.Module):
    """二维 fftshift：对 (H, W, C, B) 的前两维（空间维）做 fftshift。
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.fft.fftshift(x, dim=(0, 1))


class fft2_layer(torch.nn.Module):
    """二维 FFT：对 (H, W, C, B) 的前两维做 FFT，返回 (实部, 虚部)。
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        z = torch.fft.fft2(x, dim=(0, 1))
        return z.real, z.imag


class oversampling(torch.nn.Module):
    """过采样模块：相位图 POH (H, W, C, B) -> 衍射强度图 (2H, 2W, C, B)。
    """

    def __init__(self):
        super().__init__()
        self.sin = sin_layer()
        self.cos = cos_layer()
        self.pad = zero_padding_2d()
        self.shift = fftshift_layer()
        self.fft = fft2_layer()
        self.eps = 1e-8                    # 小常数，防止 sqrt 在 0 处梯度发散

    def forward(self, x):
        # x: (H, W, C, B) 相位图
        s = self.sin(x)
        c = self.cos(x)

        s = self.pad(s)                    # (2H, 2W, C, B)
        c = self.pad(c)

        s = self.shift(s)
        c = self.shift(c)

        Rs, Is = self.fft(s)
        Rc, Ic = self.fft(c)

        A = self.shift(Rs + Ic)            # fftshift(Rs + Ic)
        B = self.shift(Rc - Is)            # fftshift(Rc - Is)

        return torch.sqrt(A * A + B * B + self.eps)   # (2H, 2W, C, B) 幅值
