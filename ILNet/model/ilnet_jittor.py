import jittor as jt
from jittor import nn
from math import log

# --- 基础组件转换 ---

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        padding = kernel_size // 2 if dilation == 1 else dilation
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def execute(self, x):
        return self.relu(self.bn(self.conv(x)))

class DownConvBNReLU(ConvBNReLU):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, flag: bool = True):
        super().__init__(in_ch, out_ch, kernel_size, dilation)
        self.down_flag = flag

    def execute(self, x):
        if self.down_flag:
            # 显式指定所有参数名，避开位置参数歧义
            x = nn.pool2d(x, kernel_size=2, stride=2, op='max')
        return self.relu(self.bn(self.conv(x)))

class UpConvBNReLU(ConvBNReLU):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, flag: bool = True):
        super().__init__(in_ch, out_ch, kernel_size, dilation)
        self.up_flag = flag

    def execute(self, x):
        if self.up_flag:
            # 获取当前形状并计算 2 倍后的尺寸
            N, C, H, W = x.shape
            target_size = (H * 2, W * 2)
            # 直接传入明确的 size 元组
            x = jt.nn.resize(x, size=target_size, mode='bilinear', align_corners=False)
        return self.relu(self.bn(self.conv(x)))

# --- 核心算子：DODA ---

class DODA(nn.Module):
    def __init__(self, in_ch):
        super(DODA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        
        C = in_ch
        k = int(abs((jt.log2(jt.array([float(C)])) / 2 + 1).item()))
        kernel_size = k if k % 2 else k + 1
        padding = (kernel_size - 1) // 2
        
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def execute(self, x):
        y = self.avg_pool(x) 
        y = self.conv(y.squeeze(-1).transpose(1, 2)).transpose(1, 2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y

# --- 核心算子：IPOF ---

class IPOF(nn.Module):
    def __init__(self, in_ch):
        super(IPOF, self).__init__()
        self.doda = DODA(in_ch)
        self.conv_p = nn.Conv2d(in_ch, in_ch, 1, bias=False)
        self.conv_q = nn.Conv2d(in_ch, in_ch, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    def execute(self, x_shallow, x_deep):
        b, c, h, w = x_shallow.shape
        x_s = self.doda(x_shallow)
        x_d = self.doda(x_deep)
        
        q = self.conv_q(x_d).reshape(b, c, -1)
        p = self.conv_p(x_s).reshape(b, c, -1).transpose(1, 2)
        
        energy = jt.matmul(q, p)
        attention = self.softmax(energy)
        
        out = jt.matmul(attention, x_s.reshape(b, c, -1))
        out = out.reshape(b, c, h, w)
        
        return self.sigmoid(out) * x_deep + x_shallow

# --- RSU 模块 (Residual U-block) ---

class RSU(nn.Module):
    def __init__(self, height, in_ch, mid_ch, out_ch, rsu4f=False):
        super().__init__()
        self.rsu4f = rsu4f
        self.re_block = ConvBNReLU(in_ch, out_ch)
        
        self.down_layers = nn.ModuleList([DownConvBNReLU(out_ch if i==0 else mid_ch, mid_ch) for i in range(height-1)])
        self.bottom_layer = ConvBNReLU(mid_ch, mid_ch, dilation=2)
        self.up_layers = nn.ModuleList([UpConvBNReLU(mid_ch*2, mid_ch if i<height-2 else out_ch) for i in range(height-1)])

    def execute(self, x):
        h = self.re_block(x)
        
        hd = []
        temp = h
        for layer in self.down_layers:
            temp = layer(temp)
            hd.append(temp)
            
        temp = self.bottom_layer(temp)
        
        for i, layer in enumerate(self.up_layers):
            temp = layer(jt.concat([temp, hd[-(i+1)]], dim=1))
            
        return temp + h

# --- 主模型：ILNet ---
class ILNet(nn.Module):
    def __init__(self, cfg, out_ch=1):
        super().__init__()
        self.encoders = nn.ModuleList()
        for c in cfg["encode"]:
            self.encoders.append(RSU(c[0], c[1], c[2], c[3], c[4]))
            
        self.decoders = nn.ModuleList()
        for c in cfg["decode"]:
            self.decoders.append(RSU(c[0], c[1], c[2], c[3], c[4]))
            
        self.final_conv = nn.Conv2d(64, out_ch, 1)
    def execute(self, x):
        # 编码器阶段
        en_feats = []
        temp = x
        for enc in self.encoders:
            temp = enc(temp)
            en_feats.append(temp)
            # 修复此处：显式指定 kernel_size 和 stride
            temp = nn.pool2d(temp, kernel_size=2, stride=2, op='max')
        # 解码器阶段（含跳跃连接）
        de_temp = en_feats[-1]
        for i, dec in enumerate(self.decoders):
            # 动态计算上采样尺寸
            N, C, H, W = de_temp.shape
            target_size = (H * 2, W * 2)
            de_temp = jt.nn.resize(de_temp, size=target_size, mode='bilinear', align_corners=False)
            
            de_temp = dec(jt.concat([de_temp, en_feats[-(i+2)]], dim=1))
        return self.final_conv(de_temp)

def ILNet_S(out_ch: int = 1):
    cfg = {
        "encode": [[7, 3, 16, 64, False], [6, 64, 16, 64, False], [5, 64, 16, 64, False], 
                   [4, 64, 16, 64, False], [4, 64, 16, 64, True], [4, 64, 16, 64, True]],
        "decode": [[4, 128, 16, 64, True], [4, 128, 16, 64, False], [5, 128, 16, 64, False], 
                   [6, 128, 16, 64, False], [7, 128, 16, 64, False]]
    }
    return ILNet(cfg, out_ch)