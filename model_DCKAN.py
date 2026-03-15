import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torchvision.ops import DeformConv2d
from sklearn.neighbors import kneighbors_graph

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import grid

import torch
import torch.nn as nn
import torch.nn.functional as F


# class InvertedResidual(nn.Module):
#     def __init__(self, in_channels, out_channels, att, stride=1):
#         super(InvertedResidual, self).__init__()
#         self.stride = stride
#
#         # 替换DeformConv2d为图卷积层
#         self.conv1 = GraphConv2d(in_channels, in_channels, k=3)
#         self.conv2 = GraphConv2d(in_channels, in_channels, k=3, stride=stride)
#         self.conv3 = GraphConv2d(in_channels, out_channels, k=3)
#
#         # 注意力模块
#         self.conv_att1 = GraphConv2d(in_channels, in_channels // att, k=1)
#         self.conv_att2 = GraphConv2d(in_channels // att, in_channels, k=1)
#
#     def forward(self, x):
#         identity = x
#
#         out1 = F.relu(self.conv1(x))
#
#         att1 = F.relu(self.conv_att1(x))
#         attention_weights = F.sigmoid(self.conv_att2(att1))
#         out_att = out1 * attention_weights
#
#         out2 = self.conv2(out_att)
#         out2 = out2 + out_att
#         out3 = self.conv3(out2)
#
#         if self.stride == 1 and identity.size() == out3.size():
#             out3 += identity
#         return out3
#
#
# class GraphConv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, k=3, stride=1):
#         super(GraphConv2d, self).__init__()
#         self.k = k
#         self.stride = stride
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
#
#     def forward(self, x):
#         # 构建图结构并进行图卷积
#         if self.k > 1:
#             x = self.build_graph(x)
#         return self.conv(x)
#
#     def build_graph(self, x):
#
#         N, C, H, W = x.shape
#
#         # 使用unfold构建局部邻域图
#         if self.k > 1:
#             x_unfold = F.unfold(x, kernel_size=self.k, padding=self.k // 2, stride=self.stride)
#             x_unfold = x_unfold.view(N, C, self.k * self.k, -1)
#
#             # 平均池化模拟图卷积的邻域聚合
#             x = x_unfold.mean(dim=2)
#
#         # 恢复空间维度
#         if self.stride > 1:
#             H, W = H // self.stride, W // self.stride
#         x = x.view(N, C, H, W)
#
#         return x
class ChannelSparseAttention(nn.Module):
    def __init__(self, dim, reduction=8):
        super().__init__()
        self.conv_q = nn.Conv2d(dim, dim // reduction, 1)
        self.conv_k = nn.Conv2d(dim, dim // reduction, 1)
        self.conv_v = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.conv_q(x).reshape(B, -1, H * W).permute(0, 2, 1)  # [B, HW, C/r]
        k = self.conv_k(x).reshape(B, -1, H * W)  # [B, C/r, HW]
        v = self.conv_v(x).reshape(B, -1, H * W)  # [B, C, HW]

        attn = torch.bmm(q, k)  # [B, HW, HW]
        attn = F.softmax(attn, dim=-1)

        out = torch.bmm(v, attn.permute(0, 2, 1))  # [B, C, HW]
        return  out.reshape(B, C, H, W)


class SpatialSparseAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, 1, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        attn = self.conv(x)  # [B, 1, H, W]
        attn = F.softmax(attn.reshape(B, -1), dim=-1).view(B, 1, H, W)

        return  x * attn.expand_as(x)

class GlobalAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [B, N, C]

        qkv = self.qkv(x).reshape(B, -1, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # [3, B, h, N, d]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
        x = self.proj(x).permute(0, 3, 1, 2)  # [B, C, H, W]
        return x


class DepthwiseGraphConv(nn.Module):


    def __init__(self, channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # 定义可学习的邻接权重矩阵
        self.adj_conv = nn.Conv2d(channels, channels * kernel_size ** 2,
                                  kernel_size=kernel_size, stride=stride,
                                  padding=padding, groups=channels)
        nn.init.kaiming_uniform_(self.adj_conv.weight, a=math.sqrt(5))

    def forward(self, x, adj):
        B, C, H, W = x.shape

        H_out = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
        L = H_out * W_out

        # 展开输入特征图
        x_unfold = F.unfold(x, self.kernel_size, padding=self.padding, stride=self.stride)
        x_unfold = x_unfold.view(B, C, self.kernel_size ** 2, L)

        # 将邻接矩阵的空间维度展平，形状：(B, C_adj, k*k, L)
        adj = adj.view(B, -1, self.kernel_size ** 2, L)

        # 加权求和并恢复形状
        out = (x_unfold * adj).sum(dim=2)
        return out.view(B, C, H_out, W_out)


class PointwiseGraphConv(nn.Module):
    """逐点图卷积（等价于1x1卷积）"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.conv(x)



#深度图卷积
class MultiStrategyAttention(nn.Module):
    # 保持原结构不变，仅修改norm的维度调整方式
    def __init__(self, dim, att_reduction=2, num_heads=2):
        super().__init__()
        self.csa = ChannelSparseAttention(dim, att_reduction)
        self.ssa = SpatialSparseAttention(dim)
        self.ga = GlobalAttention(dim, num_heads)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x):
        identity = x
        # # SSA阶段
        # x = self.ssa(x)
        # x = self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # CSA阶段
        x = self.csa(x)
        x = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # SSA阶段
        x = self.ssa(x) + identity
        x = self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # GA阶段
        x = self.ga(x) + x
        x = self.norm3(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)



        # # SSA阶段
        # x = self.ssa(x)
        # x = self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # # GA阶段
        # x = self.ga(x) + x
        # x = self.norm3(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


        return x


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1,
                 att_reduction=3, num_heads=3):
        super().__init__()
        self.stride = stride

        # 使用图卷积替代传统卷积
        self.conv1 = PointwiseGraphConv(in_channels, in_channels)
        self.conv2 = DepthwiseGraphConv(in_channels, stride=stride)
        self.conv3 = PointwiseGraphConv(in_channels, out_channels)

        # 多策略注意力模块
        self.att = MultiStrategyAttention(in_channels, att_reduction, num_heads)

        # 跳跃连接
        self.shortcut = nn.Sequential()
        if stride == 1 and in_channels != out_channels:
            self.shortcut = nn.Sequential(
                PointwiseGraphConv(in_channels, out_channels),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        # 第一阶段图卷积
        out = F.relu(self.conv1(x))

        # 多策略注意力
        out = self.att(out)


        # 第二阶段图卷积
        out = F.relu(self.conv2(out))

        # 第三阶段图卷积
        out = self.conv3(out)

        # 残差连接
        if self.stride == 1:
            out += self.shortcut(identity)

        return out
#
#
#



# class MultiStrategyAttention(nn.Module):
#     def __init__(self, dim, att_reduction=2, num_heads=2):
#         super().__init__()
#         # CSA模块（通道稀疏注意力）
#         self.csa = ChannelSparseAttention(dim, att_reduction)
#         # SSA模块（空间稀疏注意力）
#         self.ssa = SpatialSparseAttention(dim)
#         # GA模块（全局注意力）
#         self.ga = GlobalAttention(dim, num_heads)
#
#         # 维度调整层
#         self.norm1 = nn.LayerNorm(dim)
#         self.norm2 = nn.LayerNorm(dim)
#         self.norm3 = nn.LayerNorm(dim)
#
#     def forward(self, x):
#         identity = x
#
#         # CSA阶段
#         x = self.csa(x)
#         x = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
#
#         # SSA阶段
#         x = self.ssa(x) + identity
#         x = self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
#
#         # GA阶段
#         x = self.ga(x) + x
#         x = self.norm3(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
#
#         return x
#
#
# class InvertedResidual(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1,
#                  att_reduction=3, num_heads=3):
#         super(InvertedResidual, self).__init__()
#         self.stride = stride
#
#         # 基础卷积层
#         self.conv1 = nn.Conv2d(in_channels, in_channels, 1)
#         self.conv2 = nn.Conv2d(in_channels, in_channels, 3,
#                                stride=stride, padding=1, groups=in_channels)
#         self.conv3 = nn.Conv2d(in_channels, out_channels, 1)
#
#         # 多策略注意力模块
#         self.att = MultiStrategyAttention(in_channels, att_reduction, num_heads)
#
#         # 跳跃连接
#         self.shortcut = nn.Sequential()
#         if stride == 1 and in_channels != out_channels:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=False),
#                 nn.BatchNorm2d(out_channels)
#             )
#
#     def forward(self, x):
#         identity = x
#
#         # 第一阶段卷积
#         out = F.relu(self.conv1(x))
#
#         # 多策略注意力
#         out = self.att(out)
#
#         # 第二阶段卷积
#         out = F.relu(self.conv2(out))
#
#         # 第三阶段卷积
#         out = self.conv3(out)
#
#         # 残差连接
#         if self.stride == 1:
#             out += self.shortcut(identity)
#
#         return out
#
# class InvertedResidual(nn.Module):
#     def __init__(self, in_channels, out_channels, att, stride=1):
#         super(InvertedResidual, self).__init__()
#         self.stride = stride
#         self.conv1 = DeformConv2d(in_channels, in_channels, kernel_size=1)
#         self.conv2 = DeformConv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1)
#         self.conv3 = DeformConv2d(in_channels, out_channels, kernel_size=1)
# ## 注意力模块
#         self.conv_att1 = DeformConv2d(in_channels, in_channels // att, kernel_size=1)
#         self.conv_att2 = DeformConv2d(in_channels // att, in_channels, kernel_size=1)
#
#
#     def forward(self, x):
#         identity = x
#         offset1 = self._get_offsets(x, self.conv1)
#         out1 = F.relu(self.conv1(x, offset1))
#
#         offset_att1 = self._get_offsets(out1, self.conv_att1)
#         att1 = F.relu(self.conv_att1(x, offset_att1))
#         offset_att2 = self._get_offsets(att1, self.conv_att2)
#         attention_weights = F.sigmoid(self.conv_att2(att1, offset_att2))
#         attention_weights_array = attention_weights.cpu().detach().numpy()
#         out_att = out1*attention_weights
#
#         offset2 = self._get_offsets(out_att, self.conv2)
#         out2 = self.conv2(out_att, offset2)
#         out2 = out2+out_att
#         offset3 = self._get_offsets(out2, self.conv3)
#         out3 = self.conv3(out2, offset3)
#
#         if self.stride == 1 and identity.size() == out3.size():
#             out3 += identity
#         return out3
#
#     def _get_offsets(self, x, conv):
#         N, C, H, W = x.size()
#         kernel_size = conv.kernel_size[0]  # 假设是正方形卷积核
#         offset_channels = 2 * kernel_size * kernel_size
#         return torch.zeros(N, offset_channels, H, W, device=x.device)



class KANLinear(torch.nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            enable_standalone_scale_spline=True,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                    torch.arange(-spline_order, grid_size + spline_order + 1) * h
                    + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                    (
                            torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                            - 1 / 2
                    )
                    * self.scale_noise
                    / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order: -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                            (x - grid[:, : -(k + 1)])
                            / (grid[:, k:-1] - grid[:, : -(k + 1)])
                            * bases[:, :, :-1]
                    ) + (
                            (grid[:, k + 1:] - x)
                            / (grid[:, k + 1:] - grid[:, 1:(-k)])
                            * bases[:, :, 1:]
                    )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )


        return base_output + spline_output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
                torch.arange(
                    self.grid_size + 1, dtype=torch.float32, device=x.device
                ).unsqueeze(1)
                * uniform_step
                + x_sorted[0]
                - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
                regularize_activation * regularization_loss_activation
                + regularize_entropy * regularization_loss_entropy
        )




class FixedPositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(FixedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = scale_factor * pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x):

        # x = x + self.pe[:x.size(0), :]
        x = x.permute(2, 0, 1)

        x = x + self.pe[:x.size(0), :]
        x = self.layer_norm(self.dropout(x))
        return x.transpose(0, 1)
class DCNNKAN(nn.Module):
    def __init__(self,in_channel,out_channel=64,att=8):
        super(DCNNKAN, self).__init__()
        # self.attention = GAM_Attention(15, 15, 4)
        self.conv1 = InvertedResidual(in_channel, 32, stride=1)
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(in_channel, 32, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU()
        # )#移除注意力机制DCSG
        self.pool1 = nn.MaxPool2d(2,2)
        self.dropout = nn.Dropout(p=0.25)
        self.conv2 = InvertedResidual(32, out_channel,stride=1)
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(32, out_channel, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(out_channel),
        #     nn.ReLU()
        # )#移除注意力机制
        self.pool2 = nn.MaxPool2d(2,2)
        self.dropout = nn.Dropout(p=0.25)
        self.RULPositionalEncoder = FixedPositionalEncoding(out_channel, max_len=1024)


        #KAN替换为MLP
        # self.mlp1 = nn.Linear(672 * 2, 64)  # 保持维度一致
        # self.mlp2 = nn.Linear(64, 1)  # 保持维度一致
        self.kan1 = KANLinear(672*2, 64)
        # self.lstm = nn.LSTM(input_size=672*2, hidden_size=128, batch_first=True)
        # self.dropout = nn.Dropout(p=0.25)
        self.kan2 = KANLinear(64, 1)
        # self.fc_in_features = 1344
        # self.fc = nn.Linear(self.fc_in_features, 1)


    def forward(self, x):
        # x = self.attention(x)
        # print(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.dropout(x)

        x = x.view(x.size(0),x.size(1), -1)
        x = self.RULPositionalEncoder(x)

        x = x.reshape(x.size(0), -1)
        # x, (h_n, c_n) = self.lstm(x)
        # x = self.dropout(x)
        # x = self.fc(x)

        # x = self.mlp1(x)
        # x = self.mlp2(x)

        x = self.kan1(x)
        x = self.kan2(x)
        return x

def print_parameter_details(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            params = parameter.numel()  # Number of elements in the tensor
            total_params += params
            print(f"{name}: {params}")
    print(f"Total trainable parameters: {total_params}")
