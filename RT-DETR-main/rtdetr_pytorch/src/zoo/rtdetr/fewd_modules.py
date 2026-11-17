"""FEWD-DETR building blocks for the PyTorch RT-DETR implementation.

The modules implement the mechanisms described in 模型改进文档.docx:
partial-channel re-parameterized residual backbone, EAA encoder, and
SE-guided cross-scale feature fusion. No pretrained FEWD checkpoint is implied.
"""
from collections import OrderedDict
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core import register
from .hybrid_encoder import HybridEncoder, TransformerEncoder


def _conv_bn(in_ch, out_ch, kernel, stride=1, padding=None, groups=1):
    padding = kernel // 2 if padding is None else padding
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(in_ch, out_ch, kernel, stride, padding, groups=groups, bias=False)),
        ("bn", nn.BatchNorm2d(out_ch)),
    ]))


def _fuse_conv_bn(branch):
    if branch is None:
        return None, None
    conv, bn = branch.conv, branch.bn
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    kernel = conv.weight * scale.reshape(-1, 1, 1, 1)
    bias = bn.bias - bn.running_mean * scale
    return kernel, bias


class PartialRepConv(nn.Module):
    """Apply re-parameterizable spatial processing to only part of channels."""
    def __init__(self, channels, ratio=0.25, deploy=False):
        super().__init__()
        self.channels = channels
        self.partial = max(1, min(channels, int(channels * ratio)))
        self.deploy = deploy
        self.out_act = nn.ReLU(inplace=True)
        if deploy:
            self.reparam = nn.Conv2d(self.partial, self.partial, 3, padding=1, bias=True)
        else:
            self.branch3 = _conv_bn(self.partial, self.partial, 3)
            self.branch1 = _conv_bn(self.partial, self.partial, 1, padding=0)

    def forward(self, x):
        active, bypass = x[:, :self.partial], x[:, self.partial:]
        if self.deploy:
            y = self.out_act(self.reparam(active))
        else:
            y = self.out_act(self.branch3(active) + self.branch1(active) + active)
        return torch.cat((y, bypass), dim=1) if bypass.shape[1] else y

    @torch.no_grad()
    def convert_to_deploy(self):
        if self.deploy:
            return self
        k3, b3 = _fuse_conv_bn(self.branch3)
        k1, b1 = _fuse_conv_bn(self.branch1)
        k1 = F.pad(k1, (1, 1, 1, 1))
        identity = torch.zeros_like(k3)
        idx = torch.arange(self.partial, device=k3.device)
        identity[idx, idx, 1, 1] = 1
        fused = nn.Conv2d(self.partial, self.partial, 3, padding=1, bias=True).to(k3.device, k3.dtype)
        fused.weight.copy_(k3 + k1 + identity)
        fused.bias.copy_(b3 + b1)
        self.reparam = fused
        del self.branch3, self.branch1
        self.deploy = True
        return self


class FRepBlock(nn.Module):
    """Residual unit with a partial-channel, deploy-fusible spatial operator."""
    def __init__(self, in_channels, out_channels, stride=1, ratio=0.25):
        super().__init__()
        self.reduce = _conv_bn(in_channels, out_channels, 1, stride)
        self.rep = PartialRepConv(out_channels, ratio)
        self.skip = (nn.Identity() if in_channels == out_channels and stride == 1
                     else _conv_bn(in_channels, out_channels, 1, stride))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.rep(self.reduce(x)) + self.skip(x))

    def convert_to_deploy(self):
        self.rep.convert_to_deploy()
        return self


@register
class FRepNet(nn.Module):
    """ResNet-style FRepNet returning stride 8/16/32 features for RT-DETR."""
    def __init__(self, depth=18, return_idx=(1, 2, 3), freeze_at=-1,
                 freeze_norm=False, pretrained=False, partial_ratio=0.25):
        super().__init__()
        if depth not in (18, 34):
            raise ValueError("FRepNet currently supports depth 18 or 34")
        counts = {18: (2, 2, 2, 2), 34: (3, 4, 6, 3)}[depth]
        self.stem = nn.Sequential(
            _conv_bn(3, 32, 3, 2), nn.ReLU(inplace=True),
            _conv_bn(32, 32, 3), nn.ReLU(inplace=True),
            _conv_bn(32, 64, 3), nn.ReLU(inplace=True),
        )
        channels = (64, 128, 256, 512)
        stages, in_ch = [], 64
        for stage_idx, (out_ch, count) in enumerate(zip(channels, counts)):
            blocks = []
            for block_idx in range(count):
                stride = 2 if stage_idx > 0 and block_idx == 0 else 1
                blocks.append(FRepBlock(in_ch, out_ch, stride, partial_ratio))
                in_ch = out_ch
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.ModuleList(stages)
        self.return_idx = tuple(return_idx)
        self.out_channels = [channels[i] for i in self.return_idx]
        self.out_strides = [4 * (2 ** i) for i in self.return_idx]
        if freeze_at >= 0:
            for p in self.stem.parameters(): p.requires_grad = False
            for i in range(min(freeze_at, len(self.stages))):
                for p in self.stages[i].parameters(): p.requires_grad = False
        if freeze_norm:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    for p in m.parameters(): p.requires_grad = False
        if pretrained:
            raise ValueError("FRepNet has no compatible upstream pretrained checkpoint; load a matching FEWD checkpoint explicitly")

    def forward(self, x):
        x = F.max_pool2d(self.stem(x), kernel_size=3, stride=2, padding=1)
        outputs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx in self.return_idx:
                outputs.append(x)
        return outputs

    def deploy(self):
        self.eval()
        for m in self.modules():
            if isinstance(m, PartialRepConv): m.convert_to_deploy()
        return self


class EfficientAdditiveAttention(nn.Module):
    """Linear-complexity query-key interaction used by the ETE encoder."""
    def __init__(self, dim):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.weight = nn.Parameter(torch.randn(dim, 1) * 0.02)
        self.scale = dim ** -0.5
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        q = F.normalize(self.q_proj(x) * self.weight.squeeze(-1), dim=-1)
        k = F.normalize(self.k_proj(x), dim=-1)
        global_q = F.normalize(q.sum(dim=1, keepdim=True) * self.scale, dim=-1)
        return self.out_proj(k * global_q)


class ETEEncoderLayer(nn.Module):
    def __init__(self, dim, feedforward=1024, dropout=0.0, activation="gelu"):
        super().__init__()
        self.attn = EfficientAdditiveAttention(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, feedforward), nn.GELU() if activation == "gelu" else nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(feedforward, dim))
        self.drop = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, src_mask=None, pos_embed=None):
        q = x if pos_embed is None else x + pos_embed
        x = self.norm1(x + self.drop(self.attn(q)))
        return self.norm2(x + self.drop(self.ffn(x)))


class SEGate(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(channels, hidden, 1), nn.ReLU(inplace=True),
                                nn.Conv2d(hidden, channels, 1), nn.Sigmoid())

    def forward(self, x):
        return self.fc(self.pool(x))


class RepFusionBlock(nn.Module):
    """Compact residual fusion block with deploy-time branch re-parameterization."""
    def __init__(self, channels):
        super().__init__()
        self.pw = _conv_bn(channels, channels, 1)
        self.rep = PartialRepConv(channels, 0.25)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.rep(self.pw(x)))


class CGFM(nn.Module):
    """SE-guided cross-scale fusion for a concatenated pair of feature maps."""
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        if in_channels != out_channels * 2:
            raise ValueError("CGFM expects concatenated feature maps with equal channel counts")
        self.channels = out_channels
        self.se = SEGate(in_channels)
        self.fuse = _conv_bn(out_channels * 2, out_channels, 1)
        self.rep = RepFusionBlock(out_channels)

    def forward(self, x):
        a, b = x.split(self.channels, dim=1)
        wa, wb = self.se(x).split(self.channels, dim=1)
        cross = torch.cat((a + b * wa, b + a * wb), dim=1)
        return self.rep(self.fuse(cross))


@register
class FEWDEncoder(HybridEncoder):
    """RT-DETR hybrid encoder with ETE and CGFM in both FPN and PAN paths."""
    def __init__(self, in_channels=(128, 256, 512), feat_strides=(8, 16, 32),
                 hidden_dim=256, nhead=8, dim_feedforward=1024, dropout=0.0,
                 enc_act="gelu", use_encoder_idx=(2,), num_encoder_layers=1,
                 pe_temperature=10000, expansion=0.5, depth_mult=1.0, act="silu",
                 eval_spatial_size=None):
        super().__init__(in_channels=list(in_channels), feat_strides=list(feat_strides),
                         hidden_dim=hidden_dim, nhead=nhead, dim_feedforward=dim_feedforward,
                         dropout=dropout, enc_act=enc_act, use_encoder_idx=list(use_encoder_idx),
                         num_encoder_layers=num_encoder_layers, pe_temperature=pe_temperature,
                         expansion=expansion, depth_mult=depth_mult, act=act,
                         eval_spatial_size=eval_spatial_size)
        layer = ETEEncoderLayer(hidden_dim, dim_feedforward, dropout, enc_act)
        self.encoder = nn.ModuleList([
            TransformerEncoder(copy.deepcopy(layer), num_encoder_layers)
            for _ in range(len(self.use_encoder_idx))
        ])
        self.fpn_blocks = nn.ModuleList([CGFM(hidden_dim * 2, hidden_dim) for _ in self.fpn_blocks])
        self.pan_blocks = nn.ModuleList([CGFM(hidden_dim * 2, hidden_dim) for _ in self.pan_blocks])

