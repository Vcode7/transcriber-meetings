"""
PyTorch implementation of ERes2Net / ERes2Net-Large architecture.
Based on 3D-Speaker (Alibaba DAMO Academy) under Apache 2.0 license.

This module provides a pure, self-contained PyTorch implementation of ERes2Net
with zero third-party dependencies beyond torch and torchaudio. It performs
fully offline inference without importing ModelScope or making any network requests.
"""
from __future__ import annotations

import math
import os
from typing import Optional, Union, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.compliance.kaldi as Kaldi


class ReLU(nn.Hardtanh):
    def __init__(self, inplace: bool = False):
        super(ReLU, self).__init__(0, 20, inplace)

    def __repr__(self):
        inplace_str = "inplace" if self.inplace else ""
        return self.__class__.__name__ + " (" + inplace_str + ")"


def conv1x1(in_planes: int, out_planes: int, stride: int = 1):
    """1x1 convolution without padding."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        padding=0,
        bias=False,
    )


def conv3x3(in_planes: int, out_planes: int, stride: int = 1):
    """3x3 convolution with padding."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class AFF(nn.Module):
    """Attentive Feature Fusion."""

    def __init__(self, channels: int = 64, r: int = 4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels * 2, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor, ds_y: torch.Tensor) -> torch.Tensor:
        xa = torch.cat((x, ds_y), dim=1)
        x_att = self.local_att(xa)
        x_att = 1.0 + torch.tanh(x_att)
        xo = torch.mul(x, x_att) + torch.mul(ds_y, 2.0 - x_att)
        return xo


class BasicBlockERes2Net(nn.Module):
    expansion = 2

    def __init__(self, in_planes: int, planes: int, stride: int = 1, baseWidth: int = 32, scale: int = 2):
        super(BasicBlockERes2Net, self).__init__()
        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = conv1x1(in_planes, width * scale, stride)
        self.bn1 = nn.BatchNorm2d(width * scale)
        self.nums = scale

        convs = []
        bns = []
        for _ in range(self.nums):
            convs.append(conv3x3(width, width))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.relu = ReLU(inplace=True)

        self.conv3 = conv1x1(width * scale, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )
        self.stride = stride
        self.width = width
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        residual = self.shortcut(x)
        out += residual
        out = self.relu(out)

        return out


class BasicBlockERes2Net_AFF(nn.Module):
    expansion = 2

    def __init__(self, in_planes: int, planes: int, stride: int = 1, baseWidth: int = 32, scale: int = 2):
        super(BasicBlockERes2Net_AFF, self).__init__()
        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = conv1x1(in_planes, width * scale, stride)
        self.bn1 = nn.BatchNorm2d(width * scale)
        self.nums = scale

        convs = []
        fuse_models = []
        bns = []
        for _ in range(self.nums):
            convs.append(conv3x3(width, width))
            bns.append(nn.BatchNorm2d(width))
        for _ in range(self.nums - 1):
            fuse_models.append(AFF(channels=width))

        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.fuse_models = nn.ModuleList(fuse_models)
        self.relu = ReLU(inplace=True)

        self.conv3 = conv1x1(width * scale, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )
        self.stride = stride
        self.width = width
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = self.fuse_models[i - 1](sp, spx[i])

            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        residual = self.shortcut(x)
        out += residual
        out = self.relu(out)

        return out


class TSTP(nn.Module):
    """Temporal statistics pooling (concatenates mean and std across temporal axis)."""

    def __init__(self, **kwargs):
        super(TSTP, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooling_mean = x.mean(dim=-1)
        pooling_std = torch.sqrt(torch.var(x, dim=-1) + 1e-8)
        pooling_mean = pooling_mean.flatten(start_dim=1)
        pooling_std = pooling_std.flatten(start_dim=1)
        stats = torch.cat((pooling_mean, pooling_std), 1)
        return stats


class ERes2Net(nn.Module):
    """
    ERes2Net architecture with local and global feature fusion.
    For ERes2Net-Large, m_channels=64, num_blocks=[3, 4, 6, 3], embed_dim=512 (or 192/293).
    """

    def __init__(
        self,
        block=BasicBlockERes2Net,
        block_fuse=BasicBlockERes2Net_AFF,
        num_blocks=(3, 4, 6, 3),
        m_channels: int = 64,
        feat_dim: int = 80,
        embed_dim: int = 512,
        two_emb_layer: bool = False,
    ):
        super(ERes2Net, self).__init__()
        self.in_planes = m_channels
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        self.stats_dim = int(feat_dim / 8) * m_channels * 8
        self.two_emb_layer = two_emb_layer

        self.conv1 = nn.Conv2d(1, m_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels)
        self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, m_channels * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block_fuse, m_channels * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block_fuse, m_channels * 8, num_blocks[3], stride=2)

        # Downsampling layers
        self.layer1_downsample = nn.Conv2d(m_channels * 2, m_channels * 4, kernel_size=3, stride=2, padding=1, bias=False)
        self.layer2_downsample = nn.Conv2d(m_channels * 4, m_channels * 8, kernel_size=3, stride=2, padding=1, bias=False)
        self.layer3_downsample = nn.Conv2d(m_channels * 8, m_channels * 16, kernel_size=3, stride=2, padding=1, bias=False)

        # Bottom-up fusion
        self.fuse_mode12 = AFF(channels=m_channels * 4)
        self.fuse_mode123 = AFF(channels=m_channels * 8)
        self.fuse_mode1234 = AFF(channels=m_channels * 16)

        self.pool = TSTP()
        self.seg_1 = nn.Linear(self.stats_dim * block.expansion * 2, embed_dim)
        if self.two_emb_layer:
            self.seg_bn_1 = nn.BatchNorm1d(embed_dim, affine=False)
            self.seg_2 = nn.Linear(embed_dim, embed_dim)
        else:
            self.seg_bn_1 = nn.Identity()
            self.seg_2 = nn.Identity()

    def _make_layer(self, block, planes: int, num_blocks: int, stride: int):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: [B, T, F] -> permute to [B, F, T] -> unsqueeze to [B, 1, F, T]
        x = x.permute(0, 2, 1).unsqueeze(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out1 = self.layer1(out)

        out2 = self.layer2(out1)
        out1_downsample = self.layer1_downsample(out1)
        fuse_out12 = self.fuse_mode12(out2, out1_downsample)

        out3 = self.layer3(out2)
        fuse_out12_downsample = self.layer2_downsample(fuse_out12)
        fuse_out123 = self.fuse_mode123(out3, fuse_out12_downsample)

        out4 = self.layer4(out3)
        fuse_out123_downsample = self.layer3_downsample(fuse_out123)
        fuse_out1234 = self.fuse_mode1234(out4, fuse_out123_downsample)

        stats = self.pool(fuse_out1234)
        embed_a = self.seg_1(stats)

        if self.two_emb_layer:
            out = F.relu(embed_a)
            out = self.seg_bn_1(out)
            return self.seg_2(out)
        return embed_a


class OfflineERes2NetEncoder:
    """
    High-level offline inference wrapper for ERes2Net-Large.
    Extracts 80-dim Kaldi Fbank features and passes them through the PyTorch model.
    Runs 100% offline with zero network interaction.
    """

    def __init__(
        self,
        model_dir: str,
        checkpoint_name: Optional[str] = None,
        embed_dim: int = 512,
        channels: int = 64,
        device: str = "cpu",
    ):
        self.model_dir = model_dir
        self.device = torch.device(device)
        self.feature_dim = 80
        self.embed_dim = embed_dim

        # Load weights
        ckpt_path = None
        if checkpoint_name:
            p = os.path.join(model_dir, checkpoint_name)
            if os.path.exists(p):
                ckpt_path = p

        if not ckpt_path:
            # Look for common checkpoint names
            candidates = [
                "eres2net_large_model.ckpt",
                "model.pt",
                "eres2net_large.pt",
                "pytorch_model.bin",
                "model.bin",
            ]
            for c in candidates:
                p = os.path.join(model_dir, c)
                if os.path.exists(p):
                    ckpt_path = p
                    break

        if not ckpt_path:
            raise FileNotFoundError(
                f"No checkpoint file found in {model_dir}. "
                f"Expected one of: eres2net_large_model.ckpt, model.pt, pytorch_model.bin."
            )

        state_dict = torch.load(ckpt_path, map_location="cpu")
        # Handle state_dict wrapped in keys like 'state_dict' or 'model'
        if isinstance(state_dict, dict):
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]

        # Strip any 'embedding_model.' or 'module.' prefixes
        clean_state_dict = {}
        for k, v in state_dict.items():
            clean_k = k
            if clean_k.startswith("embedding_model."):
                clean_k = clean_k[len("embedding_model."):]
            if clean_k.startswith("module."):
                clean_k = clean_k[len("module."):]
            clean_state_dict[clean_k] = v

        # Infer embed_dim and channels dynamically from checkpoint if possible
        if "seg_1.weight" in clean_state_dict:
            embed_dim = clean_state_dict["seg_1.weight"].shape[0]
            self.embed_dim = embed_dim
        if "conv1.weight" in clean_state_dict:
            channels = clean_state_dict["conv1.weight"].shape[0]

        # Instantiate model
        self.model = ERes2Net(
            m_channels=channels,
            feat_dim=self.feature_dim,
            embed_dim=embed_dim,
        )
        self.model.load_state_dict(clean_state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()

    def extract_feature(self, audio_tensor: torch.Tensor) -> torch.Tensor:
        """Extract 80-channel Kaldi Fbank features with CMVN."""
        # audio_tensor: [1, T] in range [-1.0, 1.0] or [-32768, 32767]
        # Kaldi fbank expects waveform values in standard float range
        if audio_tensor.abs().max() <= 1.0:
            # Scale to [-32768, 32767] as expected by standard Kaldi compliance
            audio_tensor = audio_tensor * 32767.0
        feat = Kaldi.fbank(audio_tensor, num_mel_bins=self.feature_dim)
        feat = feat - feat.mean(dim=0, keepdim=True)
        return feat.unsqueeze(0)  # [1, T, 80]

    @torch.no_grad()
    def __call__(self, audio: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Extract speaker embedding from mono 16 kHz audio waveform.
        Returns L2-normalized 1D numpy array.
        """
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        feat = self.extract_feature(audio)
        feat = feat.to(self.device)

        emb = self.model(feat)
        emb = emb.squeeze().detach().cpu().numpy()

        norm = np.linalg.norm(emb)
        if norm > 1e-10:
            emb = emb / norm
        return emb
