from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import torch
from torch import Tensor, nn
import torch.nn.functional as F


FULL_KERNEL_DILATION_PAIRS: tuple[tuple[int, int], ...] = tuple(
    (kernel, dilation)
    for kernel in (3, 5, 7, 9)
    for dilation in (1, 2, 4)
)
SINGLE_SCALE_PAIR: tuple[tuple[int, int], ...] = ((5, 1),)
EXPECTED_FULL_PARAMETERS = 74_405
EXPECTED_SINGLE_SCALE_PARAMETERS = 69_789


class CausalDepthwiseBranch(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_channels: int,
        kernel: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.left_padding = dilation * (kernel - 1)
        self.net = nn.Sequential(
            nn.Conv1d(
                in_features,
                in_features,
                kernel,
                dilation=dilation,
                groups=in_features,
                bias=True,
            ),
            nn.Conv1d(in_features, out_channels, 1, bias=True),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(F.pad(x, (self.left_padding, 0)))


class MultiScaleLocalEncoder(nn.Module):
    def __init__(
        self,
        *,
        in_features: int = 7,
        branch_channels: int = 5,
        d_model: int = 64,
        pairs: Iterable[tuple[int, int]] = FULL_KERNEL_DILATION_PAIRS,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.kernel_dilation_pairs = tuple(pairs)
        self.branches = nn.ModuleList(
            CausalDepthwiseBranch(
                in_features,
                branch_channels,
                kernel,
                dilation,
                dropout,
            )
            for kernel, dilation in self.kernel_dilation_pairs
        )
        self.proj = nn.Sequential(
            nn.Linear(
                len(self.kernel_dilation_pairs) * branch_channels,
                d_model,
            ),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        channels_first = x.transpose(1, 2)
        encoded = torch.cat(
            [branch(channels_first) for branch in self.branches],
            dim=1,
        )
        return self.proj(encoded.transpose(1, 2))


def _rotate_half(x: Tensor) -> Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _apply_rope(x: Tensor, base: float) -> Tensor:
    head_dim = x.shape[-1]
    if head_dim % 2:
        raise ValueError("RoPE requires an even attention-head dimension")
    positions = torch.arange(
        x.shape[-2],
        device=x.device,
        dtype=x.dtype,
    )
    frequencies = 1.0 / (
        base
        ** (
            torch.arange(
                0,
                head_dim,
                2,
                device=x.device,
                dtype=x.dtype,
            )
            / head_dim
        )
    )
    angles = torch.einsum("t,d->td", positions, frequencies)
    angles = torch.repeat_interleave(angles, 2, dim=-1)[None, None]
    return x * angles.cos() + _rotate_half(x) * angles.sin()


class RotarySelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        heads: int,
        dropout: float,
        *,
        rope_base: float,
    ) -> None:
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must be divisible by heads")
        self.heads = heads
        self.head_dim = d_model // heads
        self.rope_base = float(rope_base)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        batch, time, d_model = x.shape
        qkv = self.qkv(x).reshape(
            batch,
            time,
            3,
            self.heads,
            self.head_dim,
        )
        query, key, value = qkv.unbind(dim=2)
        query, key, value = (
            item.transpose(1, 2) for item in (query, key, value)
        )
        query = _apply_rope(query, self.rope_base)
        key = _apply_rope(key, self.rope_base)
        attention = (
            torch.matmul(query, key.transpose(-2, -1))
            / math.sqrt(self.head_dim)
        )
        attention = self.dropout(attention.softmax(dim=-1))
        result = (
            torch.matmul(attention, value)
            .transpose(1, 2)
            .reshape(batch, time, d_model)
        )
        return self.out(result)


class EncoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        heads: int,
        ffn_dim: int,
        dropout: float,
        *,
        rope_base: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = RotarySelfAttention(
            d_model,
            heads,
            dropout,
            rope_base=rope_base,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.dropout(self.attention(self.norm1(x)))
        return x + self.dropout(self.feed_forward(self.norm2(x)))


@dataclass(frozen=True)
class SOHModelOptions:
    ablation: str = "full"
    attention_heads: int = 4
    dropout: float = 0.1
    rope_base: float = 10_000.0
    residual_scale_soh: float = 0.04
    trend_delta_window: int = 3
    trend_delta_clip_soh: tuple[float, float] = (-0.02, 0.01)


class PaperMSTTVariant(nn.Module):
    """The new SOH-native MSTT topology.

    The full arm retains the archived 74,405 trainable-parameter topology.
    Its weights are new SOH weights and are not compatible evidence for the
    legacy absolute-Ah checkpoint.
    """

    def __init__(
        self,
        input_dim: int = 7,
        ablation: str = "full",
        d_model: int = 64,
        n_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        residual_scale: float = 0.04,
        rope_base: float = 10_000.0,
        trend_delta_window: int = 3,
        trend_delta_clip_soh: tuple[float, float] = (-0.02, 0.01),
    ) -> None:
        super().__init__()
        if ablation not in {"full", "single_scale"}:
            raise ValueError(f"Unsupported frozen ablation: {ablation}")
        if input_dim != 7:
            raise ValueError("The frozen SOH model requires exactly 7 inputs")
        pairs = (
            FULL_KERNEL_DILATION_PAIRS
            if ablation == "full"
            else SINGLE_SCALE_PAIR
        )
        self.options = SOHModelOptions(
            ablation=ablation,
            attention_heads=n_heads,
            dropout=dropout,
            rope_base=rope_base,
            residual_scale_soh=residual_scale,
            trend_delta_window=trend_delta_window,
            trend_delta_clip_soh=trend_delta_clip_soh,
        )
        self.ablation = ablation
        self.local = MultiScaleLocalEncoder(
            in_features=input_dim,
            branch_channels=5,
            d_model=d_model,
            pairs=pairs,
            dropout=dropout,
        )
        self.encoder = nn.ModuleList(
            EncoderBlock(
                d_model,
                n_heads,
                d_model * 2,
                dropout,
                rope_base=rope_base,
            )
            for _ in range(num_layers)
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(
        self,
        standardized_features: Tensor,
        capacity_window: Tensor | None = None,
    ) -> Tensor:
        if capacity_window is None:
            capacity_window = standardized_features[:, :, 1]
        encoded = self.local(standardized_features)
        for layer in self.encoder:
            encoded = layer(encoded)
        raw = self.head(encoded[:, -1]).squeeze(-1)
        corrections = self.options.residual_scale_soh * torch.tanh(raw)
        differences = capacity_window[:, 1:] - capacity_window[:, :-1]
        window = min(
            self.options.trend_delta_window,
            differences.shape[1],
        )
        recent_delta = differences[:, -window:].mean(dim=1)
        lower, upper = self.options.trend_delta_clip_soh
        recent_delta = recent_delta.clamp(min=lower, max=upper)
        trend = capacity_window[:, -1] + recent_delta
        return trend + corrections


def identity_record(model: PaperMSTTVariant) -> dict[str, object]:
    count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    expected = (
        EXPECTED_FULL_PARAMETERS
        if model.ablation == "full"
        else EXPECTED_SINGLE_SCALE_PARAMETERS
    )
    if count != expected:
        raise AssertionError(
            f"{model.ablation}: expected {expected:,} parameters; got {count:,}"
        )
    return {
        "class": type(model).__name__,
        "target_unit": "SOH_fraction",
        "model_version": "v0.2.0-SOH",
        "legacy_absolute_Ah_checkpoint_compatible": False,
        "ablation": model.ablation,
        "kernel_dilation_pairs": [
            list(pair)
            for pair in model.local.kernel_dilation_pairs
        ],
        "num_multiscale_branches": len(model.local.branches),
        "branch_channels": 5,
        "num_encoder_layers": len(model.encoder),
        "uses_rope": True,
        "uses_recent_delta_trend_base": True,
        "residual_scale_soh": model.options.residual_scale_soh,
        "trend_delta_window": model.options.trend_delta_window,
        "trend_delta_clip_soh": list(
            model.options.trend_delta_clip_soh
        ),
        "trainable_parameters": count,
    }
