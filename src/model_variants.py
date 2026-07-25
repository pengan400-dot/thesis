from __future__ import annotations

import torch
from torch import nn


class DepthwiseSeparableConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels,
                in_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
                groups=in_channels,
            ),
            nn.Conv1d(in_channels, out_channels, 1),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiScaleTrendBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        kernels: tuple[int, ...] = (3, 5, 7, 9),
        dilations: tuple[int, ...] = (1, 2, 4),
        dropout: float = 0.2,
        single_scale: bool = False,
    ) -> None:
        super().__init__()
        pairs = (
            [(5, 1)] if single_scale else [(k, d) for k in kernels for d in dilations]
        )
        branch_dim = d_model if single_scale else max(4, d_model // len(pairs))
        self.kernel_dilation_pairs = tuple(pairs)
        self.branches = nn.ModuleList(
            [
                DepthwiseSeparableConv1d(
                    input_dim,
                    branch_dim,
                    kernel_size=k,
                    dilation=d,
                    dropout=dropout,
                )
                for k, d in pairs
            ]
        )
        self.proj = nn.Sequential(
            nn.Linear(branch_dim * len(pairs), d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xt = x.transpose(1, 2)
        feats = [branch(xt).transpose(1, 2) for branch in self.branches]
        return self.proj(torch.cat(feats, dim=-1))


def apply_rope(x: torch.Tensor) -> torch.Tensor:
    _, _, seq_len, dim = x.shape
    half = dim // 2
    if half == 0:
        return x
    positions = torch.arange(seq_len, device=x.device, dtype=x.dtype)
    inv_freq = 1.0 / (
        10000 ** (torch.arange(0, half, device=x.device, dtype=x.dtype) / max(half, 1))
    )
    angles = positions[:, None] * inv_freq[None, :]
    sin = angles.sin()[None, None, :, :]
    cos = angles.cos()[None, None, :, :]
    x1 = x[..., :half]
    x2 = x[..., half : half * 2]
    rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    if dim % 2 == 1:
        rotated = torch.cat([rotated, x[..., -1:]], dim=-1)
    return rotated


class SelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        dropout: float = 0.2,
        use_rope: bool = True,
    ) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.use_rope = use_rope
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, d_model = x.shape
        qkv = (
            self.qkv(x)
            .view(batch, seq_len, 3, self.n_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.use_rope:
            q, k = apply_rope(q), apply_rope(k)
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        attn = self.dropout(torch.softmax(attn, dim=-1))
        out = torch.matmul(attn, v).transpose(1, 2).reshape(batch, seq_len, d_model)
        return self.out(out)


class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        dropout: float = 0.2,
        use_rope: bool = True,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = SelfAttention(d_model, n_heads, dropout, use_rope=use_rope)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x)))
        return x + self.dropout(self.ffn(self.norm2(x)))


class PaperMSTTVariant(nn.Module):
    """Paper topology plus explicitly named A2 ablations.

    full
        Exact 74,405-parameter paper topology.
    no_rope
        Same topology, but q/k are not rotary encoded.
    single_scale
        One local convolution branch (kernel=5, dilation=1).
    no_trend_residual
        Removes the recent-delta trend base; predicts around the last capacity.
    no_threshold_weight
        Same model as full. The threshold weighting is disabled in the loss.
    """

    ALLOWED = frozenset(
        {
            "full",
            "no_rope",
            "single_scale",
            "no_trend_residual",
            "no_threshold_weight",
        }
    )

    def __init__(
        self,
        input_dim: int,
        ablation: str = "full",
        d_model: int = 64,
        n_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2,
        residual_scale: float = 0.08,
    ) -> None:
        super().__init__()
        if ablation not in self.ALLOWED:
            raise ValueError(f"Unsupported ablation: {ablation}")
        if input_dim < 2:
            raise ValueError("input_dim must include cycle and capacity channels")
        self.ablation = ablation
        self.local = MultiScaleTrendBlock(
            input_dim,
            d_model,
            dropout=dropout,
            single_scale=(ablation == "single_scale"),
        )
        self.pos = nn.Dropout(dropout)
        self.encoder = nn.ModuleList(
            [
                EncoderLayer(
                    d_model,
                    n_heads,
                    dropout,
                    use_rope=(ablation != "no_rope"),
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        self.residual_scale = float(residual_scale)

    def forward(
        self,
        x: torch.Tensor,
        capacity_window: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del capacity_window
        cap = x[:, :, 1]
        last_capacity = cap[:, -1:]
        recent_delta = cap[:, -1:] - cap[:, -2:-1]
        if self.ablation == "no_trend_residual":
            trend_base = last_capacity
        else:
            trend_base = last_capacity + recent_delta.clamp(min=-0.04, max=0.02)
        z = self.local(x)
        z = self.pos(z)
        for layer in self.encoder:
            z = layer(z)
        pooled = z[:, -1] + z.mean(dim=1)
        residual = self.residual_scale * torch.tanh(self.head(self.dropout(pooled)))
        return (trend_base + residual).squeeze(-1)


def identity_record(model: PaperMSTTVariant) -> dict[str, object]:
    return {
        "class": type(model).__name__,
        "ablation": model.ablation,
        "kernel_dilation_pairs": [list(x) for x in model.local.kernel_dilation_pairs],
        "num_multiscale_branches": len(model.local.branches),
        "num_encoder_layers": len(model.encoder),
        "uses_rope": model.ablation != "no_rope",
        "uses_recent_delta_trend_base": model.ablation != "no_trend_residual",
        "uses_threshold_weight": model.ablation != "no_threshold_weight",
        "residual_scale": model.residual_scale,
        "trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
    }
