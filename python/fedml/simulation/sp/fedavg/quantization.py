"""Fixed-point conversion helpers used by the DMCFE integration."""

from __future__ import annotations

import torch


def _scale_from_precision(precision: int) -> int:
    if (
        isinstance(precision, bool)
        or not isinstance(precision, int)
        or precision < 0
        or precision > 62
    ):
        raise ValueError("precision must be an integer in [0, 62]")
    return 1 << precision


def quantize(vector: torch.Tensor, precision: int = 14) -> torch.Tensor:
    """
    将浮点模型更新转换为定点整数。

    q = round(x * 2^precision)
    """

    if not isinstance(vector, torch.Tensor):
        raise TypeError("quantize expects a torch.Tensor")
    if not torch.is_floating_point(vector):
        raise TypeError(f"quantize expects floating tensor, got {vector.dtype}")

    scale = _scale_from_precision(precision)
    value = vector.detach().to(dtype=torch.float64)
    if not torch.isfinite(value).all():
        raise ValueError("quantize does not accept NaN or infinity")
    quantized = torch.round(value * scale)
    # int64's positive range is [-2**63, 2**63-1].  Compare against 2**63
    # instead of converting max-int64 to float (which rounds to 2**63).
    if quantized.numel() and float(quantized.abs().max()) >= float(1 << 63):
        raise OverflowError("quantized values do not fit in int64")
    return quantized.to(torch.int64)


def dequantize(quantized: torch.Tensor, precision: int = 14) -> torch.Tensor:
    """
    将定点整数还原为浮点数。

    x_hat = q / 2^precision
    """

    scale = _scale_from_precision(precision)
    if not isinstance(quantized, torch.Tensor):
        raise TypeError("dequantize expects a torch.Tensor")
    if quantized.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        raise TypeError(f"dequantize expects an integer tensor, got {quantized.dtype}")
    return quantized.detach().to(dtype=torch.float64) / scale
