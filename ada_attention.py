"""
This is Minimal AdaAttention implementation.

AdaAttention is a lightweight policy layer over PyTorch attention backends.
It selects a backend by sequence-length regime and optional memory pressure.

Input shape: [B, H, N, D_head]
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


def _sdpa_backend_context(name: str):
    if name in ("default", "naive"):
        return contextlib.nullcontext()

    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend
        if name == "math":
            return sdpa_kernel(SDPBackend.MATH)
        if name == "flash":
            return sdpa_kernel(SDPBackend.FLASH_ATTENTION)
        if name == "mem_efficient":
            return sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION)
    except Exception:
        pass

    try:
        if name == "math":
            return torch.backends.cuda.sdp_kernel(
                enable_flash=False, enable_math=True, enable_mem_efficient=False
            )
        if name == "flash":
            return torch.backends.cuda.sdp_kernel(
                enable_flash=True, enable_math=False, enable_mem_efficient=False
            )
        if name == "mem_efficient":
            return torch.backends.cuda.sdp_kernel(
                enable_flash=False, enable_math=False, enable_mem_efficient=True
            )
    except Exception:
        pass

    return contextlib.nullcontext()


@dataclass
class AdaAttentionPolicy:
    """Simple but regime-based policy.

    Default rule:
        N <= small_n      -> sdpa_default
        small_n < N       -> sdpa_flash
        memory tight      -> sdpa_mem_efficient
    """

    small_n: int = 2048
    large_n: int = 32768
    prefer_mem_efficient_when_memory_tight: bool = True
    memory_safety_fraction: float = 0.80

    def estimate_qkv_memory_bytes(self, q: torch.Tensor) -> int:
        b, h, n, d = q.shape
        return 3 * b * h * n * d * q.element_size()

    def choose_backend(self, q: torch.Tensor, memory_budget_bytes: Optional[int] = None) -> str:
        _, _, n, _ = q.shape

        if n <= self.small_n:
            return "default"

        if memory_budget_bytes is not None:
            qkv_bytes = self.estimate_qkv_memory_bytes(q)
            if qkv_bytes > self.memory_safety_fraction * memory_budget_bytes:
                if self.prefer_mem_efficient_when_memory_tight:
                    return "mem_efficient"

        return "flash"


def attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    backend: str = "sdpa_default",
    policy: Optional[AdaAttentionPolicy] = None,
    memory_budget_bytes: Optional[int] = None,
) -> torch.Tensor:
    """Run attention with a fixed backend or AdaAttention policy."""
    if backend == "ada":
        policy = policy or AdaAttentionPolicy()
        chosen = policy.choose_backend(q, memory_budget_bytes=memory_budget_bytes)
        backend = "sdpa_default" if chosen == "default" else f"sdpa_{chosen}"

    if backend == "naive":
        d = q.shape[-1]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
        probs = torch.softmax(scores, dim=-1)
        return torch.matmul(probs, v)

    if backend == "sdpa_default":
        return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)

    if backend == "sdpa_math":
        with _sdpa_backend_context("math"):
            return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)

    if backend == "sdpa_flash":
        with _sdpa_backend_context("flash"):
            return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)

    if backend == "sdpa_mem_efficient":
        with _sdpa_backend_context("mem_efficient"):
            return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)

    raise ValueError(f"Unknown backend: {backend}")


def make_qkv(batch_size: int, n_heads: int, seq_len: int, head_dim: int, dtype, device):
    q = torch.randn(batch_size, n_heads, seq_len, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch_size, n_heads, seq_len, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch_size, n_heads, seq_len, head_dim, dtype=dtype, device=device)
    return q, k, v
