# AdaAttention Minimal

AdaAttention is a minimal adaptive attention wrapper for the project:

**Mechanism-Driven Analysis of Attention Scalability: From Memory Materialization to Adaptive Kernel Selection**

The goal is not to replace FlashAttention or PyTorch SDPA. AdaAttention implements a lightweight **runtime policy** that selects an attention backend based on sequence-length regime and optional memory pressure.

## Core Idea

All variants compute the same attention function:

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d)) V
```

but differ in how the computation is executed.

From the project results:

- Small sequence length: simple/default kernels can be competitive because overhead dominates.
- Large sequence length: non-materializing kernels such as FlashAttention are preferred because memory behavior dominates.
- Tight memory budget: memory-efficient kernels may be safer.

AdaAttention uses a transparent rule:

```text
small N      -> sdpa_default
medium/large N -> sdpa_flash
memory tight -> sdpa_mem_efficient
```

It does **not** train multiple kernels or run multiple kernels online. It selects one backend before execution.

## Files

```text
ada_attention.py   # adaptive attention wrapper
benchmark.py       # latency / throughput / VRAM benchmark
requirements.txt   # minimal dependencies
```

## Installation

Use a CUDA-enabled PyTorch environment.

```bash
pip install -r requirements.txt
```

## Basic Usage

```python
import torch
from ada_attention import attention_forward, make_qkv, AdaAttentionPolicy

device = torch.device("cuda")

q, k, v = make_qkv(
    batch_size=1,
    n_heads=8,
    seq_len=8192,
    head_dim=64,
    dtype=torch.float16,
    device=device,
)

policy = AdaAttentionPolicy(small_n=2048, large_n=32768)
out = attention_forward(q, k, v, backend="ada", policy=policy)
```

Input shape:

```text
[B, H, N, D_head]
```

## Benchmark

Default benchmark:

```bash
python benchmark.py
```

Dense benchmark:

```bash
python benchmark.py --dense --max_n 65536 --repeats 3
```

Safer large benchmark without naive attention:

```bash
python benchmark.py --dense --max_n 131072 --repeats 3 \
  --backends sdpa_default sdpa_flash sdpa_mem_efficient ada
```

If a backend is not supported on your GPU/PyTorch build, the script skips it.

## Benchmark Outputs

```text
adaattention_benchmark_outputs/adaattention_benchmark.csv
adaattention_benchmark_outputs/peak_vram_vs_N.png
adaattention_benchmark_outputs/latency_vs_N.png
adaattention_benchmark_outputs/throughput_vs_N.png
```

Metrics:

- **Peak VRAM (MB):** maximum GPU memory allocated during attention
- **Latency (ms):** forward-pass time
- **Throughput (tokens/sec):** sequence length divided by latency

## Policy Parameters

The default thresholds are:

```python
small_n = 2048
large_n = 32768
```

You can tune them based on benchmark results.

## Why Not Just Use PyTorch Default SDPA?

PyTorch SDPA already performs backend dispatch. AdaAttention adds a higher-level, workload-aware policy:

- PyTorch default SDPA: hardware/backend heuristic
- AdaAttention: explicit policy based on sequence length and memory budget

The purpose is transparency and deployment control, not claiming to beat PyTorch everywhere.

## Scope

This minimal implementation targets inference/forward-pass experiments. Training-time behavior would require benchmarking backward memory and recomputation.
