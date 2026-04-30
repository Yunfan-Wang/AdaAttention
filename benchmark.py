#!/usr/bin/env python3
"""Benchmark fixed attention backends and AdaAttention.

Outputs:
    - CSV
    - peak_vram_vs_N.png
    - latency_vs_N.png
    - throughput_vs_N.png
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time

import pandas as pd
import torch
import matplotlib.pyplot as plt

from ada_attention import AdaAttentionPolicy, attention_forward, make_qkv


def clear_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def dtype_from_name(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def build_seq_lengths(args):
    if args.seq_lengths:
        return sorted(set(args.seq_lengths))
    if args.dense:
        base = [512, 768, 1024, 1536, 2048, 3072, 4096, 6144,
                8192, 12288, 16384, 24576, 32768, 49152, 65536, 98304, 131072]
    else:
        base = [1024, 2048, 4096, 8192, 16384, 32768, 65536]
    return [n for n in base if n <= args.max_n]


def check_backend(backend, dtype, device, args):
    try:
        q, k, v = make_qkv(args.batch_size, args.n_heads, 256, args.head_dim, dtype, device)
        policy = AdaAttentionPolicy(args.small_n, args.large_n)
        _ = attention_forward(q, k, v, backend=backend, policy=policy)
        torch.cuda.synchronize()
        del q, k, v
        clear_cuda()
        return True
    except Exception as e:
        print(f"[skip] {backend}: {type(e).__name__}: {str(e).splitlines()[0]}")
        clear_cuda()
        return False


@torch.no_grad()
def benchmark_one(backend, n, args, dtype, device):
    q, k, v = make_qkv(args.batch_size, args.n_heads, n, args.head_dim, dtype, device)
    policy = AdaAttentionPolicy(args.small_n, args.large_n)

    memory_budget_bytes = None
    if args.use_memory_budget:
        total = torch.cuda.get_device_properties(device).total_memory
        memory_budget_bytes = int(total * args.memory_budget_fraction)

    for _ in range(args.warmup):
        _ = attention_forward(q, k, v, backend=backend, policy=policy,
                              memory_budget_bytes=memory_budget_bytes)
    torch.cuda.synchronize()

    times, peaks = [], []
    for _ in range(args.repeats):
        clear_cuda()
        torch.cuda.reset_peak_memory_stats()

        start = time.perf_counter()
        _ = attention_forward(q, k, v, backend=backend, policy=policy,
                              memory_budget_bytes=memory_budget_bytes)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        times.append(elapsed)
        peaks.append(torch.cuda.max_memory_allocated() / (1024 ** 2))

    avg_latency_s = sum(times) / len(times)
    avg_peak_mb = sum(peaks) / len(peaks)

    chosen = backend
    if backend == "ada":
        chosen = "ada->" + policy.choose_backend(q, memory_budget_bytes)

    del q, k, v
    clear_cuda()

    return {
        "backend": backend,
        "chosen_backend": chosen,
        "N": n,
        "dtype": args.dtype,
        "peak_vram_mb": avg_peak_mb,
        "latency_ms": avg_latency_s * 1000,
        "throughput_tokens_per_s": n / avg_latency_s,
    }


def plot_results(df, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    def plot(metric, ylabel, title, filename, logy=False):
        plt.figure(figsize=(8.4, 5.2))
        for backend, g in df.groupby("backend"):
            g = g.sort_values("N")
            plt.plot(g["N"], g[metric], marker="o", linewidth=1.8, label=backend)
        plt.title(title)
        plt.xlabel("Sequence Length N")
        plt.ylabel(ylabel)
        if logy:
            plt.yscale("log")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, filename), dpi=240)
        plt.close()

    plot("peak_vram_mb", "Peak VRAM (MB)", "Peak VRAM vs Sequence Length", "peak_vram_vs_N.png")
    plot("latency_ms", "Latency (ms)", "Latency vs Sequence Length", "latency_vs_N.png", logy=True)
    plot("throughput_tokens_per_s", "Throughput (tokens/sec)", "Throughput vs Sequence Length", "throughput_vs_N.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="adaattention_benchmark_outputs")
    parser.add_argument("--seq_lengths", type=int, nargs="*", default=None)
    parser.add_argument("--dense", action="store_true")
    parser.add_argument("--max_n", type=int, default=65536)
    parser.add_argument("--backends", nargs="+",
                        default=["naive", "sdpa_math", "sdpa_default", "sdpa_flash", "sdpa_mem_efficient", "ada"])
    parser.add_argument("--dtype", default="float16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--head_dim", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--small_n", type=int, default=2048)
    parser.add_argument("--large_n", type=int, default=32768)
    parser.add_argument("--naive_max_n", type=int, default=8192)
    parser.add_argument("--force_naive_large", action="store_true")
    parser.add_argument("--use_memory_budget", action="store_true")
    parser.add_argument("--memory_budget_fraction", type=float, default=0.80)
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Ignoring unknown args: {unknown}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for VRAM benchmarking.")

    device = torch.device("cuda")
    dtype = dtype_from_name(args.dtype)
    seq_lengths = build_seq_lengths(args)

    print("=" * 100)
    print("AdaAttention Benchmark")
    print("=" * 100)
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Shape: B={args.batch_size}, H={args.n_heads}, D_head={args.head_dim}")
    print(f"dtype={args.dtype}, seq_lengths={seq_lengths}")
    print(f"ada thresholds: small_n={args.small_n}, large_n={args.large_n}")
    print("=" * 100)

    available = [b for b in args.backends if check_backend(b, dtype, device, args)]

    rows = []
    for backend in available:
        for n in seq_lengths:
            if backend == "naive" and n > args.naive_max_n and not args.force_naive_large:
                print(f"{backend:18s} | N={n:7d} | SKIP: above naive_max_n")
                continue
            try:
                row = benchmark_one(backend, n, args, dtype, device)
                rows.append(row)
                print(f"{backend:18s} | {row['chosen_backend']:18s} | N={n:7d} | "
                      f"VRAM={row['peak_vram_mb']:9.2f} MB | "
                      f"latency={row['latency_ms']:9.3f} ms | "
                      f"throughput={row['throughput_tokens_per_s']:12.2f} tok/s")
            except RuntimeError as e:
                print(f"{backend:18s} | N={n:7d} | SKIP/OOM: {str(e).splitlines()[0]}")
                clear_cuda()
            except Exception as e:
                print(f"{backend:18s} | N={n:7d} | SKIP: {type(e).__name__}: {str(e).splitlines()[0]}")
                clear_cuda()

    df = pd.DataFrame(rows)
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "adaattention_benchmark.csv")
    df.to_csv(csv_path, index=False)
    if not df.empty:
        plot_results(df, args.out_dir)

    print("=" * 100)
    print(f"Saved CSV: {csv_path}")
    print(f"Saved plots to: {args.out_dir}")
    if not df.empty:
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
