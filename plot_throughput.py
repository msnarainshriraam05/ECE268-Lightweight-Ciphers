#!/usr/bin/env python3
"""Generate CTR throughput (GB/s) vs input size charts from benchmark data.

Data source: gpu_bench.cu on NVIDIA Tesla T4 (Colab), CTR mode.
CPU and GPU times measured on the same machine for fair comparison.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent

# Input sizes (KiB) and labels
SIZES_KIB = [64, 1024, 16384]
LABELS = ["64 KiB", "1 MiB", "16 MiB"]

# Wall-clock times in milliseconds (Tesla T4, gpu_bench.cu output)
DATA = {
    "SIMON CPU": [0.78, 11.64, 201.92],
    "SIMON GPU": [0.04, 0.03, 0.48],
    "GIFT CPU":  [77.22, 1290.40, 23516.44],
    "GIFT GPU":  [0.39, 3.33, 88.56],
}

# Google Slides theme accent colors
COLORS = {
    "SIMON CPU": "#4285F4",
    "SIMON GPU": "#212121",
    "GIFT CPU":  "#78909C",
    "GIFT GPU":  "#0097A7",
}
MARKERS = {
    "SIMON CPU": "o",
    "SIMON GPU": "s",
    "GIFT CPU":  "^",
    "GIFT GPU":  "D",
}


def throughput_gbps(size_kib, time_ms):
    size_bytes = size_kib * 1024
    return size_bytes / (time_ms / 1000.0) / 1e9


def main():
    x = SIZES_KIB

    # ── Chart 1: all four series ───────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, times in DATA.items():
        y = [throughput_gbps(k, t) for k, t in zip(x, times)]
        ax.plot(x, y, marker=MARKERS[name], linewidth=2, markersize=8,
                label=name, color=COLORS[name])

    ax.set_xscale("log", base=2)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.set_xlabel("Input Data Size")
    ax.set_ylabel("CTR Throughput (GB/s)")
    ax.set_title("CTR Throughput vs. Input Size (NVIDIA Tesla T4)")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    path1 = OUT_DIR / "throughput_ctr_all.png"
    fig.savefig(path1, dpi=150)
    plt.close(fig)
    print(f"Wrote {path1}")

    # ── Chart 2: SIMON CPU vs GPU ────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name in ("SIMON CPU", "SIMON GPU"):
        times = DATA[name]
        y = [throughput_gbps(k, t) for k, t in zip(x, times)]
        ax.plot(x, y, marker=MARKERS[name], linewidth=2, markersize=8,
                label=name, color=COLORS[name])
    ax.set_xscale("log", base=2)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.set_xlabel("Input Data Size")
    ax.set_ylabel("CTR Throughput (GB/s)")
    ax.set_title("SIMON 64/128 — CPU vs GPU")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    path2 = OUT_DIR / "throughput_ctr_simon.png"
    fig.savefig(path2, dpi=150)
    plt.close(fig)
    print(f"Wrote {path2}")

    # ── Chart 3: GIFT CPU vs GPU ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name in ("GIFT CPU", "GIFT GPU"):
        times = DATA[name]
        y = [throughput_gbps(k, t) for k, t in zip(x, times)]
        ax.plot(x, y, marker=MARKERS[name], linewidth=2, markersize=8,
                label=name, color=COLORS[name])
    ax.set_xscale("log", base=2)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.set_xlabel("Input Data Size")
    ax.set_ylabel("CTR Throughput (GB/s)")
    ax.set_title("GIFT-128 — CPU vs GPU")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    path3 = OUT_DIR / "throughput_ctr_gift.png"
    fig.savefig(path3, dpi=150)
    plt.close(fig)
    print(f"Wrote {path3}")

    # ── Chart 4: two-panel (SIMON | GIFT) for presentation slide ──
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, cipher, cpu_key, gpu_key in [
        (axes[0], "SIMON 64/128", "SIMON CPU", "SIMON GPU"),
        (axes[1], "GIFT-128", "GIFT CPU", "GIFT GPU"),
    ]:
        for name in (cpu_key, gpu_key):
            times = DATA[name]
            y = [throughput_gbps(k, t) for k, t in zip(x, times)]
            ax.plot(x, y, marker=MARKERS[name], linewidth=2, markersize=7,
                    label=name.replace("SIMON ", "").replace("GIFT ", ""),
                    color=COLORS[name])
        ax.set_xscale("log", base=2)
        ax.set_xticks(x)
        ax.set_xticklabels(LABELS, fontsize=9)
        ax.set_xlabel("Input Size")
        ax.set_ylabel("GB/s")
        ax.set_title(cipher)
        ax.legend(fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_ylim(bottom=0)
    fig.suptitle("CTR Throughput — CPU vs GPU (Tesla T4)", fontsize=12)
    fig.tight_layout()
    path4 = OUT_DIR / "throughput_ctr_panels.png"
    fig.savefig(path4, dpi=150)
    plt.close(fig)
    print(f"Wrote {path4}")


if __name__ == "__main__":
    main()
