import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

from models.superlora import SuperLoRALinear, count_parameters


#Count adapter params for a 768×768 ViT-base attention layer

def count_adapter_params_for_config(
    variant: str,
    rank: int,
    n_groups: int,
    tensor_order: int = 2,
    n_splits: int = 2,
    projection_ratio: float = 1.0,
    d: int = 768,
    n_layers: int = 12,   # 12 layers × 2 (query + value) = 24 modules
    n_qv: int = 2,
) -> int:
    linear = nn.Linear(d, d)
    adapted = SuperLoRALinear(
        linear,
        rank=rank,
        variant=variant,
        n_groups=n_groups,
        tensor_order=tensor_order,
        n_splits=n_splits,
        projection_ratio=projection_ratio,
    )
    # Each layer has query + value
    return adapted.num_adapter_parameters * n_layers * n_qv


#Sweep configs

SWEEP_CONFIGS = [
    # (label, variant, ranks, n_groups, tensor_order, n_splits, proj_ratio, color)
    ("LoRA (weight-wise)", "lora",  [1,2,4,8,16,32,64,128], 24, 2, 2, 1.0, "#e74c3c"),
    ("SuperLoRA 2D",       "lora",  [1,2,4,8,16,32,64,128],  1, 2, 2, 1.0, "#3498db"),
    ("LoRTA 3D",           "lorta", [1,2,4,8],                1, 3, 2, 1.0, "#2ecc71"),
    ("LoRTA 4D",           "lorta", [1,2,4],                  1, 4, 2, 1.0, "#9b59b6"),
    ("SuperLoRA+proj 0.1", "lora",  [1,2,4,8,16,32,64],       1, 2, 2, 0.1, "#f39c12"),
    ("LoKr",               "lokr",  [1,2,4,8,16,32],          1, 2, 2, 1.0, "#1abc9c"),
]


def run_sweep():
    print("Counting adapter parameters for each configuration...\n")
    results: Dict[str, List[int]] = {}

    for label, variant, ranks, n_groups, tensor_order, n_splits, proj_ratio, _ in SWEEP_CONFIGS:
        counts = []
        for r in ranks:
            try:
                n = count_adapter_params_for_config(
                    variant=variant,
                    rank=r,
                    n_groups=n_groups,
                    tensor_order=tensor_order,
                    n_splits=n_splits,
                    projection_ratio=proj_ratio,
                )
                counts.append(n)
            except Exception as e:
                counts.append(None)
                print(f"  [{label}] rank={r}: error — {e}")
        results[label] = counts
        print(f"  {label:30s}  params: {[c for c in counts if c is not None]}")

    return results


def plot_pareto(results: Dict, output_path: str = "pareto_params.png"):
    fig, ax = plt.subplots(figsize=(10, 6))

    for (label, variant, ranks, n_groups, tensor_order, n_splits, proj_ratio, color), counts in zip(
        SWEEP_CONFIGS, results.values()
    ):
        valid = [(c, r) for c, r in zip(counts, ranks) if c is not None and c > 0]
        if not valid:
            continue
        xs = [v[0] for v in valid]
        # Simulated accuracy: log-scale saturation curve (replace with real numbers)
        ys = [0.88 + 0.04 * (1 - math.exp(-x / 5e4)) + 0.01 * (rank / 128)
              for x, (_, rank) in zip(xs, valid)]

        ax.semilogx(xs, ys, "o-", label=label, color=color, linewidth=2, markersize=5)

    ax.set_xlabel("# Adapter Parameters", fontsize=13)
    ax.set_ylabel("Top-1 Accuracy (estimated)", fontsize=13)
    ax.set_title("SuperLoRA: Accuracy vs. Parameter Efficiency\n"
                 "(ViT-Base, CIFAR-100 transfer — estimated curve)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_ylim(0.87, 0.94)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nPareto plot saved to: {output_path}")
    return fig


if __name__ == "__main__":
    results = run_sweep()
    plot_pareto(results, output_path="pareto_params.png")