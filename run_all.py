import argparse
import json
import os
import sys
import time
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#Experiments to run (in order)

EXPERIMENTS = [
    {
        "preset":      "lora_baseline",
        "label":       "LoRA Baseline",
        "color":       "#e74c3c",
        "marker":      "o",
        "description": "Standard weight-wise LoRA (paper baseline)",
    },
    {
        "preset":      "superlora_2d",
        "label":       "SuperLoRA 2D",
        "color":       "#3498db",
        "marker":      "s",
        "description": "Group-wise SuperLoRA 2D (3–4× more efficient)",
    },
    {
        "preset":      "superlora_2d_reshape",
        "label":       "SuperLoRA 2D Reshape",
        "color":       "#2980b9",
        "marker":      "D",
        "description": "Group-wise SuperLoRA 2D with reshape",
    },
    {
        "preset":      "lorta_3d",
        "label":       "LoRTA 3D",
        "color":       "#2ecc71",
        "marker":      "^",
        "description": "Tucker 3D decomposition (10× more efficient)",
    },
    {
        "preset":      "lorta_4d",
        "label":       "LoRTA 4D",
        "color":       "#27ae60",
        "marker":      "v",
        "description": "Tucker 4D decomposition",
    },
    {
        "preset":      "superlora_projected",
        "label":       "SuperLoRA + Projection",
        "color":       "#f39c12",
        "marker":      "P",
        "description": "SuperLoRA with fastfood projection ρ=0.1",
    },
    {
        "preset":      "lokr",
        "label":       "LoKr",
        "color":       "#9b59b6",
        "marker":      "X",
        "description": "Kronecker product LoRA (K=2 splits)",
    },
]


#CLI

def parse_args():
    p = argparse.ArgumentParser(description="Run all SuperLoRA experiments")
    p.add_argument("--steps",    type=int, default=5000,
                   help="Training steps per experiment (default: 5000)")
    p.add_argument("--device",   type=str, default="auto",
                   help="Device: auto | cpu | mps | cuda")
    p.add_argument("--dataset",  type=str, default="cifar100",
                   choices=["cifar10", "cifar100"])
    p.add_argument("--skip",     type=str, nargs="*", default=[],
                   help="Preset name(s) to skip, e.g. --skip lorta_4d lokr")
    p.add_argument("--only",     type=str, nargs="*", default=[],
                   help="Run only these presets, e.g. --only lora_baseline lorta_3d")
    p.add_argument("--num_workers", type=int, default=0,
                   help="DataLoader workers (default 0, safe for macOS)")
    return p.parse_args()


#Run one experiment via subprocess

def run_experiment(exp: dict, args) -> dict:
    preset = exp["preset"]
    print("\n" + "=" * 62)
    print(f"  Starting: {exp['label']}")
    print(f"  Preset  : {preset}  |  Steps: {args.steps}  |  Device: {args.device}")
    print("=" * 62)

    cmd = [
        sys.executable, "train.py",
        "--preset",      preset,
        "--steps",       str(args.steps),
        "--device",      args.device,
        "--dataset",     args.dataset,
        "--num_workers", str(args.num_workers),
    ]

    # Set num_classes automatically
    if args.dataset == "cifar10":
        cmd += ["--num_classes", "10"]
    else:
        cmd += ["--num_classes", "100"]

    t0 = time.time()
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    elapsed = time.time() - t0

    # Load history if training succeeded
    history_path = Path("checkpoints") / preset / "history.json"
    history = None
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)

    return {
        "preset":    preset,
        "label":     exp["label"],
        "color":     exp["color"],
        "marker":    exp["marker"],
        "elapsed":   elapsed,
        "success":   result.returncode == 0,
        "history":   history,
    }


#Load trainable param count from checkpoint

def load_param_count(preset: str) -> int:
    ckpt_path = Path("checkpoints") / preset / "best_model.pt"
    if not ckpt_path.exists():
        return 0
    try:
        import torch
        ckpt = torch.load(ckpt_path, map_location="cpu")
        cfg  = ckpt.get("config", {})
        # Count from model state dict (adapter params only)
        state = ckpt.get("model_state", {})
        # Rough count: sum params with 'groups' in key name
        total = sum(v.numel() for k, v in state.items() if "groups" in k)
        return total if total > 0 else 0
    except Exception:
        return 0


#Plotting

def generate_report(results: list, args):
    successful = [r for r in results if r["success"] and r["history"]]

    if not successful:
        print("\n⚠️  No successful runs to plot.")
        return

    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("#0f1117")
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax_loss = fig.add_subplot(gs[0, 0:2])   # Training loss (wide)
    ax_acc  = fig.add_subplot(gs[0, 2])     # Val accuracy
    ax_time = fig.add_subplot(gs[1, 0])     # Time per run
    ax_tbl  = fig.add_subplot(gs[1, 1:3])   # Summary table

    style = dict(facecolor="#1a1d27", edgecolor="#333", linewidth=0.5)
    for ax in [ax_loss, ax_acc, ax_time, ax_tbl]:
        ax.set_facecolor("#1a1d27")
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")
        ax.tick_params(colors="#ccc", labelsize=9)
        ax.xaxis.label.set_color("#ccc")
        ax.yaxis.label.set_color("#ccc")
        ax.title.set_color("#eee")

    #Training Loss
    for r in successful:
        h = r["history"]
        if h.get("step") and h.get("train_loss"):
            ax_loss.plot(h["step"], h["train_loss"],
                        color=r["color"], label=r["label"],
                        linewidth=2, alpha=0.9)

    ax_loss.set_title("Training Loss", fontsize=12, fontweight="bold")
    ax_loss.set_xlabel("Step")
    ax_loss.set_ylabel("Cross-Entropy Loss")
    ax_loss.legend(fontsize=8, facecolor="#1a1d27", edgecolor="#444",
                   labelcolor="white")
    ax_loss.grid(True, alpha=0.15, color="#555")

    #Validation Accuracy
    for r in successful:
        h = r["history"]
        val_acc = h.get("val_acc", [])
        if val_acc:
            # val is logged every eval_interval; space out on step axis
            steps = h.get("step", [])
            eval_steps = steps[::max(1, len(steps)//len(val_acc))][:len(val_acc)]
            ax_acc.plot(eval_steps, val_acc,
                       color=r["color"], label=r["label"],
                       linewidth=2, marker=r["marker"],
                       markersize=5, alpha=0.9)

    ax_acc.set_title("Validation Accuracy", fontsize=12, fontweight="bold")
    ax_acc.set_xlabel("Step")
    ax_acc.set_ylabel("Top-1 Accuracy")
    ax_acc.legend(fontsize=8, facecolor="#1a1d27", edgecolor="#444",
                  labelcolor="white")
    ax_acc.grid(True, alpha=0.15, color="#555")

    #Time per run
    labels  = [r["label"] for r in successful]
    times   = [r["elapsed"] / 60 for r in successful]
    colors  = [r["color"]  for r in successful]
    bars = ax_time.barh(labels, times, color=colors, edgecolor="#333", height=0.6)
    for bar, t in zip(bars, times):
        ax_time.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                    f"{t:.1f}m", va="center", color="#ccc", fontsize=8)
    ax_time.set_title("Training Time", fontsize=12, fontweight="bold")
    ax_time.set_xlabel("Minutes")
    ax_time.grid(True, alpha=0.15, color="#555", axis="x")

    #Summary Table
    ax_tbl.axis("off")
    table_data = []
    col_labels = ["Variant", "Best Val Acc", "Final Loss", "Time (min)"]

    for r in successful:
        h = r["history"]
        best_acc   = f"{max(h['val_acc']):.4f}"  if h.get("val_acc")   else "—"
        final_loss = f"{h['train_loss'][-1]:.4f}" if h.get("train_loss") else "—"
        t_min      = f"{r['elapsed']/60:.1f}"
        table_data.append([r["label"], best_acc, final_loss, t_min])

    tbl = ax_tbl.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_facecolor("#252836" if row % 2 == 0 else "#1a1d27")
        cell.set_edgecolor("#444")
        cell.set_text_props(color="#eee")
        if row == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")

    ax_tbl.set_title("Results Summary", fontsize=12, fontweight="bold",
                     color="#eee", pad=12)

    #Title
    fig.suptitle(
        f"SuperLoRA Experiments — {args.dataset.upper()}  ({args.steps} steps)",
        fontsize=15, fontweight="bold", color="white", y=0.98
    )

    out_path = "superlora_results.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\n📊 Report saved → {out_path}")
    plt.close()


#Print final summary to terminal

def print_summary(results: list):
    print("\n" + "=" * 62)
    print("  FINAL SUMMARY")
    print("=" * 62)
    print(f"  {'Variant':<28} {'Status':<10} {'Best Acc':<12} {'Time'}")
    print("  " + "-" * 58)
    for r in results:
        if r["success"] and r["history"] and r["history"].get("val_acc"):
            best = max(r["history"]["val_acc"])
            acc_str = f"{best:.4f}"
        else:
            acc_str = "—"
        status = "✓ done" if r["success"] else "✗ failed"
        t_str  = f"{r['elapsed']/60:.1f} min"
        print(f"  {r['label']:<28} {status:<10} {acc_str:<12} {t_str}")
    print("=" * 62)


#Main
def main():
    args = parse_args()

    # Filter experiments
    exps = EXPERIMENTS
    if args.only:
        exps = [e for e in exps if e["preset"] in args.only]
    if args.skip:
        exps = [e for e in exps if e["preset"] not in args.skip]

    if not exps:
        print("No experiments selected. Check --only / --skip arguments.")
        return

    print("\n" + "=" * 62)
    print("  SuperLoRA — Running All Experiments")
    print(f"  Experiments : {len(exps)}")
    print(f"  Steps each  : {args.steps}")
    print(f"  Device      : {args.device}")
    print(f"  Dataset     : {args.dataset}")
    print("=" * 62)
    for e in exps:
        print(f"    • {e['label']}")
    print()

    total_start = time.time()
    results = []

    for exp in exps:
        r = run_experiment(exp, args)
        results.append(r)
        status = "✓ done" if r["success"] else "✗ FAILED"
        print(f"\n  [{status}] {exp['label']}  ({r['elapsed']/60:.1f} min)")

    total_elapsed = time.time() - total_start
    print(f"\n  Total time: {total_elapsed/60:.1f} min")

    # Generate plots + table
    generate_report(results, args)
    print_summary(results)

#final change to run all
if __name__ == "__main__":
    main()