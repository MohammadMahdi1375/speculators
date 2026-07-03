#!/usr/bin/env python3
import argparse
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

"""
python ./logs/plot.py   --log ./logs/train_dsv4_00.log   --label dsv4   --smoothing 0.5   --position-step 1000   --out-prefix dsv4_512 --outdir ./logs/ --curve-label dsv4
"""
FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def parse_dflash_log(path):
    with open(path, "r", errors="ignore") as f:
        text = f.read()

    text = re.sub(r"\n\s+", " ", text)

    block_pat = re.compile(
        rf"train/loss=({FLOAT}).*?"
        rf"train/full_acc=({FLOAT}).*?"
        rf"global_step=(\d+)",
        re.DOTALL,
    )

    pos_pat = re.compile(
        rf"train/position\s+(\d+)\s+acc=({FLOAT})"
    )

    records = {}

    for m in block_pat.finditer(text):
        chunk = text[m.start():m.end()]

        step = int(m.group(3))

        records[step] = {
            "step": step,
            "loss": float(m.group(1)),
            "full_acc": float(m.group(2)),
            "pos_acc": {
                int(pm.group(1)): float(pm.group(2))
                for pm in pos_pat.finditer(chunk)
            },
        }

    return [records[s] for s in sorted(records)]


def ema_smooth(values, weight):
    if weight <= 0:
        return list(values)

    out = []
    last = 0.0
    n = 0

    for v in values:
        last = last * weight + (1.0 - weight) * v
        n += 1
        out.append(last / (1.0 - weight ** n))

    return out


def nearest_step(records, target_step):
    return min(records, key=lambda r: abs(r["step"] - target_step))


def set_plot_style():
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "font.size": 12,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    })


def plot_training(curve_label, records, label, out, smoothing=0.9, show_raw=True):
    steps = [r["step"] for r in records]
    losses = [r["loss"] for r in records]
    accs = [r["full_acc"] for r in records]

    set_plot_style()

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))

    ax = axes[0]
    if show_raw:
        ax.plot(steps, losses, alpha=0.08, linewidth=0.8)
    ax.plot(
        steps,
        ema_smooth(losses, smoothing),
        linewidth=3.0,
        label=f"{curve_label}",
    )
    ax.set_title("Training Loss", fontsize=15, fontweight="bold")
    ax.set_xlabel("Global step")
    ax.set_ylabel("Loss")
    ax.legend()

    ax = axes[1]
    if show_raw:
        ax.plot(steps, accs, alpha=0.08, linewidth=0.8)
    ax.plot(
        steps,
        ema_smooth(accs, smoothing),
        linewidth=3.0,
        label=f"{curve_label}",
    )
    ax.set_title("Full Accuracy", fontsize=15, fontweight="bold")
    ax.set_xlabel("Global step")
    ax.set_ylabel("Accuracy")
    ax.legend()

    fig.suptitle(label, fontsize=17, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    print(f"[saved] {out}")


def plot_position_curves(records, label, out, smoothing=0.9, show_raw=False):
    by_pos = defaultdict(list)

    for r in records:
        for pos, acc in r["pos_acc"].items():
            by_pos[pos].append((r["step"], acc))

    set_plot_style()

    fig, ax = plt.subplots(figsize=(12, 6.5))

    for pos in sorted(by_pos):
        xs = [x for x, _ in by_pos[pos]]
        ys = [y for _, y in by_pos[pos]]

        if show_raw:
            ax.plot(xs, ys, alpha=0.08, linewidth=0.8)

        ax.plot(
            xs,
            ema_smooth(ys, smoothing),
            linewidth=1.8,
            label=f"position {pos}",
        )

    ax.set_title(
        "Per-Position Accuracy During Training",
        fontsize=15,
        fontweight="bold",
    )
    ax.set_xlabel("Global step")
    ax.set_ylabel("Accuracy")
    ax.legend(ncol=3, fontsize=9)

    fig.suptitle(label, fontsize=17, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    print(f"[saved] {out}")


def plot_position_at_step(records, target_step, label, out, percent=True):
    r = nearest_step(records, target_step)

    positions = sorted(r["pos_acc"])

    if percent:
        accs = [100.0 * r["pos_acc"][p] for p in positions]
        ylabel = "Accuracy (%)"
        value_fmt = "{:.1f}%"
        full_acc = 100.0 * r["full_acc"]
        full_acc_text = f"{full_acc:.2f}%"
    else:
        accs = [r["pos_acc"][p] for p in positions]
        ylabel = "Accuracy"
        value_fmt = "{:.3f}"
        full_acc_text = f"{r['full_acc']:.4f}"

    set_plot_style()

    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.bar(
        positions,
        accs,
        width=0.72,
        edgecolor="black",
        linewidth=0.8,
        alpha=0.9,
    )

    ymax = max(accs) if accs else 1.0
    label_offset = ymax * 0.015

    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + label_offset,
            value_fmt.format(acc),
            ha="center",
            va="bottom",
            fontsize=9,
            rotation=90,
        )

    ax.set_title(
        f"Per-Position Accuracy @ Global Step {r['step']}",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )

    ax.set_xlabel("Draft position", fontsize=13, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=13, fontweight="bold")
    ax.set_xticks(positions)

    ax.set_ylim(0, ymax * 1.22)

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.35,
    )

    subtitle = (
        f"{label} | "
        f"Loss = {r['loss']:.4f} | "
        f"Full Accuracy = {full_acc_text}"
    )

    fig.suptitle(subtitle, fontsize=12)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    print(f"[saved] {out}")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--log", required=True, help="Path to .log file")
    ap.add_argument("--label", default=None)
    ap.add_argument("--smoothing", type=float, default=0.9)
    ap.add_argument("--position-step", type=int, default=None)
    ap.add_argument("--raw", action=argparse.BooleanOptionalAction, default=True)

    ap.add_argument(
        "--position-fraction",
        action="store_true",
        help="Plot per-position accuracy as fractions instead of percentages",
    )

    ap.add_argument(
        "--curve-label",
        default="dsv4",
        help="loss and accuracy legend",
    )

    ap.add_argument(
        "--outdir",
        default="plots",
        help="Directory where all figures will be saved",
    )

    ap.add_argument(
        "--out-prefix",
        default="dflash",
        help="Prefix for output plot filenames",
    )

    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    label = args.label or os.path.splitext(os.path.basename(args.log))[0]
    records = parse_dflash_log(args.log)

    if not records:
        raise RuntimeError("No DFlash metrics found in log.")

    print(f"[ok] parsed {len(records)} metric points")
    print(f"[ok] steps {records[0]['step']}..{records[-1]['step']}")
    print(f"[ok] saving plots to: {os.path.abspath(args.outdir)}")

    plot_training(
        args.curve_label,
        records,
        label,
        os.path.join(args.outdir, f"{args.out_prefix}_loss_fullacc.png"),
        smoothing=args.smoothing,
        show_raw=args.raw,
    )

    plot_position_curves(
        records,
        label,
        os.path.join(args.outdir, f"{args.out_prefix}_position_curves.png"),
        smoothing=args.smoothing,
        show_raw=False,
    )

    if args.position_step is not None:
        plot_position_at_step(
            records,
            args.position_step,
            label,
            os.path.join(
                args.outdir,
                f"{args.out_prefix}_position_step_{args.position_step}.png",
            ),
            percent=not args.position_fraction,
        )


if __name__ == "__main__":
    main()