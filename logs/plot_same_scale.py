#!/usr/bin/env python3
"""Parse DFlash training logs (SpecForge + speculators) and plot loss / accuracy
curves with TensorBoard-style smoothing. Accepts ANY number of logs of either
kind; each becomes its own labelled curve. Missing files, unparsable logs, and
missing metrics are skipped (with a note) instead of crashing.

Different data-parallel degrees (DP=8 vs 4 vs 1) make `global_step` NON-comparable
across runs, so the default x-axis is EPOCH PROGRESS (fraction of the dataset seen),
which is DP-independent. Use --xaxis step to see raw optimizer steps instead.

Usage:
  python plot_dflash.py \
      --speculator colo=logs/colo.log sep=logs/sep.log \
      --specforge  sf=logs/specforge.log \
      --xaxis epoch --smoothing 0.9 --out dflash_compare.png

python plot_same_scale.py --speculator colocate=train_colocate.log separate=train_separate.log --specforge  sf=train_specforge.log --xaxis epoch --smoothing 0.9 --out dflash_compare.png
"""
import argparse
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_specforge(path):
    pat = re.compile(r"METRIC\s+(\{.*\})")
    recs = {}
    with open(path, errors="ignore") as f:
        for line in f:
            m = pat.search(line)
            if not m:
                continue
            try:
                d = json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
            if d.get("type") != "step":
                continue
            recs[int(d["step"])] = (d.get("loss"), d.get("accuracy"), d.get("epoch"))
    steps = sorted(recs)
    return (steps, [recs[s][0] for s in steps],
            [recs[s][1] for s in steps], [recs[s][2] for s in steps])


def parse_speculator(path):
    with open(path, errors="ignore") as f:
        text = f.read()
    loss_pat = re.compile(r"train/loss=([-+0-9.eE]+)")
    acc_pat = re.compile(r"train/full_acc=([-+0-9.eE]+)")
    step_pat = re.compile(r"global_step=(\d+)")
    ep_pat = re.compile(r"epoch=(\d+)")
    recs = {}
    starts = [m.start() for m in re.finditer(r"train/loss=", text)]
    starts.append(len(text))
    for i in range(len(starts) - 1):
        chunk = text[starts[i]:starts[i + 1]]
        sm = step_pat.search(chunk)
        if not sm:
            continue
        step = int(sm.group(1))
        lm, am, em = loss_pat.search(chunk), acc_pat.search(chunk), ep_pat.search(chunk)
        recs[step] = (float(lm.group(1)) if lm else None,
                      float(am.group(1)) if am else None,
                      int(em.group(1)) if em else None)
    steps = sorted(recs)
    return (steps, [recs[s][0] for s in steps],
            [recs[s][1] for s in steps], [recs[s][2] for s in steps])


def ema_smooth(values, weight):
    if weight <= 0 or len(values) == 0:
        return list(values)
    out, last, n = [], 0.0, 0
    for v in values:
        last = last * weight + (1.0 - weight) * v
        n += 1
        out.append(last / (1.0 - weight ** n))
    return out


def collect(entries, parser, fw):
    runs = []
    for entry in entries:
        if "=" in entry and not os.path.exists(entry):
            label, path = entry.split("=", 1)
        else:
            path = entry
            label = os.path.splitext(os.path.basename(path))[0]
        try:
            steps, loss, acc, ep = parser(path)
        except FileNotFoundError:
            print(f"[skip] {fw} '{path}': file not found"); continue
        except Exception as e:
            print(f"[skip] {fw} '{path}': parse error ({e})"); continue
        has_loss = any(v is not None for v in loss)
        has_acc = any(v is not None for v in acc)
        if not steps or (not has_loss and not has_acc):
            print(f"[skip] {fw} '{path}': no loss or accuracy found"); continue
        runs.append({"fw": fw, "label": label, "steps": steps,
                     "loss": loss, "acc": acc, "epoch": ep})
        print(f"[ok]   {fw:11s} '{label}': {len(steps)} pts "
              f"(steps {steps[0]}..{steps[-1]}, loss={has_loss}, acc={has_acc})")
    return runs


def compute_x(run, mode):
    steps = run["steps"]
    if mode == "step" or not steps:
        return list(steps)
    # epoch-progress: x in units of epochs (fraction of dataset seen), DP-independent
    max_step = steps[-1]
    ep = [e for e in run["epoch"] if e is not None]
    num_epochs = (max(ep) + 1) if ep else 1
    spe = max_step / num_epochs if num_epochs else max_step  # steps per epoch
    if not spe:
        return list(steps)
    return [s / spe for s in steps]


def _plot_metric(ax, run, key, color, ls, smoothing, show_raw):
    pts = [(x, v) for x, v in zip(run["x"], run[key]) if v is not None]
    if not pts:
        return False
    xs, ys = zip(*pts)
    if show_raw:
        ax.plot(xs, ys, color=color, alpha=0.12, linewidth=1.0, linestyle=ls)
    ax.plot(xs, ema_smooth(ys, smoothing), color=color, alpha=0.95,
            linewidth=2.0, linestyle=ls, label=f"{run['label']} \u00b7 {run['fw']}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specforge", nargs="*", default=[])
    ap.add_argument("--speculator", nargs="*", default=[])
    ap.add_argument("--xaxis", choices=["epoch", "step"], default="epoch",
                    help="epoch=fraction of dataset seen (DP-independent, default); "
                         "step=raw global_step (NOT comparable across different DP)")
    ap.add_argument("--smoothing", type=float, default=0.9)
    ap.add_argument("--raw", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--title", default=None)
    ap.add_argument("--out", default="dflash_compare.png")
    args = ap.parse_args()

    runs = (collect(args.specforge, parse_specforge, "SpecForge")
            + collect(args.speculator, parse_speculator, "speculators"))
    if not runs:
        ap.error("nothing to plot: no usable logs given")
    for run in runs:
        run["x"] = compute_x(run, args.xaxis)

    n = len(runs)
    cmap = plt.cm.tab10 if n <= 10 else plt.cm.tab20
    colors = [cmap(i % cmap.N) for i in range(n)]

    plt.rcParams.update({
        "figure.dpi": 130, "font.size": 11,
        "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(14, 5))

    drew_loss = drew_acc = False
    for run, color in zip(runs, colors):
        ls = "--" if run["fw"] == "SpecForge" else "-"
        drew_loss |= _plot_metric(ax_loss, run, "loss", color, ls, args.smoothing, args.raw)
        drew_acc |= _plot_metric(ax_acc, run, "acc", color, ls, args.smoothing, args.raw)

    xlabel = "epoch progress (fraction of dataset seen)" if args.xaxis == "epoch" else "global step"
    ax_loss.set(title="Training loss", xlabel=xlabel, ylabel="loss")
    ax_acc.set(title="Training accuracy", xlabel=xlabel, ylabel="accuracy")
    if drew_loss:
        ax_loss.legend(frameon=False, fontsize=9)
    if drew_acc:
        ax_acc.legend(frameon=False, fontsize=9)
    fig.suptitle(args.title or f"DFlash training  (x={args.xaxis}, smoothing={args.smoothing}, "
                               "solid=speculators, dashed=SpecForge)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(args.out, bbox_inches="tight")
    print(f"saved {args.out}  ({n} run(s), x-axis={args.xaxis})")


if __name__ == "__main__":
    main()
