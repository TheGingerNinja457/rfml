#!/usr/bin/env python3
"""
train_cnn.py - deep model on raw IQ for the 2.4 GHz RF source classifier (Step 4)

Two-branch 1D CNN:
  time branch      raw I/Q (2 x 1024), residual conv stack
  frequency branch log-power spectrum of the same window (1 x 1024), conv stack
  -> concatenated -> MLP head -> 9-class softmax (collapsed to 5 groups at eval)

Built for a 6 GB laptop GPU (RTX 3060 mobile):
  * the whole dataset lives on the GPU as float16 (~0.5 GB) - no DataLoader,
    no Windows multiprocessing problems, GPU never waits for data
  * mixed precision, AdamW, one-cycle LR, label smoothing
  * on-the-fly augmentation: random phase, small CFO, circular time shift,
    +-1 dB gain, optional extra noise, random channel (multipath) - all on GPU
  * early stopping on val macro-F1, best checkpoint saved

Evaluates exactly like baseline.py (test / test_real / test_unseen), writes into
results/cnn/, and if results/baseline/metrics.json exists also draws the
baseline-vs-CNN comparison plot.

Usage (PowerShell, venv active):
  python train_cnn.py
  python train_cnn.py --epochs 40 --batch 512 --width 64
  python train_cnn.py --eval-only            (re-evaluate the saved best model)
"""
import argparse
import json
import math
import time
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from jam_metrics import binary_eval, plot_binary, plot_pfa_by_class

GROUPS = ["clean", "wifi", "bluetooth", "other_comm", "interference", "jamming"]


# ------------------------------------------------------------------ data
def load_h5(path):
    with h5py.File(path, "r") as f:
        d = {k: f[k][:] for k in ("iq", "label", "split", "snr_db", "jnr_db")}
        d["source"] = f["source"][:].astype(str)
        meta = {k: json.loads(f.attrs[k]) for k in ("classes", "splits", "group_of")}
        meta["fs"] = float(f.attrs["fs"])
    return d, meta


# ----------------------------------------------------------------- model
class ResBlock(nn.Module):
    def __init__(self, cin, cout, k=7, stride=1):
        super().__init__()
        self.c1 = nn.Conv1d(cin, cout, k, stride, k // 2, bias=False)
        self.b1 = nn.BatchNorm1d(cout)
        self.c2 = nn.Conv1d(cout, cout, k, 1, k // 2, bias=False)
        self.b2 = nn.BatchNorm1d(cout)
        self.skip = (nn.Sequential(nn.Conv1d(cin, cout, 1, stride, bias=False), nn.BatchNorm1d(cout))
                     if (cin != cout or stride != 1) else nn.Identity())

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(y + self.skip(x))


def branch(cin, w):
    return nn.Sequential(
        nn.Conv1d(cin, w, 7, 1, 3, bias=False), nn.BatchNorm1d(w), nn.ReLU(),
        ResBlock(w, w, 7, 2),          # 512
        ResBlock(w, 2 * w, 5, 2),      # 256
        ResBlock(2 * w, 2 * w, 5, 2),  # 128
        ResBlock(2 * w, 4 * w, 3, 2),  # 64
        ResBlock(4 * w, 4 * w, 3, 2),  # 32
    )


class RFNet(nn.Module):
    def __init__(self, n_classes, w=48, dropout=0.3):
        super().__init__()
        self.time = branch(2, w)
        self.freq = branch(1, w // 2)
        feat = 4 * w * 2 + 4 * (w // 2) * 2  # avg + max pool for each branch
        self.head = nn.Sequential(nn.Linear(feat, 256), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(256, n_classes))

    @staticmethod
    def spectrum(iq):
        z = torch.complex(iq[:, 0].float(), iq[:, 1].float())
        win = torch.hann_window(z.shape[-1], device=z.device)
        P = torch.fft.fftshift(torch.fft.fft(z * win), dim=-1).abs() ** 2 / z.shape[-1]
        return (10 * torch.log10(P + 1e-6)).unsqueeze(1) / 10.0  # ~unit scale

    def forward(self, iq):
        t = self.time(iq.float())
        s = self.freq(self.spectrum(iq))
        pool = lambda h: torch.cat([h.mean(-1), h.amax(-1)], 1)
        return self.head(torch.cat([pool(t), pool(s)], 1))


# ---------------------------------------------------------- augmentation
def augment(x, fs, p_noise=0.3, p_mp=0.3, cfo_hz=200e3):
    """x: (B, 2, N) float on GPU. Noise floor = power 1, so keep gain changes small."""
    B, _, N = x.shape
    dev = x.device
    z = torch.complex(x[:, 0].float(), x[:, 1].float())
    n = torch.arange(N, device=dev)
    phase = torch.rand(B, 1, device=dev) * 2 * math.pi
    cfo = (torch.rand(B, 1, device=dev) * 2 - 1) * cfo_hz
    z = z * torch.exp(1j * (phase + 2 * math.pi * cfo * n / fs))
    shift = int(torch.randint(0, N, (1,)))
    z = torch.roll(z, shift, dims=-1)
    z = z * 10 ** ((torch.rand(B, 1, device=dev) * 2 - 1) / 20)  # +-1 dB
    if p_mp > 0:  # short random multipath on a subset
        m = torch.rand(B, device=dev) < p_mp
        if m.any():
            k = int(m.sum())
            taps = torch.randn(k, 3, dtype=torch.cfloat, device=dev) * torch.tensor([1.0, 0.4, 0.2], device=dev)
            taps = taps / taps.abs().pow(2).sum(1, keepdim=True).sqrt()
            zm = z[m]
            z[m] = taps[:, :1] * zm + taps[:, 1:2] * torch.roll(zm, 1, -1) + taps[:, 2:3] * torch.roll(zm, 2, -1)
    if p_noise > 0:  # raise the floor by 0-3 dB on a subset (harder, more varied SNR)
        m = (torch.rand(B, 1, device=dev) < p_noise).float()
        extra = m * (10 ** (torch.rand(B, 1, device=dev) * 0.3) - 1)
        z = z + torch.sqrt(extra / 2) * torch.complex(torch.randn(B, N, device=dev), torch.randn(B, N, device=dev))
    return torch.stack([z.real, z.imag], 1)


# -------------------------------------------------------------- helpers
@torch.no_grad()
def predict(model, X, batch=2048, amp=True):
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        with torch.autocast(X.device.type, dtype=torch.float16, enabled=amp and X.device.type == "cuda"):
            out.append(model(X[i:i + batch]).float().softmax(-1))
    return torch.cat(out).cpu().numpy()


def plot_cm(y_true, y_pred, labels, title, path):
    cols = [l for l in labels if l in set(y_true) | set(y_pred)]
    rows = [l for l in cols if l in set(y_true)]
    cm = confusion_matrix(y_true, y_pred, labels=cols, normalize="true")
    cm = cm[[cols.index(r) for r in rows]]
    fig, ax = plt.subplots(figsize=(1.0 + 0.8 * len(cols), 0.8 + 0.7 * len(rows)))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(len(rows)):
        for j in range(len(cols)):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if cm[i, j] > 0.6 else "black")
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(rows)), rows)
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/rf24.h5")
    ap.add_argument("--out", default="results/cnn")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--width", type=int, default=48)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("!! CUDA not available - running on CPU (slow). Check your torch install.")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    d, meta = load_h5(args.data)
    classes_all, splits = meta["classes"], meta["splits"]
    present = sorted(set(d["label"].tolist()))
    classes = [classes_all[i] for i in present]          # only classes in this dataset
    remap = np.full(len(classes_all), -1)
    remap[present] = np.arange(len(present))
    y_all = remap[d["label"]]
    split_name = np.array(splits)[d["split"]]
    print(f"{len(y_all)} windows, classes: {classes}")

    X_all = torch.from_numpy(d["iq"]).to(dev)            # float16, (N, 2, 1024)
    idx = {s: np.where(split_name == s)[0] for s in splits if (split_name == s).any()}
    Xtr, ytr = X_all[idx["train"]], torch.from_numpy(y_all[idx["train"]]).long().to(dev)
    Xva, yva = X_all[idx["val"]], y_all[idx["val"]]

    model = RFNet(len(classes), w=args.width).to(dev)
    print(f"model params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    ckpt = out / "best.pt"

    if not args.eval_only:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
        steps = math.ceil(len(Xtr) / args.batch)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, epochs=args.epochs,
                                                    steps_per_epoch=steps, pct_start=0.15)
        scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
        hist, best_f1, bad = [], -1.0, 0
        for ep in range(1, args.epochs + 1):
            model.train()
            t0 = time.time()
            perm = torch.randperm(len(Xtr), device=dev)
            tot, correct, loss_sum = 0, 0, 0.0
            for i in range(steps):
                b = perm[i * args.batch:(i + 1) * args.batch]
                xb, yb = Xtr[b].float(), ytr[b]
                if not args.no_augment:
                    xb = augment(xb, meta["fs"])
                with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                    logits = model(xb)
                    loss = F.cross_entropy(logits.float(), yb, label_smoothing=0.05)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                loss_sum += loss.item() * len(b)
                correct += (logits.argmax(1) == yb).sum().item()
                tot += len(b)
            pv = predict(model, Xva).argmax(1)
            va_acc, va_f1 = accuracy_score(yva, pv), f1_score(yva, pv, average="macro")
            hist.append({"epoch": ep, "train_loss": loss_sum / tot, "train_acc": correct / tot,
                         "val_acc": va_acc, "val_macro_f1": va_f1, "lr": sched.get_last_lr()[0],
                         "sec": time.time() - t0})
            flag = ""
            if va_f1 > best_f1:
                best_f1, bad, flag = va_f1, 0, " *"
                torch.save({"model": model.state_dict(), "classes": classes, "args": vars(args)}, ckpt)
            else:
                bad += 1
            print(f"ep {ep:3d}  loss {loss_sum / tot:.4f}  train acc {correct / tot:.4f}  "
                  f"val acc {va_acc:.4f}  val F1 {va_f1:.4f}  ({time.time() - t0:.0f}s){flag}")
            if dev.type == "cuda" and ep == 1:
                print(f"   peak GPU memory {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
            if bad >= args.patience:
                print(f"early stop: no val improvement for {args.patience} epochs")
                break
        h = pd.DataFrame(hist)
        h.to_csv(out / "history.csv", index=False)
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
        ax[0].plot(h.epoch, h.train_loss)
        ax[0].set_title("training loss")
        ax[0].set_xlabel("epoch")
        ax[1].plot(h.epoch, h.train_acc, label="train (augmented)")
        ax[1].plot(h.epoch, h.val_acc, label="val")
        ax[1].set_title("accuracy")
        ax[1].set_xlabel("epoch")
        ax[1].legend()
        for a in ax:
            a.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "training_curves.png", dpi=150)
        plt.close(fig)

    # ------------------------------------------------------------ evaluate
    state = torch.load(ckpt, map_location=dev)
    model.load_state_dict(state["model"])
    group_of = meta["group_of"]
    metrics = {"val_macro_f1": float(f1_score(yva, predict(model, Xva).argmax(1), average="macro"))}

    # inference time per window: batch of 1024 and single window
    xb = Xva[:1024]
    predict(model, xb)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(5):
        predict(model, xb)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    metrics["inference_us_per_window"] = (time.perf_counter() - t0) / (5 * len(xb)) * 1e6

    report = []
    eval_splits = [s for s in ("test", "test_real", "test_unseen") if s in idx]
    for split in eval_splits:
        ii = idx[split]
        probs = predict(model, X_all[ii])
        pred = np.array(classes)[probs.argmax(1)]
        jam_score = probs[:, [i for i, c in enumerate(classes) if c.startswith("jam")]].sum(1)
        true = np.array(classes_all)[d["label"][ii]]
        level = np.where(np.char.startswith(true.astype(str), "jam"), d["jnr_db"][ii], d["snr_db"][ii])
        gt = np.array([group_of[c] for c in true])
        gp = np.array([group_of[c] for c in pred])
        metrics[split] = {
            "n": int(len(ii)),
            "accuracy": float(accuracy_score(true, pred)),
            "macro_f1": float(f1_score(true, pred, labels=sorted(set(true)), average="macro", zero_division=0)),
            "group_accuracy": float(accuracy_score(gt, gp)),
            "binary": binary_eval(true, pred, jam_score, level if split == "test" else None),
        }
        plot_cm(true, pred, classes, f"CNN - {split}", out / f"confusion_cnn_{split}.png")
        plot_cm(gt, gp, GROUPS, f"CNN - {split} (groups)", out / f"confusion_cnn_{split}_groups.png")
        report.append(f"===== {split} =====\n" + classification_report(true, pred, zero_division=0))
        if split == "test":
            curve = pd.DataFrame({"level": level, "correct": pred == true}).dropna()
            curve["bin"] = pd.cut(curve["level"], np.arange(-6, 28, 2))
            acc = curve.groupby("bin", observed=True)["correct"].mean()
            metrics["acc_vs_level"] = {float(b.mid): float(v) for b, v in acc.items()}
    (out / "report_cnn.txt").write_text("\n".join(report))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))

    # comparison with the baseline, if it has been run
    base_path = Path("results/baseline/metrics.json")
    def bin_cols(s, splits_):
        return {f"{sp}_{k}": s[sp]["binary"][k] for sp in splits_ if sp in s and "binary" in s[sp]
                for k in ("pd", "pfa", "auc") if k in s[sp]["binary"]}

    rows = [{"model": "cnn", **{f"{s}_{k}": metrics[s][k] for s in eval_splits
                                for k in ("accuracy", "macro_f1", "group_accuracy")},
             **bin_cols(metrics, eval_splits),
             "inference_us_per_window": metrics["inference_us_per_window"]}]
    binary_all = {}
    fig, ax = plt.subplots(figsize=(7, 4.2))
    if base_path.exists():
        base = json.loads(base_path.read_text())
        for name, s in base.items():
            if "acc_vs_level" in s:
                xs, ys = zip(*sorted((float(k), v) for k, v in s["acc_vs_level"].items()))
                ax.plot(xs, ys, marker="o", ms=3, lw=1, alpha=0.6, label=name)
            rows.append({"model": name, **{f"{sp}_{k}": s[sp][k] for sp in eval_splits if sp in s
                                           for k in ("accuracy", "macro_f1", "group_accuracy")},
                         **bin_cols(s, eval_splits),
                         "inference_us_per_window": s.get("inference_us_per_window")})
            if "test" in s and "binary" in s["test"]:
                binary_all[name] = s["test"]["binary"]
    xs, ys = zip(*sorted(metrics["acc_vs_level"].items()))
    ax.plot(xs, ys, marker="o", ms=4, lw=2.2, color="black", label="cnn")
    ax.set_xlabel("SNR (comm signals) / JNR (jammers), dB")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("Accuracy vs signal strength (synthetic test)")
    fig.tight_layout()
    fig.savefig(out / "acc_vs_level_comparison.png", dpi=150)
    plt.close(fig)

    binary_all["cnn"] = metrics["test"]["binary"]
    plot_binary(binary_all, out / "jamming_detection_comparison.png")
    plot_pfa_by_class(binary_all, out / "false_alarms_by_class.png")
    b = metrics["test"]["binary"]
    print(f"\nCNN jamming detection (synthetic test): Pd {b['pd']:.3f}  Pfa {b['pfa']:.3f}  AUC {b['auc']:.3f}  "
          f"| Pd at 1%/5% false alarms: {b['pd_at_pfa']['0.01']:.3f} / {b['pd_at_pfa']['0.05']:.3f}")
    print("  false alarms by benign class: " + "  ".join(f"{c} {100 * v:.1f}%" for c, v in b["pfa_by_class"].items()))
    print("  detection by jammer type:     " + "  ".join(f"{c} {100 * v:.1f}%" for c, v in b["pd_by_class"].items()))
    if "test_real" in metrics:
        print(f"  real Wi-Fi/BT false-alarm rate: {100 * metrics['test_real']['binary']['pfa']:.1f}%")
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "comparison.csv", index=False)
    pd.set_option("display.width", 220)
    print("\n" + summary.round(3).to_string(index=False))
    print(f"\nall outputs in {out.resolve()}")


if __name__ == "__main__":
    main()
