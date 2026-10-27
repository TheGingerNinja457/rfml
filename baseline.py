#!/usr/bin/env python3
"""
baseline.py - classic ML baseline on the hand-crafted features from build_dataset.py

Trains Random Forest, XGBoost and RBF-SVM on data/rf24.features.csv, picks
hyperparameters on the val split, then reports on:
  test        held-out synthetic windows
  test_real   real FKIE Wi-Fi / Bluetooth bursts (never seen in training)
  test_unseen held-out Alhazbi jammer (only if you built with Alhazbi)

Outputs go to results/baseline/:
  metrics.json, summary.csv
  confusion_<model>_<split>.png   (fine classes + 5-group version)
  acc_vs_level.png                accuracy vs SNR/JNR for each model
  feature_importance.png          XGBoost gain + permutation importance
  report_<model>.txt              per-class precision/recall/F1

Usage (PowerShell, venv active):
  python baseline.py
  python baseline.py --features data\\rf24.features.csv --svm-max 20000
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from jam_metrics import binary_eval, plot_binary, plot_pfa_by_class

CLASSES = ["noise", "wifi_ofdm", "wifi_dsss", "bluetooth", "bpsk_link",
           "jam_tone", "jam_barrage", "jam_sweep", "jam_pulsed", "wifi_congestion"]
GROUP_OF = {"noise": "clean", "wifi_ofdm": "wifi", "wifi_dsss": "wifi",
            "bluetooth": "bluetooth", "bpsk_link": "other_comm",
            "jam_tone": "jamming", "jam_barrage": "jamming",
            "jam_sweep": "jamming", "jam_pulsed": "jamming",
            "wifi_congestion": "interference"}
GROUPS = ["clean", "wifi", "bluetooth", "other_comm", "interference", "jamming"]
META = ["idx", "label", "group", "split", "source", "snr_db", "jnr_db"]


def load(path):
    df = pd.read_csv(path)
    feats = [c for c in df.columns if c not in META]
    df[feats] = df[feats].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feats)
    present = [c for c in CLASSES if c in set(df["label"])]
    df["y"] = df["label"].map({c: i for i, c in enumerate(present)})
    # "level" for the accuracy-vs-strength plot: JNR for jammers, SNR for comm signals
    df["level"] = np.where(df["label"].str.startswith("jam"), df["jnr_db"], df["snr_db"])
    return df, feats, present


def models(seed, n_classes):
    return {
        "random_forest": [
            RandomForestClassifier(n_estimators=300, max_depth=d, min_samples_leaf=2,
                                   n_jobs=-1, random_state=seed)
            for d in (None, 20)],
        "xgboost": [
            XGBClassifier(n_estimators=400, max_depth=d, learning_rate=lr, subsample=0.8,
                          colsample_bytree=0.8, tree_method="hist", n_jobs=-1,
                          objective="multi:softprob", num_class=n_classes,
                          eval_metric="mlogloss", random_state=seed)
            for d, lr in ((6, 0.1), (8, 0.05))],
        "svm_rbf": [
            make_pipeline(StandardScaler(), SVC(C=c, gamma="scale", random_state=seed))
            for c in (3.0, 30.0)],
    }


def plot_cm(y_true, y_pred, labels, title, path, normalize=True):
    cols = [l for l in labels if l in set(y_true) | set(y_pred)]
    rows = [l for l in cols if l in set(y_true)]  # only classes that occur in this split
    cm = confusion_matrix(y_true, y_pred, labels=cols, normalize="true" if normalize else None)
    cm = cm[[cols.index(r) for r in rows]]
    fig, ax = plt.subplots(figsize=(1.0 + 0.8 * len(cols), 0.8 + 0.7 * len(rows)))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalize else None, aspect="auto")
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = cm[i, j]
            ax.text(j, i, f"{v:.2f}" if normalize else f"{v:d}", ha="center", va="center",
                    color="white" if (v > 0.6 if normalize else v > cm.max() * 0.6) else "black", fontsize=9)
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(rows)), rows)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default="data/rf24.features.csv")
    ap.add_argument("--out", default="results/baseline")
    ap.add_argument("--svm-max", type=int, default=20000,
                    help="max training rows for the SVM (it scales ~quadratically)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df, feats, classes = load(args.features)
    print(f"{len(df)} windows, {len(feats)} features, classes: {classes}")
    print(df.groupby("split").size().to_string(), "\n")

    tr, va = df[df.split == "train"], df[df.split == "val"]
    eval_splits = [s for s in ("test", "test_real", "test_unseen") if (df.split == s).any()]
    Xtr, ytr = tr[feats].to_numpy(), tr["y"].to_numpy()
    Xva, yva = va[feats].to_numpy(), va["y"].to_numpy()
    rng = np.random.default_rng(args.seed)

    metrics, best = {}, {}
    for name, candidates in models(args.seed, len(classes)).items():
        Xfit, yfit = Xtr, ytr
        if name == "svm_rbf" and len(Xtr) > args.svm_max:
            sel = rng.choice(len(Xtr), args.svm_max, replace=False)
            Xfit, yfit = Xtr[sel], ytr[sel]
        scores = []
        for m in candidates:
            t0 = time.time()
            m.fit(Xfit, yfit)
            f1 = f1_score(yva, m.predict(Xva), average="macro")
            scores.append(f1)
            print(f"  {name:13s} {str(m.get_params().get('max_depth', m.get_params().get('svc__C', ''))):>5s}"
                  f"  val macro-F1 {f1:.4f}  ({time.time() - t0:.0f}s)")
        m = candidates[int(np.argmax(scores))]
        best[name] = m
        metrics[name] = {"val_macro_f1": float(max(scores))}

        # inference time per window (single-threaded-ish batch of 2000)
        Xb = Xva[:2000]
        t0 = time.perf_counter()
        m.predict(Xb)
        metrics[name]["inference_us_per_window"] = (time.perf_counter() - t0) / len(Xb) * 1e6

        report_lines = []
        for split in eval_splits:
            d = df[df.split == split]
            Xs = d[feats].to_numpy()
            if hasattr(m, "predict_proba"):
                proba = m.predict_proba(Xs)
                pred = np.array(classes)[proba.argmax(1)]
                jam_score = proba[:, [i for i, c in enumerate(classes) if c.startswith("jam")]].sum(1)
            else:  # SVC without probabilities: hard decisions only (no ROC)
                pred = np.array(classes)[m.predict(Xs)]
                jam_score = None
            true = d["label"].to_numpy()
            g_true = np.array([GROUP_OF[c] for c in true])
            g_pred = np.array([GROUP_OF[c] for c in pred])
            metrics[name][split] = {
                "n": int(len(d)),
                "accuracy": float(accuracy_score(true, pred)),
                # macro-F1 over the classes actually present in this split
                "macro_f1": float(f1_score(true, pred, labels=sorted(set(true)), average="macro",
                                           zero_division=0)),
                "group_accuracy": float(accuracy_score(g_true, g_pred)),
                "binary": binary_eval(true, pred, jam_score,
                                      d["level"].to_numpy() if split == "test" else None),
            }
            plot_cm(true, pred, classes, f"{name} - {split}", out / f"confusion_{name}_{split}.png")
            plot_cm(g_true, g_pred, GROUPS,
                    f"{name} - {split} (groups)", out / f"confusion_{name}_{split}_groups.png")
            report_lines.append(f"===== {split} =====\n" + classification_report(true, pred, zero_division=0))
            if split == "test":
                d = d.assign(correct=(pred == true))
                metrics[name]["_curve"] = d
        (out / f"report_{name}.txt").write_text("\n".join(report_lines))
        s = metrics[name]
        print(f"  -> {name}: " + "  ".join(f"{sp} acc {s[sp]['accuracy']:.3f}" for sp in eval_splits)
              + f"  | {s['inference_us_per_window']:.1f} us/window")
        b = s["test"]["binary"]
        print(f"     jamming detection (test): Pd {b.get('pd', float('nan')):.3f}  "
              f"Pfa {b.get('pfa', float('nan')):.3f}" + (f"  AUC {b['auc']:.3f}" if "auc" in b else "") + "\n")

    # accuracy vs SNR/JNR on the synthetic test split
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bins = np.arange(-6, 28, 2)
    for name in best:
        d = metrics[name].pop("_curve")
        d = d[d["level"].notna()]
        d["bin"] = pd.cut(d["level"], bins)
        acc = d.groupby("bin", observed=True)["correct"].mean()
        centres = [b.mid for b in acc.index]
        ax.plot(centres, acc.values, marker="o", ms=3, label=name)
        metrics[name]["acc_vs_level"] = dict(zip(map(float, centres), map(float, acc.values)))
    ax.set_xlabel("SNR (comm signals) / JNR (jammers), dB")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("Baseline accuracy vs signal strength (synthetic test)")
    fig.tight_layout()
    fig.savefig(out / "acc_vs_level.png", dpi=150)
    plt.close(fig)

    # feature importance: XGBoost gain + permutation (on val)
    xgb = best["xgboost"]
    gain = xgb.feature_importances_
    perm = permutation_importance(xgb, Xva[:5000], yva[:5000], n_repeats=5,
                                  random_state=args.seed, scoring="f1_macro", n_jobs=-1)
    order = np.argsort(perm.importances_mean)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    axes[0].barh(np.array(feats)[order], gain[order])
    axes[0].set_title("XGBoost gain")
    axes[1].barh(np.array(feats)[order], perm.importances_mean[order], xerr=perm.importances_std[order])
    axes[1].set_title("Permutation importance (drop in macro-F1)")
    fig.tight_layout()
    fig.savefig(out / "feature_importance.png", dpi=150)
    plt.close(fig)

    # jamming-vs-not plots
    plot_binary({n: metrics[n]["test"]["binary"] for n in best}, out / "jamming_detection.png",
                "Baseline jamming detection (synthetic test)")
    plot_pfa_by_class({n: metrics[n]["test"]["binary"] for n in best}, out / "false_alarms_by_class.png")

    # summary
    rows = []
    for name, s in metrics.items():
        row = {"model": name, "val_macro_f1": s["val_macro_f1"],
               "inference_us_per_window": s["inference_us_per_window"]}
        for sp in eval_splits:
            for k in ("accuracy", "macro_f1", "group_accuracy"):
                row[f"{sp}_{k}"] = s[sp][k]
            for k in ("pd", "pfa", "auc"):
                if k in s[sp]["binary"]:
                    row[f"{sp}_{k}"] = s[sp]["binary"][k]
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    pd.set_option("display.width", 200)
    print(summary.round(3).to_string(index=False))
    print(f"\nall outputs in {out.resolve()}")


if __name__ == "__main__":
    main()
