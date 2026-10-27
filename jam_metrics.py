"""
jam_metrics.py - binary "intentional (jamming) vs unintentional / benign" evaluation

Used by baseline.py and train_cnn.py. Collapses the multi-class prediction to
the research question: is this window jammed or not?

  Pd   probability of detection  = jammed windows flagged as jamming
  Pfa  probability of false alarm = benign windows (clean, Wi-Fi, Bluetooth,
                                     congestion, ...) flagged as jamming
"""
import numpy as np


def is_jam(labels):
    return np.char.startswith(np.asarray(labels).astype(str), "jam")


def binary_eval(true, pred, score=None, level=None, bins=np.arange(-6, 28, 2)):
    """true/pred: class-name arrays. score: P(jamming) per window (optional).
    level: SNR for benign signals / JNR for jammers (optional, for curves)."""
    true, pred = np.asarray(true).astype(str), np.asarray(pred).astype(str)
    tj, pj = is_jam(true), is_jam(pred)
    out = {"n_jam": int(tj.sum()), "n_benign": int((~tj).sum())}
    if tj.any():
        out["pd"] = float(pj[tj].mean())
        out["pd_by_class"] = {c: float(pj[true == c].mean()) for c in sorted(set(true[tj]))}
    if (~tj).any():
        out["pfa"] = float(pj[~tj].mean())
        out["pfa_by_class"] = {c: float(pj[true == c].mean()) for c in sorted(set(true[~tj]))}
    if tj.any() and (~tj).any():
        tp, fp = (pj & tj).sum(), (pj & ~tj).sum()
        out["precision"] = float(tp / max(tp + fp, 1))
        out["balanced_acc"] = float((out["pd"] + 1 - out["pfa"]) / 2)
    if score is not None and tj.any() and (~tj).any():
        from sklearn.metrics import roc_auc_score, roc_curve
        score = np.asarray(score, float)
        out["auc"] = float(roc_auc_score(tj, score))
        fpr, tpr, _ = roc_curve(tj, score)
        keep = np.unique(np.linspace(0, len(fpr) - 1, min(300, len(fpr))).astype(int))
        out["roc"] = {"fpr": fpr[keep].round(5).tolist(), "tpr": tpr[keep].round(5).tolist()}
        # Pd at fixed false-alarm rates - the operating points a deployment would pick
        out["pd_at_pfa"] = {f"{p:g}": float(np.interp(p, fpr, tpr)) for p in (0.01, 0.05, 0.10)}
    if level is not None:
        level = np.asarray(level, float)
        centres = [(a + b) / 2 for a, b in zip(bins[:-1], bins[1:])]
        dig = np.digitize(level, bins) - 1
        pd_c, pfa_c = {}, {}
        for i, c in enumerate(centres):
            m = dig == i
            if (m & tj).sum() >= 10:
                pd_c[c] = float(pj[m & tj].mean())
            if (m & ~tj).sum() >= 10:
                pfa_c[c] = float(pj[m & ~tj].mean())
        out["pd_vs_jnr"], out["pfa_vs_snr"] = pd_c, pfa_c
    return out


def plot_binary(results, path, title="Jamming detection (synthetic test)"):
    """results: {model_name: binary_eval dict (test split)}. Draws Pd vs JNR,
    Pfa vs SNR and the ROC curve side by side."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    for name, r in results.items():
        lw, col = (2.2, "black") if name == "cnn" else (1.2, None)
        if r.get("pd_vs_jnr"):
            x, y = zip(*sorted((float(k), v) for k, v in r["pd_vs_jnr"].items()))
            ax[0].plot(x, y, marker="o", ms=3, lw=lw, color=col, label=name)
        if r.get("pfa_vs_snr"):
            x, y = zip(*sorted((float(k), v) for k, v in r["pfa_vs_snr"].items()))
            ax[1].plot(x, y, marker="o", ms=3, lw=lw, color=col, label=name)
        if r.get("roc"):
            ax[2].plot(r["roc"]["fpr"], r["roc"]["tpr"], lw=lw, color=col,
                       label=f"{name} (AUC {r['auc']:.3f})")
    ax[0].set(xlabel="JNR, dB", ylabel="P(detect)", title="Detection rate vs jammer strength", ylim=(0, 1.02))
    ax[1].set(xlabel="SNR of benign signal, dB", ylabel="P(false alarm)",
              title="False alarms on benign signals", ylim=(0, None))
    ax[2].plot([0, 1], [0, 1], ls=":", color="grey", lw=1)
    ax[2].set(xlabel="P(false alarm)", ylabel="P(detect)", title="ROC: jamming vs not", xlim=(0, 1), ylim=(0, 1.02))
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pfa_by_class(results, path, title="False-alarm rate by benign signal type"):
    """Grouped bars: which benign signals get mistaken for jamming."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    classes = sorted({c for r in results.values() for c in r.get("pfa_by_class", {})})
    if not classes:
        return
    names = list(results)
    w = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(1.4 + 1.1 * len(classes), 4))
    for i, n in enumerate(names):
        v = [100 * results[n].get("pfa_by_class", {}).get(c, np.nan) for c in classes]
        ax.bar(np.arange(len(classes)) + (i - (len(names) - 1) / 2) * w, v, w, label=n,
               color="black" if n == "cnn" else None)
    ax.set_xticks(range(len(classes)), classes, rotation=20, ha="right")
    ax.set_ylabel("% flagged as jamming")
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
