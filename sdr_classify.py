#!/usr/bin/env python3
"""
sdr_classify.py - run the trained CNN on real IQ from a Pluto / AD9363 SDR

Commands:
  check      confirm the SDR is reachable and print its settings
  capture    record IQ to a file (contiguous buffers at 20 MS/s)
  classify   classify a capture file -> CSV, summary table, report figure
  live       capture + classify continuously, print a rolling summary

Typical session (PowerShell, venv active, SDR on USB):
  python sdr_classify.py check
  python sdr_classify.py capture --channel 6 --seconds 5 --out captures\\ch6_idle.npz
  python sdr_classify.py classify captures\\ch6_idle.npz
  python sdr_classify.py live --channel 6

Notes
  * The model was trained with every window referenced to a noise floor of
    power 1 over 20 MHz. A real capture is rescaled the same way: either
    from a reference capture taken with the antenna removed (best, use
    --floor-ref) or estimated automatically from the quietest windows.
  * Gain is fixed (manual) so the floor stays constant within a capture.
    Keep the same --gain for the capture and its floor reference.
  * Each SDR buffer is contiguous; windows never straddle the gap between
    buffers (USB can't stream 20 MS/s without gaps, and doesn't need to).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

FS = 20e6
WIN = 1024
WIFI_CH = {**{c: 2412e6 + 5e6 * (c - 1) for c in range(1, 14)}, 14: 2484e6}
GROUP_OF = {"noise": "clean", "wifi_ofdm": "wifi", "wifi_dsss": "wifi",
            "bluetooth": "bluetooth", "bpsk_link": "other_comm",
            "jam_tone": "jamming", "jam_barrage": "jamming",
            "jam_sweep": "jamming", "jam_pulsed": "jamming",
            "wifi_congestion": "interference"}
COLORS = {"noise": "#d9d9d9", "wifi_ofdm": "#1f77b4", "wifi_dsss": "#17becf",
          "bluetooth": "#2ca02c", "bpsk_link": "#8c564b", "jam_tone": "#d62728",
          "jam_barrage": "#ff7f0e", "jam_sweep": "#e377c2", "jam_pulsed": "#9467bd", "wifi_congestion": "#aec7e8",
          "uncertain": "#000000"}


# ------------------------------------------------------------------- SDR
def open_sdr(args):
    try:
        import adi
    except ImportError:
        sys.exit("!! pyadi-iio not installed: pip install pylibiio pyadi-iio  (and install libiio v0.25 first)")
    sdr = adi.Pluto(args.uri)
    sdr.sample_rate = int(FS)
    sdr.rx_rf_bandwidth = int(20e6)
    sdr.rx_lo = int(freq_of(args))
    sdr.gain_control_mode_chan0 = "manual"
    sdr.rx_hardwaregain_chan0 = float(args.gain)
    sdr.rx_buffer_size = int(args.buffer)
    for _ in range(4):  # flush stale buffers after retuning
        sdr.rx()
    return sdr


def freq_of(args):
    if getattr(args, "channel", None):
        return WIFI_CH[args.channel]
    return args.freq


def add_sdr_args(p):
    p.add_argument("--uri", default="ip:192.168.2.1", help="SDR address (USB default ip:192.168.2.1)")
    p.add_argument("--channel", type=int, choices=sorted(WIFI_CH), help="Wi-Fi channel to centre on")
    p.add_argument("--freq", type=float, default=2437e6, help="centre frequency, Hz (if no --channel)")
    p.add_argument("--gain", type=float, default=40.0, help="manual RX gain, dB (0-70)")
    p.add_argument("--buffer", type=int, default=2 ** 20, help="samples per contiguous buffer")


def cmd_check(args):
    sdr = open_sdr(args)
    print(f"connected: {args.uri}")
    for k in ("sample_rate", "rx_rf_bandwidth", "rx_lo", "gain_control_mode_chan0", "rx_hardwaregain_chan0"):
        print(f"  {k:24s} {getattr(sdr, k)}")
    x = sdr.rx()
    p = np.mean(np.abs(x) ** 2)
    peak = np.max(np.abs(np.concatenate([x.real, x.imag])))
    print(f"  buffer: {len(x)} samples, mean power {10 * np.log10(p):.1f} dBFS-ish, peak |I/Q| {peak:.0f} / 2048")
    if peak > 1900:
        print("  !! close to ADC full scale - lower --gain")
    elif peak < 30:
        print("  !! very low level - raise --gain or check the antenna")


def cmd_capture(args):
    sdr = open_sdr(args)
    n = max(1, int(round(args.seconds * FS / args.buffer)))
    bufs = np.empty((n, args.buffer), np.complex64)
    t0 = time.time()
    for i in range(n):
        bufs[i] = sdr.rx()
    clip = np.mean(np.maximum(np.abs(bufs.real), np.abs(bufs.imag)) > 2000)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {"fs": FS, "freq": freq_of(args), "gain": args.gain, "buffer": args.buffer,
            "uri": args.uri, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "note": args.note or ""}
    np.savez(out, iq=bufs, meta=json.dumps(meta))
    print(f"captured {n} buffers x {args.buffer} samples ({n * args.buffer / FS:.2f} s of signal) "
          f"in {time.time() - t0:.1f} s wall time -> {out}")
    if clip > 1e-4:
        print(f"  !! {clip * 100:.2f}% of samples near ADC full scale - recapture with lower --gain")


# ------------------------------------------------------------ processing
def load_capture(path):
    z = np.load(path)
    meta = json.loads(str(z["meta"]))
    return z["iq"], meta


def prep(bufs, fs, dc_remove=True):
    """complex buffers -> list of (nwin, WIN) complex128 arrays at 20 MS/s."""
    out = []
    for b in bufs:
        b = b.astype(np.complex128)
        if fs != FS:
            from fractions import Fraction
            from scipy.signal import resample_poly
            fr = Fraction(FS / fs).limit_denominator(1000)
            b = resample_poly(b, fr.numerator, fr.denominator)
        if dc_remove:
            b = b - b.mean()  # Pluto LO leakage / DC offset
        k = len(b) // WIN
        out.append(b[:k * WIN].reshape(k, WIN))
    return out


def window_floor(w):
    """Per-window noise-floor estimate (power per sample): median of the
    mid-band periodogram, corrected for the exponential distribution (/ln2).
    Skips DC and the outer 10% of the band (analog filter roll-off)."""
    h = np.hanning(WIN)
    P = np.abs(np.fft.fftshift(np.fft.fft(w * h, axis=-1), axes=-1)) ** 2 / np.sum(h ** 2)
    lo, hi, c = int(0.1 * WIN), int(0.9 * WIN), WIN // 2
    mid = np.concatenate([P[:, lo:c - 4], P[:, c + 5:hi]], axis=-1)
    return np.median(mid, axis=-1) / np.log(2)


def estimate_floor(wins, quiet_frac=0.1):
    f = window_floor(np.concatenate(wins))
    k = max(1, int(len(f) * quiet_frac))
    return float(np.mean(np.sort(f)[:k])), f


def _smooth_circ(S, k=9):
    return np.convolve(np.r_[S[-k:], S, S[:k]], np.ones(k) / k, "same")[k:-k]


def receiver_shape(wins):
    """Receiver frequency response (noise PSD shape) from a noise-only capture,
    normalised to median 1. The Pluto's filters roll the noise off by ~2 dB at
    the band edges; the model was trained on perfectly white noise and is very
    sensitive to that, so we divide it out."""
    w = np.concatenate(wins)
    # per-bin MEDIAN over windows: robust to the odd Wi-Fi burst that leaks into the
    # board even with the antenna off (a mean would bake those bursts into the shape)
    S = np.median(np.abs(np.fft.fft(w, axis=-1)) ** 2, axis=0)
    return S / np.median(S)


def equalise(wins, H):
    return [np.fft.ifft(np.fft.fft(w, axis=-1) / np.sqrt(H), axis=-1) for w in wins]


def quiet_level(wins, quiet_frac=0.1):
    """Mean power of the quietest windows (after equalisation the floor is white)."""
    p = np.mean(np.abs(np.concatenate(wins)) ** 2, axis=-1)
    k = max(1, int(len(p) * quiet_frac))
    return float(np.mean(np.sort(p)[:k]))


def calibrate(wins, ref_path, level_mode, dc=True, meta=None):
    """-> (equalised wins, floor power, description). With a reference capture
    (antenna off, same gain) the receiver shape is divided out; the floor level
    comes from the capture's own quietest windows (auto) or from the reference."""
    if not ref_path:
        floor, _ = estimate_floor(wins)
        return wins, floor, "auto level, no receiver equalisation (pass --floor-ref for better results)"
    rb, rmeta = load_capture(ref_path)
    if meta and rmeta.get("gain") != meta.get("gain"):
        print(f"!! reference gain {rmeta.get('gain')} dB != capture gain {meta.get('gain')} dB")
    rw = prep(rb, rmeta["fs"], dc)
    H = receiver_shape(rw)
    wins = equalise(wins, H)
    ref_level = quiet_level(equalise(rw, H), quiet_frac=0.5)
    own_level = quiet_level(wins)
    rise = 10 * np.log10(own_level / ref_level)
    floor = own_level if level_mode == "auto" else ref_level
    how = (f"receiver equalised from {Path(ref_path).name}; level {level_mode}; "
           f"background is {rise:+.1f} dB vs antenna-off floor")
    return wins, floor, how


def load_model(ckpt, dev):
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_cnn import RFNet
    state = torch.load(ckpt, map_location=dev)
    classes = state["classes"]
    model = RFNet(len(classes), w=state["args"].get("width", 48)).to(dev)
    model.load_state_dict(state["model"])
    model.eval()
    return model, classes


def run_model(model, X, dev, batch=2048):
    import torch
    probs = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i + batch]).to(dev)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                probs.append(model(xb).float().softmax(-1).cpu().numpy())
    return np.concatenate(probs)


def to_model_input(wins, floor):
    w = np.concatenate(wins) / np.sqrt(floor)
    return np.stack([w.real, w.imag], 1).astype(np.float32)


def summarise(pred, conf, classes, thresh):
    lab = np.where(conf >= thresh, np.array(classes)[pred], "uncertain")
    names = [c for c in classes] + ["uncertain"]
    counts = {c: int(np.sum(lab == c)) for c in names}
    return lab, counts


def device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------------------------------------- classify
def cmd_classify(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    bufs, meta = load_capture(args.capture)
    wins = prep(bufs, meta["fs"], not args.no_dc)
    wins, floor, how = calibrate(wins, args.floor_ref, args.floor_level, not args.no_dc, meta)
    dev = device()
    model, classes = load_model(args.model, dev)
    X = to_model_input(wins, floor)
    probs = run_model(model, X, dev)
    pred, conf = probs.argmax(1), probs.max(1)
    lab, counts = summarise(pred, conf, classes, args.threshold)
    snr_db = 10 * np.log10(np.mean(np.abs(X[:, 0] + 1j * X[:, 1]) ** 2, axis=1))  # power over floor

    out = Path(args.out or Path(args.capture).with_suffix(""))
    out.mkdir(parents=True, exist_ok=True)
    nb = [len(w) for w in wins]
    buf_idx = np.repeat(np.arange(len(nb)), nb)
    win_idx = np.concatenate([np.arange(k) for k in nb])
    df = pd.DataFrame({"buffer": buf_idx, "window": win_idx,
                       "t_ms_in_buffer": win_idx * WIN / FS * 1e3,
                       "pred": lab, "confidence": conf, "power_over_floor_db": snr_db})
    for i, c in enumerate(classes):
        df[f"p_{c}"] = probs[:, i]
    df.to_csv(out / "windows.csv", index=False)

    total = len(lab)
    active = total - counts.get("noise", 0)
    print(f"\ncapture: {Path(args.capture).name}  {meta['freq'] / 1e6:.1f} MHz  gain {meta['gain']} dB  "
          f"{total} windows ({total * WIN / FS * 1e3:.0f} ms of signal)")
    print(f"noise floor: {how}")
    print(f"\n{'class':16s} {'windows':>8s} {'% all':>7s} {'% active':>9s} {'mean conf':>10s}")
    for c in classes + ["uncertain"]:
        n = counts.get(c, 0)
        if n == 0:
            continue
        m = lab == c
        pa = f"{100 * n / active:8.1f}%" if c != "noise" and active else "        -"
        print(f"{c:16s} {n:8d} {100 * n / total:6.1f}% {pa} {conf[m].mean():10.2f}")
    groups = {}
    for c, n in counts.items():
        g = GROUP_OF.get(c, "uncertain")
        groups[g] = groups.get(g, 0) + n
    print("\ngroups: " + "  ".join(f"{g} {100 * n / total:.1f}%" for g, n in groups.items() if n))
    jam = groups.get("jamming", 0)
    if jam:
        print(f"note: {jam} windows ({100 * jam / total:.1f}%) flagged as jamming - in a home capture these are "
              f"almost certainly false alarms (e.g. microwave oven, unseen signal types).")

    # ---- report figure: spectrogram of one buffer + class strip + class bar chart
    from scipy.signal import stft
    b = int(np.clip(args.plot_buffer, 0, len(wins) - 1))
    sig = wins[b].ravel() / np.sqrt(floor)
    f, t, Z = stft(sig, fs=FS, nperseg=256, noverlap=128, return_onesided=False)
    Z = np.fft.fftshift(Z, axes=0)
    f = np.fft.fftshift(f)
    S = 20 * np.log10(np.abs(Z) + 1e-9)
    fig = plt.figure(figsize=(12, 6.5))
    gs = fig.add_gridspec(3, 2, height_ratios=[6, 0.6, 0.2], width_ratios=[3.2, 1], hspace=0.08, wspace=0.25)
    ax = fig.add_subplot(gs[0, 0])
    vmin = np.percentile(S, 5)
    ax.pcolormesh(t * 1e3, f / 1e6, S, shading="auto", cmap="viridis", vmin=vmin, vmax=vmin + 45)
    ax.set_ylabel("offset from %.0f MHz (MHz)" % (meta["freq"] / 1e6))
    ax.set_title(f"{Path(args.capture).name} - buffer {b} ({len(wins[b]) * WIN / FS * 1e3:.1f} ms)")
    plt.setp(ax.get_xticklabels(), visible=False)
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax)
    sel = buf_idx == b
    strip_lab = lab[sel]
    tw = np.arange(len(strip_lab)) * WIN / FS * 1e3
    for c in np.unique(strip_lab):
        m = strip_lab == c
        ax2.bar(tw[m], 1, width=WIN / FS * 1e3, align="edge", color=COLORS.get(c, "#000"), linewidth=0)
    ax2.set_yticks([])
    ax2.set_xlabel("time (ms)")
    ax2.set_xlim(0, len(strip_lab) * WIN / FS * 1e3)
    ax3 = fig.add_subplot(gs[0:2, 1])
    shown = [c for c in classes + ["uncertain"] if counts.get(c, 0)]
    ax3.barh(shown, [100 * counts[c] / total for c in shown], color=[COLORS.get(c, "#000") for c in shown])
    ax3.invert_yaxis()
    ax3.set_xlabel("% of windows (whole capture)")
    ax3.set_title("predicted class")
    fig.savefig(out / "classification.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    summary = {"capture": str(args.capture), "meta": meta, "floor_method": how, "floor_power": floor,
               "windows": total, "threshold": args.threshold, "counts": counts, "groups": groups}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"\noutputs in {out.resolve()}")


# ------------------------------------------------------------------ live
def cmd_live(args):
    dev = device()
    model, classes = load_model(args.model, dev)
    sdr = open_sdr(args)
    print("calibrating from the first 8 buffers ...")
    first = prep([sdr.rx() for _ in range(8)], FS)
    H = None
    if args.floor_ref:
        rb, rmeta = load_capture(args.floor_ref)
        rw = prep(rb, rmeta["fs"])
        H = receiver_shape(rw)
        ref_level = quiet_level(equalise(rw, H), quiet_frac=0.5)
        first = equalise(first, H)
        own = quiet_level(first)
        floor = own if args.floor_level == "auto" else ref_level
        print(f"receiver equalised; background {10 * np.log10(own / ref_level):+.1f} dB vs antenna-off floor")
    else:
        floor, _ = estimate_floor(first)
        print("!! no --floor-ref: receiver not equalised, expect extra 'uncertain' windows")
    print(f"{freq_of(args) / 1e6:.1f} MHz, gain {args.gain} dB - Ctrl+C to stop\n")
    order = [c for c in classes if c != "noise"] + ["uncertain"]
    hist = []
    try:
        while True:
            wins = prep([sdr.rx()], FS)
            if H is not None:
                wins = equalise(wins, H)
            probs = run_model(model, to_model_input(wins, floor), dev)
            lab, counts = summarise(probs.argmax(1), probs.max(1), classes, args.threshold)
            hist.append(counts)
            hist = hist[-args.smooth:]
            agg = {c: sum(h.get(c, 0) for h in hist) for c in classes + ["uncertain"]}
            tot = sum(agg.values())
            busy = 100 * (tot - agg.get("noise", 0)) / tot
            top = sorted((c for c in order if agg.get(c, 0)), key=lambda c: -agg[c])[:4]
            parts = "  ".join(f"{c} {100 * agg[c] / tot:.0f}%" for c in top)
            jam = sum(agg.get(c, 0) for c in classes if c.startswith("jam"))
            flag = "  <-- JAMMING?" if jam / tot > args.alarm else ""
            print(f"{time.strftime('%H:%M:%S')}  busy {busy:3.0f}%  jam {100 * jam / tot:3.0f}% | {parts}{flag}",
                  flush=True)
    except KeyboardInterrupt:
        print("\nstopped")


# -------------------------------------------------------------- diagnose
def cmd_diagnose(args):
    """Quick health check of captures: background level vs the antenna-off
    floor, impulsiveness of the quiet windows, clipping, receiver shape."""
    ref = None
    if args.floor_ref:
        rb, rmeta = load_capture(args.floor_ref)
        rw = prep(rb[:args.buffers], rmeta["fs"])
        H = receiver_shape(rw)
        ref = quiet_level(equalise(rw, H), 0.5)
        print(f"reference {Path(args.floor_ref).name}: receiver shape spans "
              f"{10 * np.log10(H.min()):+.1f} .. {10 * np.log10(H.max()):+.1f} dB across the band")
    print(f"\n{'capture':28s} {'note':24s} {'bg vs floor':>11s} {'kurtosis':>9s} {'PAPR dB':>8s} "
          f"{'clip %':>7s}   (white noise: kurtosis ~3.2, PAPR ~8.6)")
    for path in args.captures:
        bufs, meta = load_capture(path)
        raw = bufs[:args.buffers]
        clip = 100 * np.mean(np.maximum(np.abs(raw.real), np.abs(raw.imag)) > 2000)
        w = prep(raw, meta["fs"])
        if args.floor_ref:
            w = equalise(w, H)
        w = np.concatenate(w)
        p = np.mean(np.abs(w) ** 2, axis=1)
        q = w[np.argsort(p)[:max(1, len(p) // 10)]]
        a = np.abs(q)
        kurt = np.median(np.mean((a - a.mean(1, keepdims=True)) ** 4, 1) / a.var(1) ** 2)
        papr = np.median(10 * np.log10((a ** 2).max(1) / (a ** 2).mean(1)))
        rise = f"{10 * np.log10(np.mean(np.sort(p)[:len(p) // 10]) / ref):+.1f} dB" if ref else "-"
        print(f"{Path(path).name:28s} {meta.get('note', '')[:24]:24s} {rise:>11s} {kurt:9.1f} {papr:8.1f} {clip:7.3f}")
    print("\nbg vs floor > ~3 dB with kurtosis >> 3 = impulsive interference reaching the antenna "
          "(often the laptop / USB right next to it).")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("check", help="test the SDR connection")
    add_sdr_args(p)
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("capture", help="record IQ to .npz")
    add_sdr_args(p)
    p.add_argument("--seconds", type=float, default=5.0, help="seconds of signal to record")
    p.add_argument("--out", required=True)
    p.add_argument("--note", help="free text saved with the capture, e.g. 'microwave on'")
    p.set_defaults(func=cmd_capture)

    p = sub.add_parser("diagnose", help="background level / impulsiveness of captures")
    p.add_argument("captures", nargs="+")
    p.add_argument("--floor-ref", help="antenna-off capture, same gain")
    p.add_argument("--buffers", type=int, default=4, help="buffers to analyse per capture")
    p.set_defaults(func=cmd_diagnose)

    for name, fn, hlp in (("classify", cmd_classify, "classify a capture file"),
                          ("live", cmd_live, "continuous capture + classification")):
        p = sub.add_parser(name, help=hlp)
        p.add_argument("--model", default="results/cnn/best.pt")
        p.add_argument("--floor-ref", help="capture taken with the antenna removed, same gain")
        p.add_argument("--floor-level", choices=["auto", "ref"], default="auto",
                       help="noise floor level: this capture's quietest windows (auto) or the reference")
        p.add_argument("--threshold", type=float, default=0.6, help="below this confidence -> 'uncertain'")
        p.set_defaults(func=fn)
        if name == "classify":
            p.add_argument("capture")
            p.add_argument("--out", help="output folder (default: next to the capture)")
            p.add_argument("--plot-buffer", type=int, default=0, help="which buffer to draw")
            p.add_argument("--no-dc", action="store_true", help="skip DC-offset removal")
        else:
            add_sdr_args(p)
            p.add_argument("--smooth", type=int, default=5, help="buffers in the rolling summary")
            p.add_argument("--alarm", type=float, default=0.2, help="jamming fraction that raises the flag")

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
