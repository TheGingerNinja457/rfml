#!/usr/bin/env python3
"""
build_dataset.py - 2.4 GHz RF source classification dataset builder
Classes: noise / Wi-Fi OFDM / Wi-Fi DSSS / Bluetooth / BPSK link / 4 jammer types /
         Wi-Fi congestion (overlapping co-/adjacent-channel Wi-Fi + Bluetooth, benign)

Builds ONE HDF5 file of fixed-length complex baseband windows at a common
sample rate (20 MS/s, 1024 samples = 51.2 us), from:
  1. Synthetic Wi-Fi, Bluetooth and jammer signals         (always)
  2. Alhazbi et al. indoor jamming IQ .mat files            (optional)
  3. Fraunhofer FKIE SigMF bursts (real Wi-Fi / Bluetooth)  (optional, real-world test set)

Also writes <out>.features.csv (hand-crafted features for the classic-ML
baseline) and <out>.preview.png (one spectrogram per class, sanity check).

Scaling convention: every window is referenced to a noise floor of power 1.0
across the full 20 MHz. SNR / JNR are measured over the active part of the
signal relative to that floor. Windows are NOT normalised per-window, so
absolute level stays meaningful (otherwise full-band barrage jamming and the
plain noise floor are indistinguishable).

Usage (PowerShell, venv active):
  python build_dataset.py build
  python build_dataset.py build --alhazbi-dir data\\alhazbi --alhazbi-fs <Hz> --fkie-dir data\\fkie
  python build_dataset.py inspect data\\alhazbi\\w1.mat
  python build_dataset.py inspect data\\fkie
"""
import argparse
import csv
import json
import re
import sys
from fractions import Fraction
from pathlib import Path

import h5py
import numpy as np
from scipy import signal as sps
from scipy.io import loadmat

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional
    def tqdm(it, **kw):
        return it

FS = 20e6   # common sample rate, Hz
WIN = 1024  # samples per window

CLASSES = ["noise", "wifi_ofdm", "wifi_dsss", "bluetooth", "bpsk_link",
           "jam_tone", "jam_barrage", "jam_sweep", "jam_pulsed", "wifi_congestion"]
GROUP_OF = {"noise": "clean", "wifi_ofdm": "wifi", "wifi_dsss": "wifi",
            "bluetooth": "bluetooth", "bpsk_link": "other_comm",
            "jam_tone": "jamming", "jam_barrage": "jamming",
            "jam_sweep": "jamming", "jam_pulsed": "jamming",
            "wifi_congestion": "interference"}
GROUPS = ["clean", "wifi", "bluetooth", "other_comm", "jamming", "interference"]
SPLITS = ["train", "val", "test", "test_unseen", "test_real"]
COMM = ["wifi_ofdm", "wifi_dsss", "bluetooth"]
JAMS = ["jam_tone", "jam_barrage", "jam_sweep", "jam_pulsed"]
BACKGROUNDS = ["none", "wifi_ofdm", "wifi_dsss", "bluetooth", "bpsk_link", "wifi_congestion"]


# ----------------------------------------------------------------- helpers
def awgn(rng, n, power=1.0):
    return np.sqrt(power / 2) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))


def unit_power(x):
    """Scale so the ACTIVE (non-zero) part has power 1."""
    m = np.abs(x) > 1e-12
    p = np.mean(np.abs(x[m]) ** 2) if m.any() else 0.0
    return x / np.sqrt(p) if p > 0 else x


def tvec(n):
    return np.arange(n) / FS


def qam(rng, m, shape):
    if m == 2:
        return (2 * rng.integers(0, 2, shape) - 1).astype(complex)
    k = int(np.sqrt(m))
    lv = np.arange(-k + 1, k, 2)
    s = rng.choice(lv, shape) + 1j * rng.choice(lv, shape)
    return s / np.sqrt(2 * np.mean(lv.astype(float) ** 2))


# --------------------------------------------------- communication signals
def gen_wifi_ofdm(rng, n):
    """802.11a/g/n-style OFDM: 64-pt IFFT, 52 used subcarriers, 16-sample CP, 20 MHz."""
    nsym = n // 80 + 2
    used = np.r_[np.arange(-26, 0), np.arange(1, 27)] % 64
    X = np.zeros((nsym, 64), complex)
    X[:, used] = qam(rng, int(rng.choice([2, 4, 16, 64])), (nsym, 52))
    x = np.fft.ifft(X, axis=1)
    x = np.concatenate([x[:, -16:], x], axis=1).ravel()
    s = int(rng.integers(0, 80))
    return unit_power(x[s:s + n])


BARKER = np.array([1, -1, 1, 1, -1, 1, 1, 1, -1, -1, -1], float)


def gen_wifi_dsss(rng, n):
    """802.11b DSSS: 11-chip Barker at 11 Mchip/s, DBPSK (1 Mb/s) or DQPSK (2 Mb/s)."""
    m = int(rng.choice([2, 4]))
    nsym = n // 20 + 6
    d = np.exp(1j * np.cumsum(rng.integers(0, m, nsym)) * 2 * np.pi / m)
    chips = (d[:, None] * BARKER[None, :]).ravel()
    x = sps.resample_poly(chips, 20, 11)  # 11 Mchip/s -> 20 MS/s
    s = int(rng.integers(0, len(x) - n))
    return unit_power(x[s:s + n])


def gen_bluetooth(rng, n):
    """GFSK at 1 Msym/s, BT=0.5. Classic (h~0.32) or BLE (h=0.5), random channel offset."""
    sps_ = int(FS / 1e6)
    h = 0.5 if rng.random() < 0.5 else rng.uniform(0.28, 0.35)
    nb = n // sps_ + 8
    up = np.repeat(2 * rng.integers(0, 2, nb) - 1, sps_).astype(float)
    t = np.arange(-2 * sps_, 2 * sps_ + 1) / sps_
    sigma = np.sqrt(np.log(2)) / (2 * np.pi * 0.5)
    g = np.exp(-t ** 2 / (2 * sigma ** 2))
    g /= g.sum()
    f = np.convolve(up, g, mode="same")
    x = np.exp(1j * np.pi * h * np.cumsum(f) / sps_)
    s = int(rng.integers(2 * sps_, len(x) - n - 2 * sps_))
    x = x[s:s + n]
    fo = rng.uniform(-(FS / 2 - 1.5e6), FS / 2 - 1.5e6)
    return x * np.exp(2j * np.pi * fo * tvec(n))


COMM_GEN = {"wifi_ofdm": gen_wifi_ofdm, "wifi_dsss": gen_wifi_dsss, "bluetooth": gen_bluetooth}


def burst_gate(rng, x, p=0.3):
    """With probability p the frame starts or ends inside the window (30-90% occupancy)."""
    if rng.random() >= p:
        return x
    n = len(x)
    L = int(n * rng.uniform(0.3, 0.9))
    env = np.zeros(n)
    if rng.random() < 0.5:
        env[:L] = 1
    else:
        env[n - L:] = 1
    return x * env


def channel(rng, x, multipath=True, cfo_hz=50e3):
    """Random phase, CFO, optional short indoor multipath; keeps active power at 1."""
    n = len(x)
    mask = np.abs(x) > 1e-12
    x = x * np.exp(1j * (rng.uniform(0, 2 * np.pi)
                         + 2 * np.pi * rng.uniform(-cfo_hz, cfo_hz) * tvec(n)))
    if multipath and rng.random() < 0.7:
        ntap = int(rng.integers(2, 5))
        p = np.exp(-np.arange(ntap) * rng.uniform(0.5, 2.0))
        h = np.sqrt(p / p.sum() / 2) * (rng.standard_normal(ntap) + 1j * rng.standard_normal(ntap))
        x = np.convolve(x, h)[:n]
    if mask.any():
        x = x / np.sqrt(np.mean(np.abs(x[mask]) ** 2))
    return x


# ---------------------------------------------------------------- jammers
def gen_jam_tone(rng, n):
    k = int(rng.choice([1, 1, 1, 2, 3]))
    t = tvec(n)
    x = sum(np.exp(1j * (2 * np.pi * rng.uniform(-0.45, 0.45) * FS * t + rng.uniform(0, 2 * np.pi)))
            for _ in range(k))
    return unit_power(x)


def bandlimited_noise(rng, n, frac, fc):
    X = np.fft.fft(awgn(rng, n))
    f = np.fft.fftfreq(n, 1 / FS)
    X[np.abs(f - fc) > frac * FS / 2] = 0
    return unit_power(np.fft.ifft(X))


def gen_jam_barrage(rng, n):
    frac = rng.uniform(0.2, 1.0)  # fraction of the 20 MHz covered
    fc = rng.uniform(-(1 - frac) * FS / 2, (1 - frac) * FS / 2)
    return bandlimited_noise(rng, n, frac, fc)


def gen_jam_sweep(rng, n):
    B = rng.uniform(0.3, 0.95) * FS
    T = rng.uniform(n / FS / 8, n / FS * 4)  # 8 sweeps per window ... 1/4 sweep per window
    fc = rng.uniform(-(FS - B) / 2, (FS - B) / 2)
    t = tvec(n) + rng.uniform(0, T)
    ramp = (t / T) % 1.0 - 0.5
    finst = fc + B * (ramp if rng.random() < 0.5 else -ramp)
    return np.exp(2j * np.pi * np.cumsum(finst) / FS)


def gen_jam_pulsed(rng, n):
    base = gen_jam_barrage(rng, n) if rng.random() < 0.6 else gen_jam_tone(rng, n)
    period = max(int(rng.uniform(0.1, 0.6) * n), 16)
    on = max(int(period * rng.uniform(0.1, 0.5)), 8)
    ph = int(rng.integers(0, period))
    env = (((np.arange(n) + ph) % period) < on).astype(float)
    return base * env  # on-period power stays 1


JAM_GEN = {"jam_tone": gen_jam_tone, "jam_barrage": gen_jam_barrage,
           "jam_sweep": gen_jam_sweep, "jam_pulsed": gen_jam_pulsed}


# ------------------------------------------------- Wi-Fi congestion (benign)
def overlap_gate(rng, x, lo=0.6):
    """Occupy 60-100% of the window, anchored at the start or the end.
    Any two signals gated like this overlap by at least 20% of the window."""
    n = len(x)
    L = int(n * rng.uniform(lo, 1.0))
    env = np.zeros(n)
    if rng.random() < 0.5:
        env[:L] = 1
    else:
        env[n - L:] = 1
    return x * env


def adjacent_ofdm(rng, n, f_off):
    """OFDM frame on a neighbouring channel: generate at 40 MS/s, shift, filter to +-10 MHz."""
    pad = 64
    x = sps.resample_poly(gen_wifi_ofdm(rng, n + pad), 2, 1)
    x = x * np.exp(2j * np.pi * f_off * np.arange(len(x)) / (2 * FS))
    x = sps.resample_poly(x, 1, 2)[pad // 2:pad // 2 + n]
    return x  # NOT unit power: only the in-band part survives


def gen_congestion(rng, n, snr_rng):
    """Busy 2.4 GHz channel: a co-channel OFDM frame overlapped by 1-3 other
    sources (colliding co-channel frame, neighbouring-channel frame, legacy
    802.11b, Bluetooth hop). Each source has its own SNR and timing."""
    snr0 = rng.uniform(*snr_rng)
    y = channel(rng, overlap_gate(rng, gen_wifi_ofdm(rng, n))) * 10 ** (snr0 / 20)
    kinds = ["cochannel", "adjacent", "adjacent", "dsss", "bluetooth"]
    for kind in rng.choice(kinds, size=int(rng.integers(1, 4)), replace=False):
        s_snr = rng.uniform(snr_rng[0], snr_rng[1] + 5)
        if kind == "cochannel":
            s = channel(rng, gen_wifi_ofdm(rng, n))
        elif kind == "adjacent":
            f_off = float(rng.choice([-15e6, -10e6, -5e6, 5e6, 10e6, 15e6]))
            s = adjacent_ofdm(rng, n, f_off)
        elif kind == "dsss":
            s = channel(rng, gen_wifi_dsss(rng, n))
        else:
            s = gen_bluetooth(rng, n)
        y = y + overlap_gate(rng, s) * 10 ** (s_snr / 20)
    return y, snr0


def make_synthetic(rng, label, snr_rng, jnr_rng, bg_prob):
    y = awgn(rng, WIN)
    snr = jnr = np.nan
    bg = "none"
    if label in COMM:
        s = channel(rng, burst_gate(rng, COMM_GEN[label](rng, WIN)))
        snr = rng.uniform(*snr_rng)
        y = y + s * 10 ** (snr / 20)
    elif label == "wifi_congestion":
        s, snr = gen_congestion(rng, WIN, snr_rng)
        y = y + s
    elif label in JAMS:
        if rng.random() < bg_prob:  # jammer on top of real traffic, so classes share backgrounds
            bg = str(rng.choice(COMM + ["wifi_congestion"]))
            if bg == "wifi_congestion":
                s, snr = gen_congestion(rng, WIN, snr_rng)
            else:
                s = channel(rng, burst_gate(rng, COMM_GEN[bg](rng, WIN)))
                snr = rng.uniform(*snr_rng)
                s = s * 10 ** (snr / 20)
            y = y + s
        j = channel(rng, JAM_GEN[label](rng, WIN), multipath=False, cfo_hz=0)
        jnr = rng.uniform(*jnr_rng)
        y = y + j * 10 ** (jnr / 20)
    return y, snr, jnr, bg


# --------------------------------------------------------------- features
FEATURE_NAMES = ["power_db", "papr_db", "amp_kurtosis", "spec_flatness", "occ_bw",
                 "centroid", "peak_to_median_db", "n_spec_peaks", "duty",
                 "blk_power_cv", "ifreq_std"]


def features(x):
    p = np.abs(x) ** 2
    a = np.abs(x)
    mp = p.mean()
    P = np.abs(np.fft.fftshift(np.fft.fft(x * np.hanning(len(x))))) ** 2
    Pb = P.reshape(128, -1).sum(1)
    Pb = Pb / Pb.sum()
    med = np.median(Pb)
    cs = np.cumsum(Pb)
    freqs = np.linspace(-0.5, 0.5, 128, endpoint=False)
    blk = p.reshape(-1, 32).mean(1)
    ifreq = np.diff(np.unwrap(np.angle(x)))
    return [
        10 * np.log10(mp),
        10 * np.log10(p.max() / mp),
        np.mean((a - a.mean()) ** 4) / (a.var() ** 2 + 1e-12),
        np.exp(np.mean(np.log(Pb + 1e-12))) / np.mean(Pb),
        (np.searchsorted(cs, 0.995) - np.searchsorted(cs, 0.005) + 1) / 128,
        abs(float((Pb * freqs).sum())),
        10 * np.log10(Pb.max() / med),
        int(np.sum(Pb > 10 * med)),
        float(np.mean(blk > 2.0)),  # 32-sample blocks >3 dB above the floor
        blk.std() / blk.mean(),
        ifreq.std(),
    ]


# ----------------------------------------------------------------- writer
class Writer:
    def __init__(self, path, chunk=4096):
        self.f = h5py.File(path, "w")
        self.chunk = chunk
        self.n = 0
        self.f.create_dataset("iq", (0, 2, WIN), maxshape=(None, 2, WIN), dtype="float16",
                              chunks=(256, 2, WIN), compression="lzf")
        for name, dt in [("label", "int8"), ("group", "int8"), ("split", "int8"),
                         ("background", "int8"), ("snr_db", "float32"), ("jnr_db", "float32")]:
            self.f.create_dataset(name, (0,), maxshape=(None,), dtype=dt)
        self.f.create_dataset("source", (0,), maxshape=(None,), dtype=h5py.string_dtype())
        self.buf = {k: [] for k in ["iq", "label", "group", "split", "background",
                                     "snr_db", "jnr_db", "source"]}
        self.csv_f = open(path.with_suffix(".features.csv"), "w", newline="")
        self.csv = csv.writer(self.csv_f)
        self.csv.writerow(["idx", "label", "group", "split", "source", "snr_db", "jnr_db"] + FEATURE_NAMES)
        self.counts = {}
        self.examples = {}

    def add(self, y, label, split, snr, jnr, bg, source):
        idx = self.n + len(self.buf["iq"])
        self.buf["iq"].append(np.stack([y.real, y.imag]).astype(np.float16))
        self.buf["label"].append(CLASSES.index(label))
        self.buf["group"].append(GROUPS.index(GROUP_OF[label]))
        self.buf["split"].append(SPLITS.index(split))
        self.buf["background"].append(BACKGROUNDS.index(bg))
        self.buf["snr_db"].append(snr)
        self.buf["jnr_db"].append(jnr)
        self.buf["source"].append(source)
        self.csv.writerow([idx, label, GROUP_OF[label], split, source,
                           f"{snr:.2f}", f"{jnr:.2f}"] + [f"{v:.6g}" for v in features(y)])
        key = (label, split)
        self.counts[key] = self.counts.get(key, 0) + 1
        # preview: prefer clear (>=15 dB) examples with no background signal
        ex_key = label + (" (real)" if split == "test_real" else "")
        level = np.nanmax([snr if bg == "none" else np.nan, jnr, -99.0])
        score = int(bg == "none") + int(level >= 15)
        if ex_key not in self.examples or score > self.examples[ex_key][0]:
            self.examples[ex_key] = (score, y)
        if len(self.buf["iq"]) >= self.chunk:
            self.flush()

    def flush(self):
        k = len(self.buf["iq"])
        if not k:
            return
        for name, vals in self.buf.items():
            ds = self.f[name]
            ds.resize(self.n + k, axis=0)
            ds[self.n:self.n + k] = np.stack(vals) if name == "iq" else np.array(vals, dtype=ds.dtype)
            vals.clear()
        self.n += k

    def close(self, config):
        self.flush()
        a = self.f.attrs
        a["classes"] = json.dumps(CLASSES)
        a["groups"] = json.dumps(GROUPS)
        a["group_of"] = json.dumps(GROUP_OF)
        a["splits"] = json.dumps(SPLITS)
        a["backgrounds"] = json.dumps(BACKGROUNDS)
        a["fs"] = FS
        a["win"] = WIN
        a["scaling"] = "noise floor power = 1 over full band; SNR/JNR over active portion"
        a["config"] = json.dumps(config)
        self.f.close()
        self.csv_f.close()


# --------------------------------------------------------- resampling etc.
def make_resampler(fs_src):
    fr = Fraction(FS / fs_src).limit_denominator(1000)
    up, down = fr.numerator, fr.denominator
    return (lambda x: sps.resample_poly(x, up, down)), up, down


def out_of_band_noise(rng, n, fs_src):
    """Unit-floor noise only OUTSIDE the source bandwidth (fills the band like synthetic data)."""
    X = np.fft.fft(awgn(rng, n))
    X[np.abs(np.fft.fftfreq(n, 1 / FS)) < fs_src / 2] = 0
    return np.fft.ifft(X)


# ----------------------------------------------------- Alhazbi .mat loader
ALH_MODES = {"nojamming": "bpsk_link", "sine": "jam_tone", "gaussian": "jam_barrage"}


class MatSource:
    """Lazy reader for MAT files holding Nx2 (or 2xN, or complex) I/Q matrices."""

    def __init__(self, path):
        self.path = path
        if h5py.is_hdf5(path):  # MATLAB v7.3
            self.h5 = h5py.File(path, "r")
            self.data = None
            self.keys = [k for k in self.h5.keys() if not k.startswith("#")]
        else:  # MATLAB v5/v7: must load fully
            print(f"  {path.name}: pre-v7.3 MAT file, loading into RAM (may take a while) ...")
            self.h5 = None
            self.data = {k: v for k, v in loadmat(path).items() if not k.startswith("__")}
            self.keys = list(self.data)

    def ds(self, key):
        return self.h5[key] if self.h5 is not None else self.data[key]

    def layout(self, key):
        d = self.ds(key)
        sh = d.shape
        names = d.dtype.names
        if (names and "real" in names) or np.issubdtype(d.dtype, np.complexfloating):
            return int(np.argmax(sh)), True
        if len(sh) == 2 and sh[1] == 2:
            return 0, False
        if len(sh) == 2 and sh[0] == 2:
            return 1, False
        raise ValueError(f"{self.path.name}:{key} has unexpected shape {sh} / dtype {d.dtype}")

    def length(self, key):
        axis, _ = self.layout(key)
        return self.ds(key).shape[axis]

    def read(self, key, start, count):
        d = self.ds(key)
        axis, cplx = self.layout(key)
        sl = [slice(None)] * d.ndim
        sl[axis] = slice(start, start + count)
        a = d[tuple(sl)]
        if cplx:
            if a.dtype.names:
                a = a["real"] + 1j * a["imag"]
            return np.asarray(a, complex).ravel()
        a = np.asarray(a, np.float64)
        if axis == 1:
            a = a.T
        return a[:, 0] + 1j * a[:, 1]

    def find(self, name):
        for k in self.keys:
            if k.lower().replace("_", "") == name:
                return k
        return None

    def close(self):
        if self.h5 is not None:
            self.h5.close()


def add_alhazbi(w, rng, args):
    files = sorted(Path(args.alhazbi_dir).glob("*.mat"))
    if not files:
        print(f"!! no .mat files in {args.alhazbi_dir}, skipping Alhazbi")
        return
    unseen = {s.strip().lower() for s in args.alhazbi_unseen.split(",") if s.strip()}
    resample, up, down = make_resampler(args.alhazbi_fs)
    blk_win = 32
    src_count = int(np.ceil((blk_win * WIN + 256) * down / up))
    for path in files:
        tag = path.stem.lower()
        src = MatSource(path)
        keys = {mode: src.find(mode) for mode in ALH_MODES}
        if not all(keys.values()):
            print(f"!! {path.name}: expected Nojamming/Sine/Gaussian, found {src.keys}; skipping")
            src.close()
            continue
        # noise-floor density from the no-jamming capture (median Welch PSD)
        L0 = src.length(keys["nojamming"])
        probe = src.read(keys["nojamming"], L0 // 2, min(2_000_000, L0 // 2))
        _, P = sps.welch(probe, fs=args.alhazbi_fs, nperseg=1024, return_onesided=False,
                         scaling="density")
        d_src = float(np.median(P))
        a2 = 1.0 / (FS * d_src)  # scale so in-band floor density matches synthetic floor
        p_nojam = float(np.mean(np.abs(probe) ** 2))
        floor_src = d_src * args.alhazbi_fs
        link_snr = 10 * np.log10(max(p_nojam - floor_src, 1e-30) / floor_src)
        print(f"  {path.name}: est. link SNR {link_snr:.1f} dB (in source bandwidth)")
        is_unseen = tag in unseen
        for mode, label in ALH_MODES.items():
            key = keys[mode]
            L = src.length(key)
            pr = src.read(key, L // 2, min(500_000, L // 2))
            p_mode = float(np.mean(np.abs(pr) ** 2))
            jnr = 10 * np.log10(max(a2 * (p_mode - p_nojam), 1e-30)) if mode != "nojamming" else np.nan
            snr = 10 * np.log10(max(a2 * (p_nojam - floor_src), 1e-30))
            if is_unseen:
                plan = [("test_unseen", 0.0, 1.0, args.alhazbi_per_mode // 3)]
            else:
                n = args.alhazbi_per_mode
                plan = [("train", 0.0, 0.70, int(n * 0.70)),
                        ("val", 0.70, 0.85, int(n * 0.15)),
                        ("test", 0.85, 1.0, int(n * 0.15))]
            for split, f0, f1, nwin in plan:
                lo, hi = int(L * f0), int(L * f1) - src_count
                made = 0
                while made < nwin and hi > lo:
                    st = int(rng.integers(lo, hi))
                    x = resample(src.read(key, st, src_count)) * np.sqrt(a2)
                    for k in range(blk_win):
                        if made >= nwin:
                            break
                        seg = x[128 + k * WIN:128 + (k + 1) * WIN]
                        if len(seg) < WIN:
                            break
                        if args.alhazbi_fs < FS:
                            seg = seg + out_of_band_noise(rng, WIN, args.alhazbi_fs)
                        bg = "bpsk_link" if label != "bpsk_link" else "none"
                        w.add(seg, label, split, snr, jnr, bg, f"alhazbi_{tag}")
                        made += 1
            print(f"    {mode:9s} -> {label:12s} JNR~{jnr:5.1f} dB {'(unseen test)' if is_unseen else ''}")
        src.close()


# ------------------------------------------------------- FKIE SigMF loader
SIGMF_DT = {"f32": np.float32, "f64": np.float64, "i32": np.int32, "i16": np.int16,
            "i8": np.int8, "u32": np.uint32, "u16": np.uint16, "u8": np.uint8}


def read_sigmf(meta_path):
    meta = json.loads(meta_path.read_text())
    g = meta.get("global", {})
    m = re.fullmatch(r"c(f32|f64|i32|i16|i8|u32|u16|u8)(_le|_be)?", g.get("core:datatype", ""))
    if not m:
        raise ValueError(f"unsupported datatype {g.get('core:datatype')}")
    base = np.dtype(SIGMF_DT[m.group(1)]).newbyteorder(">" if m.group(2) == "_be" else "<")
    raw = np.fromfile(meta_path.with_suffix(".sigmf-data"), dtype=base).astype(np.float64)
    if m.group(1).startswith("u"):
        raw -= (np.iinfo(SIGMF_DT[m.group(1)]).max + 1) / 2
    return raw[0::2] + 1j * raw[1::2], float(g["core:sample_rate"]), meta


FKIE_LABELS = {"wifi": "wifi_ofdm", "wifi-dsss": "wifi_dsss",
               "bluetooth": "bluetooth", "bluetooth le": "bluetooth"}


def guess_label(path, meta):
    """Exact match on the annotation label (FKIE format), else the parent folder name."""
    for a in meta.get("annotations", []):
        lab = FKIE_LABELS.get(str(a.get("core:label", "")).strip().lower())
        if lab:
            return lab
    return FKIE_LABELS.get(Path(path).parent.name.strip().lower())


def burst_span(meta, n):
    """(start, stop) of the annotated burst, falling back to the whole file."""
    for a in meta.get("annotations", []):
        if "core:sample_start" in a and "core:sample_count" in a:
            s = int(a["core:sample_start"])
            return s, min(n, s + int(a["core:sample_count"]))
    return 0, n


def add_fkie(w, rng, args):
    metas = sorted(Path(args.fkie_dir).rglob("*.sigmf-meta"))
    if not metas:
        print(f"!! no .sigmf-meta files under {args.fkie_dir}, skipping FKIE")
        return
    found, skipped = {}, []
    for mp in tqdm(metas, desc="FKIE bursts"):
        try:
            x, fs_src, meta = read_sigmf(mp)
        except Exception as e:  # noqa: BLE001
            skipped.append(f"{mp.name} ({e})")
            continue
        label = guess_label(mp, meta)
        if label is None:
            skipped.append(mp.name)
            continue
        s0, s1 = burst_span(meta, len(x))
        resample, _, _ = make_resampler(fs_src)
        x = resample(x[s0:s1])
        if fs_src < FS:  # narrowband capture (e.g. Bluetooth at 2 MS/s): random channel offset
            fo = rng.uniform(-(FS - fs_src) / 2, (FS - fs_src) / 2)
            x = x * np.exp(2j * np.pi * fo * tvec(len(x)))
        if len(x) < WIN:
            pad = np.zeros(WIN, complex)
            o = int(rng.integers(0, WIN - len(x) + 1))
            pad[o:o + len(x)] = x
            segs = [pad]
        else:
            segs = [x[k * WIN:(k + 1) * WIN] for k in range(min(args.fkie_max_per_burst, len(x) // WIN))]
        for seg in segs:
            snr = rng.uniform(*args.snr)
            y = awgn(rng, WIN) + unit_power(seg) * 10 ** (snr / 20)
            w.add(y, label, "test_real", snr, np.nan, "none", f"fkie_{mp.stem}")
        found[label] = found.get(label, 0) + 1
    print(f"  FKIE bursts used per class: {found}")
    if skipped:
        print(f"  ignored {len(skipped)} bursts of other protocols (drone links, DECT, etc.)")


# ---------------------------------------------------------------- preview
def save_preview(examples, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = [c for c in CLASSES if c in examples] + sorted(k for k in examples if k.endswith("(real)"))
    cols = 4
    rows = int(np.ceil(len(keys) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
    for ax, k in zip(axes.ravel(), keys):
        f, t, Z = sps.stft(examples[k][1], fs=FS, nperseg=64, noverlap=48, return_onesided=False)
        Z = np.fft.fftshift(Z, axes=0)
        f = np.fft.fftshift(f)
        ax.pcolormesh(t * 1e6, f / 1e6, 20 * np.log10(np.abs(Z) + 1e-9), shading="auto", cmap="viridis")
        ax.set_title(k)
        ax.set_xlabel("time (us)")
        ax.set_ylabel("freq (MHz)")
    for ax in axes.ravel()[len(keys):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ------------------------------------------------------------------ build
def build(args):
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    w = Writer(out)

    print("== synthetic")
    for label in ["noise"] + COMM + JAMS + ["wifi_congestion"]:
        n = args.per_class
        bounds = [("train", int(n * 0.70)), ("val", int(n * 0.85)), ("test", n)]
        for i in tqdm(range(n), desc=f"  {label}"):
            split = next(s for s, b in bounds if i < b)
            y, snr, jnr, bg = make_synthetic(rng, label, args.snr, args.jnr, args.bg_prob)
            w.add(y, label, split, snr, jnr, bg, "synthetic")

    if args.alhazbi_dir:
        print("== Alhazbi")
        if not args.alhazbi_fs:
            sys.exit("!! --alhazbi-fs is required (source sample rate in Hz). "
                     "Check the BloodHound paper / dataset description.")
        add_alhazbi(w, rng, args)

    if args.fkie_dir:
        print("== FKIE")
        add_fkie(w, rng, args)

    examples = w.examples
    counts = w.counts
    w.close({k: v for k, v in vars(args).items() if k != "func"})

    print(f"\nwrote {out}  ({w.n} windows)")
    print(f"wrote {out.with_suffix('.features.csv')}")
    if not args.no_preview:
        save_preview(examples, out.with_suffix(".preview.png"))
        print(f"wrote {out.with_suffix('.preview.png')}")
    used_splits = [s for s in SPLITS if any(k[1] == s for k in counts)]
    print("\n" + f"{'class':16s}" + "".join(f"{s:>12s}" for s in used_splits))
    for c in CLASSES:
        if any(k[0] == c for k in counts):
            print(f"{c:16s}" + "".join(f"{counts.get((c, s), 0):12d}" for s in used_splits))


# ---------------------------------------------------------------- inspect
def inspect(args):
    p = Path(args.path)
    if p.is_dir():
        exts = {}
        for f in p.rglob("*"):
            if f.is_file():
                exts[f.suffix] = exts.get(f.suffix, 0) + 1
        print(f"{p}: {exts}")
        metas = sorted(p.rglob("*.sigmf-meta"))
        for mp in metas[:5]:
            meta = json.loads(mp.read_text())
            print(f"\n{mp.relative_to(p)} -> guessed label: {guess_label(mp, meta)}")
            print(json.dumps(meta.get("global", {}), indent=1)[:800])
        return
    if p.suffix == ".mat":
        src = MatSource(p)
        for k in src.keys:
            d = src.ds(k)
            print(f"{k:20s} shape={d.shape} dtype={d.dtype}")
        if src.h5 is not None:
            for k, v in src.h5.attrs.items():
                print(f"attr {k} = {v}")
        src.close()
        return
    if p.name.endswith(".sigmf-meta"):
        meta = json.loads(p.read_text())
        print(json.dumps(meta, indent=1)[:3000])
        print(f"\nguessed label: {guess_label(p, meta)}")
        return
    if p.suffix == ".h5":
        with h5py.File(p, "r") as f:
            for k in f:
                print(f"{k:12s} {f[k].shape} {f[k].dtype}")
            for k, v in f.attrs.items():
                print(f"attr {k} = {str(v)[:200]}")
        return
    sys.exit(f"don't know how to inspect {p}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="build the HDF5 dataset")
    b.add_argument("--out", default="data/rf24.h5")
    b.add_argument("--per-class", type=int, default=15000, help="synthetic windows per class")
    b.add_argument("--snr", type=float, nargs=2, default=[-5.0, 25.0], help="comm signal SNR range, dB")
    b.add_argument("--jnr", type=float, nargs=2, default=[-5.0, 25.0], help="jammer JNR range, dB")
    b.add_argument("--bg-prob", type=float, default=0.5, help="P(jammer sits on top of Wi-Fi/BT traffic)")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--alhazbi-dir")
    b.add_argument("--alhazbi-fs", type=float, help="Alhazbi capture sample rate, Hz (required with --alhazbi-dir)")
    b.add_argument("--alhazbi-per-mode", type=int, default=6000, help="windows per mode per file")
    b.add_argument("--alhazbi-unseen", default="w21", help="comma list of files held out as unseen-jammer test")
    b.add_argument("--fkie-dir")
    b.add_argument("--fkie-max-per-burst", type=int, default=4)
    b.add_argument("--no-preview", action="store_true")
    b.set_defaults(func=build)

    i = sub.add_parser("inspect", help="look inside a .mat / .sigmf-meta / folder / .h5")
    i.add_argument("path")
    i.set_defaults(func=inspect)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
