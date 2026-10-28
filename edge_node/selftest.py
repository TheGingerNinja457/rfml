#!/usr/bin/env python3
"""selftest - prove the chain works before you rely on it in the field.

    python3 selftest.py            # everything except the live link test
    python3 selftest.py --live     # also run tx and rx against a virtual radio

Checks, in order (11-17 cover multi-band and the shared link):
  1  DSP power calibration against tones of known strength
  2  tile stitching - no seams, no gaps, right frequencies
  3  DC/LO-leakage repair
  4  codec round trip, including the awkward values
  5  codec survives truncation and corruption without crashing
  6  bandwidth: measured bytes/sweep against the 17 kbit/s ceiling
  7  emitter detection and RSSI tracking under simulated fading
  8  relative-range maths against a known path-loss curve
  9  sequence-gap loss accounting
 11  band configuration: presets, --band parsing, validation
 12  multi-band sweeping: per-band tiles, gain and levels, unreachable bands
 13  the Pluto driver path against a mocked radio: stale buffers, scaling,
     LO range, gain order across band hops
 14  wire format v2 (band id), v1 still decodes
 15  link sharing: spectrum <= kbps - reserve, aux <= reserve, no band
     starves, interleaved frames reassemble
 16  base station: per-band views, names, aux channels, Pi clock skew, logs
 17  live: 3-band sdr_tx + aux test + fake GNSS -> virtual radio (with
     loss) -> base_rx, with position-tagged logs at both ends
 18  GNSS: your max_m8q_gps parser -> 31 B fix -> base, clock handling,
     position interpolation, the real serial path on a pseudo-terminal
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import rssi_wire as W
from base_rx import EmitterTrack, RxState, find_emitters, label_for
from rssi_wire import Sweep, decode_sweep, encode_sweep, encoded_size
from sdr_scan import (SimSource, SweepEngine, TilePlan, repair_dc,
                      robust_noise_floor, welch_psd)

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    mark = "ok  " if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f"   {detail}" if detail else ""))
    return cond


def section(t):
    print(f"\n{t}\n" + "-" * len(t))


# --------------------------------------------------------------- 1. power cal

def _gain_for(dbm, target_dbfs=-20.0, cal=-10.0):
    """Gain that puts a signal at a sensible level, well clear of the rails."""
    return int(max(0, min(71, round(target_dbfs - dbm + cal))))


def test_power_calibration():
    section("1. DSP power calibration")
    for level in (-40.0, -55.0, -70.0, -85.0):
        g = _gain_for(level)
        src = SimSource(gain_db=g, noise_dbm=-115.0, seed=11, motion=False,
                        fading=False,
                        emitters=[(2_437_000_000.0, 1.0, level, "cw", 1.0)])
        eng = SweepEngine(src, nbins=256, nfft=1024, navg=16)
        sw = eng.sweep()
        _, peak = sw.peak()
        err = peak - level
        check(f"CW at {level:6.1f} dBm (gain {g:2d}) reads {peak:7.2f} dBm",
              abs(err) < 1.0, f"error {err:+.2f} dB")

    # Total power of a wideband signal must survive being split into bins.
    src = SimSource(gain_db=15, noise_dbm=-120.0, seed=12, motion=False,
                    fading=False,
                    emitters=[(2_437_000_000.0, 5e6, -50.0, "ofdm", 1.0)])
    eng = SweepEngine(src, nbins=256, nfft=1024, navg=16)
    sw = eng.sweep()
    b = np.asarray(sw.bins_dbm)
    lo, hi = sw.bin_of(2_433e6), sw.bin_of(2_441e6)
    tot = 10 * math.log10((10 ** (b[lo:hi + 1] / 10)).sum())
    check(f"5 MHz signal integrates to {tot:.2f} dBm (injected -50.0)",
          abs(tot + 50.0) < 1.0, f"error {tot + 50.0:+.2f} dB")

    # Gain must cancel out: same signal, different gain, same dBm. This is the
    # property the whole localisation rests on - if it fails, changing gain
    # looks identical to the target moving.
    reads = []
    for g in (10, 25, 40):
        src = SimSource(gain_db=g, noise_dbm=-115.0, seed=13, motion=False,
                        fading=False,
                        emitters=[(2_437_000_000.0, 1.0, -60.0, "cw", 1.0)])
        sw = SweepEngine(src, nbins=256, nfft=1024, navg=16).sweep()
        reads.append(sw.peak()[1])
    spread = max(reads) - min(reads)
    check(f"gain 10/25/40 dB all read the same signal ({spread:.2f} dB spread)",
          spread < 1.0, f"{['%.2f' % r for r in reads]}")

    # Saturation must be detected, because a clipped sweep reads LOW and would
    # otherwise be indistinguishable from the target getting further away.
    src = SimSource(gain_db=60, noise_dbm=-115.0, seed=14, motion=False,
                    fading=False,
                    emitters=[(2_437_000_000.0, 1.0, -40.0, "cw", 1.0)])
    sw = SweepEngine(src, nbins=256, nfft=1024, navg=16).sweep()
    check(f"ADC saturation is flagged (reads {sw.peak()[1]:.1f} dBm, "
          f"injected -40.0)", sw.clipped)
    eng2 = SweepEngine(src, nbins=256, nfft=1024, navg=16)
    want = eng2.suggest_gain(sw)
    check(f"and the fix is suggested: drop gain {sw.gain_db} -> {want} dB",
          want is not None and want < sw.gain_db)

    src = SimSource(gain_db=25, noise_dbm=-115.0, seed=15, motion=False,
                    fading=False,
                    emitters=[(2_437_000_000.0, 1.0, -60.0, "cw", 1.0)])
    sw = SweepEngine(src, nbins=256, nfft=1024, navg=16).sweep()
    check("a healthy sweep is not flagged as clipped", not sw.clipped)


# ------------------------------------------------------------- 2. stitching

def test_stitching():
    section("2. tile plan and stitching")
    for mode in ("offset", "notch"):
        plan = TilePlan(2.4e9, 2.4835e9, 20e6, 0.75, lo_mode=mode)
        print(f"  {mode:6s}: {plan.n_tiles:2d} tiles of {plan.step/1e6:5.2f} MHz, "
              f"LO offset {plan.lo_offset/1e6:.2f} MHz, "
              f"{len(plan.blind_spots())} blind spots")

        gaps = [plan.edges(i + 1)[0] - plan.edges(i)[1]
                for i in range(plan.n_tiles - 1)]
        check(f"[{mode}] tile boundaries abut exactly (no gaps, no overlap)",
              all(abs(g) < 1.0 for g in gaps))
        check(f"[{mode}] tiles span exactly f_start..f_stop",
              abs(plan.edges(0)[0] - 2.4e9) < 1 and
              abs(plan.edges(plan.n_tiles - 1)[1] - 2.4835e9) < 1)

        # Every owned frequency must sit inside the trusted filter region,
        # measured from the LO - which is NOT the tile centre in offset mode.
        worst = max(abs(plan.edges(i)[k] - plan.lo_freq(i))
                    for i in range(plan.n_tiles) for k in (0, 1))
        check(f"[{mode}] owned bins stay inside the flat filter region "
              f"({worst/1e6:.2f} of {20*0.75/2:.2f} MHz)",
              worst <= 20e6 * 0.75 / 2 + 1)

    plan = TilePlan(2.4e9, 2.4835e9, 20e6, 0.75, lo_mode="offset")
    check("offset mode keeps DC clear of every owned region",
          all(not (plan.edges(i)[0] <= plan.lo_freq(i) < plan.edges(i)[1])
              for i in range(plan.n_tiles)))
    check("offset mode reports zero blind spots",
          plan.blind_spots() == [])

    # REGRESSION: a tone exactly at a tile centre used to be erased by the DC
    # notch, reading -136 dBm instead of -60. Six invisible frequencies across
    # the band, silently.
    errs = []
    for i in range(plan.n_tiles):
        fc = plan.centres[i]
        src = SimSource(gain_db=25, noise_dbm=-120.0, seed=23, motion=False,
                        fading=False, emitters=[(fc, 1.0, -60.0, "cw", 1.0)])
        sw = SweepEngine(src, nbins=256, nfft=1024, navg=16,
                         lo_mode="offset").sweep()
        errs.append(sw.peak()[1] + 60.0)
    check(f"a tone AT a tile centre is measured, not notched away "
          f"(worst error {max(map(abs, errs)):+.2f} dB)",
          all(abs(e) < 2.0 for e in errs))

    # And confirm 'notch' mode really does have the blind spot, so the
    # documented trade-off is the actual behaviour rather than a claim.
    pn = TilePlan(2.4e9, 2.4835e9, 20e6, 0.75, lo_mode="notch")
    src = SimSource(gain_db=25, noise_dbm=-120.0, seed=24, motion=False,
                    fading=False,
                    emitters=[(pn.centres[2], 1.0, -60.0, "cw", 1.0)])
    sw = SweepEngine(src, nbins=256, nfft=1024, navg=16, lo_mode="notch").sweep()
    check(f"notch mode is genuinely blind at tile centres, as documented "
          f"(reads {sw.peak()[1]:.0f} dBm)", sw.peak()[1] < -90,
          f"{len(pn.blind_spots())} spots x "
          f"{pn.blind_width_hz(1024)/1e3:.0f} kHz")

    # Tones placed at the seams must land at the right frequency and level.
    seams = [plan.edges(i)[1] for i in range(plan.n_tiles - 1)]
    errs = []
    for fs in seams:
        src = SimSource(gain_db=25, noise_dbm=-115.0, seed=21, motion=False,
                        fading=False, emitters=[(fs, 1.0, -60.0, "cw", 1.0)])
        eng = SweepEngine(src, nbins=256, nfft=1024, navg=16, lo_mode="offset")
        sw = eng.sweep()
        i, p = sw.peak()
        errs.append((abs(sw.f0_hz + i * sw.df_hz - fs), p + 60.0))
    check("tones sitting exactly on a tile seam are found at the right freq",
          all(df <= 400e3 for df, _ in errs),
          f"max freq error {max(d for d, _ in errs)/1e3:.0f} kHz")
    # Two FFT grids meet at a seam with an arbitrary sub-bin phase offset, so
    # they do not sample the continuous spectrum identically. A NARROWBAND
    # tone within one FFT bin (~20 kHz) of a seam therefore reads up to ~2.5 dB
    # off. That is 11 seams x 40 kHz = 0.5% of the band, on a link where
    # multipath fading swings 20 dB. Measured, bounded, documented, not fixed.
    check("tones on a seam are within 2.5 dB of true level",
          all(abs(dp) < 2.5 for _, dp in errs),
          f"max level error {max(abs(dp) for _, dp in errs):+.2f} dB")

    # The realistic case - anything wideband integrates over many bins and the
    # seam error washes out almost entirely.
    werr = []
    for fs in seams[:4]:
        src = SimSource(gain_db=15, noise_dbm=-125.0, seed=78, motion=False,
                        fading=False, emitters=[(fs, 20e6, -50.0, "ofdm", 1.0)])
        sw = SweepEngine(src, nbins=256, nfft=1024, navg=16,
                         lo_mode="offset").sweep()
        b = np.asarray(sw.bins_dbm)
        lo, hi = sw.bin_of(fs - 11e6), sw.bin_of(fs + 11e6)
        werr.append(10 * math.log10((10 ** (b[lo:hi + 1] / 10)).sum()) + 50.0)
    check("a 20 MHz carrier straddling a seam is within 1 dB",
          all(abs(e) < 1.0 for e in werr),
          f"max {max(map(abs, werr)):+.2f} dB")

    # A flat band must come out flat: a stitching bug shows up as a ripple.
    src = SimSource(gain_db=25, noise_dbm=-90.0, seed=22, motion=False,
                    fading=False, emitters=[])
    eng = SweepEngine(src, nbins=256, nfft=1024, navg=32, lo_mode="offset")
    b = np.asarray(eng.sweep().bins_dbm)
    means = [float(c.mean()) for c in np.array_split(b, eng.plan.n_tiles)]
    ripple = max(means) - min(means)
    check(f"flat noise stitches flat across all {eng.plan.n_tiles} tiles "
          f"({ripple:.2f} dB ripple)", ripple < 1.5)


# ------------------------------------------------------------------- 3. DC

def test_dc_repair():
    section("3. DC / LO-leakage repair")
    p = np.full(1024, 1e-9)
    p[512] = 1.0                       # a huge fake DC spike
    p[511] = p[513] = 0.1
    out = repair_dc(p, notch=2)
    check("DC spike is interpolated away",
          out[512] < 1e-6, f"{p[512]:.1e} -> {out[512]:.1e}")
    check("bins away from DC are untouched",
          out[100] == p[100] and out[900] == p[900])

    # End to end: the simulator injects LO leakage, so if it were leaking
    # through we would see a fake emitter at every tile's LO frequency.
    for mode in ("offset", "notch"):
        src = SimSource(gain_db=25, noise_dbm=-95.0, seed=31, motion=False,
                        fading=False, emitters=[])
        eng = SweepEngine(src, nbins=256, nfft=1024, navg=16, lo_mode=mode)
        sw = eng.sweep()
        b = np.asarray(sw.bins_dbm)
        worst = max((b[max(0, sw.bin_of(f) - 1):sw.bin_of(f) + 2].max()
                     - sw.noise_dbm) for f in eng.plan.centres)
        check(f"[{mode}] no fake emitter at any of the {eng.plan.n_tiles} tile "
              f"centres (worst {worst:.1f} dB above noise)", worst < 8.0)


# ---------------------------------------------------------------- 4-5. codec

def test_codec():
    section("4. wire format round trip")
    rng = np.random.default_rng(7)
    bins = (-100 + rng.normal(0, 3, 256)).tolist()
    bins[100] = -42.0
    sw = Sweep(seq=1234, t=time.time(), node_id=3, f0_hz=2400163085.0,
               df_hz=326171.875, bins_dbm=bins, noise_dbm=-100.4, gain_db=40)

    back = decode_sweep(encode_sweep(sw))
    check("seq / node / gain survive", back.seq == 1234 and back.node_id == 3
          and back.gain_db == 40)
    check("frequency grid survives",
          abs(back.f0_hz - sw.f0_hz) < 1000 and abs(back.df_hz - sw.df_hz) < 1)
    check("timestamp survives", abs(back.t - sw.t) < 0.002,
          f"delta {abs(back.t - sw.t)*1000:.1f} ms")
    err = np.abs(np.array(back.bins_dbm) - np.array(sw.bins_dbm))
    check(f"quantisation error <= 0.25 dB (max {err.max():.3f})",
          err.max() <= 0.2501)
    check("noise floor survives", abs(back.noise_dbm - sw.noise_dbm) < 0.06)

    sw.clipped = True
    check("clip flag survives", decode_sweep(encode_sweep(sw)).clipped)
    sw.clipped = False

    # Values that break naive codecs.
    nasty = Sweep(seq=65535, t=time.time(), node_id=255, f0_hz=2.4e9,
                  df_hz=326171.875,
                  bins_dbm=[float("nan"), float("-inf"), -999.0, 50.0]
                           + [-100.0] * 252,
                  noise_dbm=-100.0, gain_db=-5)
    b2 = decode_sweep(encode_sweep(nasty))
    check("NaN / inf / out-of-range bins do not corrupt the frame",
          b2.nbins == 256 and all(math.isfinite(v) for v in b2.bins_dbm))
    check("seq wraps at 65535 without breaking", b2.seq == 65535)

    section("5. codec robustness")
    good = encode_sweep(sw)
    bad = 0
    for cut in range(0, len(good), 7):
        try:
            decode_sweep(good[:cut])
        except W.DecodeError:
            bad += 1
        except Exception as e:
            check(f"truncation at {cut} raised {type(e).__name__}", False, str(e))
            return
    check(f"every truncation raises DecodeError, never a crash ({bad} cases)",
          True)

    rng = np.random.default_rng(99)
    crashes = decoded = rejected = 0
    for _ in range(400):
        m = bytearray(good)
        for _ in range(rng.integers(1, 6)):
            m[rng.integers(0, len(m))] ^= 1 << rng.integers(0, 8)
        try:
            decode_sweep(bytes(m))
            decoded += 1
        except W.DecodeError:
            rejected += 1
        except Exception:
            crashes += 1
    check(f"400 corrupted frames: {rejected} rejected, {decoded} accepted, "
          f"{crashes} crashes", crashes == 0)


# ------------------------------------------------------------- 6. bandwidth

def test_bandwidth():
    section("6. bandwidth against the 17 kbit/s ceiling")
    limit_bps = 17_000 / 8.0
    print(f"  ceiling {limit_bps:.0f} B/s\n")

    src = SimSource(gain_db=40, noise_dbm=-95.0, seed=41)
    for nbins in (128, 256, 512):
        eng = SweepEngine(src, nbins=nbins, nfft=1024, navg=8)
        sizes = [len(encode_sweep(eng.sweep())) for _ in range(8)]
        mean = sum(sizes) / len(sizes)
        wire = encoded_size(nbins, eng.sweep())
        worst = encoded_size(nbins)
        print(f"  {nbins:4d} bins: payload {mean:6.1f} B mean, "
              f"wire {wire:4d} B -> {limit_bps/wire:5.2f} sweeps/s  "
              f"(worst case {worst} B -> {limit_bps/worst:.2f}/s)")
        if nbins == 256:
            check("256 bins fits several sweeps per second at 17 kbit/s",
                  limit_bps / wire >= 3.0, f"{limit_bps/wire:.2f} sweeps/s")
            check("even an incompressible 256-bin sweep fits in one second",
                  worst < limit_bps, f"{worst} B vs {limit_bps:.0f} B/s")

    # Compression must actually be earning its place.
    eng = SweepEngine(src, nbins=256, nfft=1024, navg=8)
    sw = eng.sweep()
    raw = len(encode_sweep(sw, compress=False))
    comp = len(encode_sweep(sw, compress=True))
    check(f"compression helps: {raw} B -> {comp} B "
          f"({100*(1-comp/raw):.0f}% saved)", comp < raw)

    # The budget has to hold when the SDR outruns the link, which it will.
    # Offering 40 sweeps/s into a 17 kbit/s pipe must drop, not queue.
    from sdr_tx import BURST_SECS, ByteBudget
    bud = ByteBudget(limit_bps, burst_secs=BURST_SECS)
    t0 = time.time()
    allowed = refused = 0
    while time.time() - t0 < 3.0:
        if bud.allow(180):
            allowed += 1
        else:
            refused += 1
        time.sleep(0.025)                       # offer 40 sweeps/s
    el = time.time() - t0

    # A token bucket's actual guarantee is bytes <= rate*time + burst, not a
    # flat cap on every window - the bucket starts full so a short measurement
    # sees the burst as inflated throughput. Test the real contract.
    ceiling = limit_bps * (el + BURST_SECS)
    check(f"budget holds under overload: {allowed} sent, {refused} dropped, "
          f"{bud.spent} B in {el:.1f}s (contract allows {ceiling:.0f} B)",
          bud.spent <= ceiling and refused > 0)
    steady = bud.spent / (el + BURST_SECS)
    check(f"steady-state rate stays under 17 kbit/s "
          f"({steady*8/1000:.2f} kbit/s)", steady <= limit_bps)
    check("excess is dropped, not queued "
          f"({100*refused/(allowed+refused):.0f}% dropped)", refused > allowed)


# ------------------------------------------------- 7-8. detection & tracking

def test_detection_tracking():
    section("7. emitter detection")
    src = SimSource(gain_db=40, noise_dbm=-100.0, seed=51, motion=False,
                    fading=False, emitters=[
                        (2_412_000_000.0, 16e6, -55.0, "ofdm", 1.0),
                        (2_437_000_000.0, 1.0, -65.0, "cw", 1.0),
                        (2_480_000_000.0, 1.0, -75.0, "cw", 1.0)])
    eng = SweepEngine(src, nbins=256, nfft=1024, navg=16)
    sw = eng.sweep()
    dets = find_emitters(sw, snr_db=8.0)
    found = sorted(d["f_hz"] / 1e6 for d in dets)
    check(f"all 3 emitters detected ({len(dets)} found)", len(dets) == 3,
          f"at {['%.1f' % f for f in found]} MHz")
    for want in (2412.0, 2437.0, 2480.0):
        near = min((abs(f - want) for f in found), default=999)
        check(f"emitter at {want} MHz located within 1 MHz", near < 1.0,
              f"off by {near*1000:.0f} kHz")
    check("detections are ordered strongest first",
          all(dets[i]["dbm"] >= dets[i + 1]["dbm"] for i in range(len(dets) - 1)))

    # The noise floor must survive a band that is mostly occupied. Three
    # overlapping WiFi carriers cover 60 of 83.5 MHz, which is where a median
    # estimator silently reports a floor ~15 dB too high.
    src = SimSource(gain_db=40, noise_dbm=-115.0, seed=52, motion=False,
                    fading=False, emitters=[
                        (2_412_000_000.0, 20e6, -72.0, "ofdm", 1.0),
                        (2_437_000_000.0, 20e6, -74.0, "ofdm", 1.0),
                        (2_462_000_000.0, 20e6, -76.0, "ofdm", 1.0)])
    eng = SweepEngine(src, nbins=256, nfft=1024, navg=16)
    sw = eng.sweep()
    true_floor = -115.0 - 10 * math.log10(20e6 / sw.df_hz)
    err = sw.noise_dbm - true_floor
    occ = 60.0 / 83.5 * 100
    check(f"noise floor correct with {occ:.0f}% of the band occupied "
          f"({sw.noise_dbm:.1f} vs true {true_floor:.1f} dBm)",
          abs(err) < 3.0, f"error {err:+.2f} dB")
    med_err = float(np.median(sw.bins_dbm)) - true_floor
    print(f"        (a median estimator would have said "
          f"{float(np.median(sw.bins_dbm)):.1f} dBm, {med_err:+.1f} dB off)")

    # An empty band must still read right - the percentile must not bias low.
    src = SimSource(gain_db=40, noise_dbm=-115.0, seed=53, motion=False,
                    fading=False, emitters=[])
    sw = SweepEngine(src, nbins=256, nfft=1024, navg=16).sweep()
    err = sw.noise_dbm - true_floor
    check(f"noise floor still right on an empty band ({sw.noise_dbm:.1f} dBm)",
          abs(err) < 1.5, f"error {err:+.2f} dB")

    # Fragmentation: a faded wideband carrier must stay ONE emitter.
    rng = np.random.default_rng(54)
    bins = np.full(256, -115.0)
    lo, hi = 40, 100                          # a 60-bin wide carrier
    bins[lo:hi] = -90.0 + rng.normal(0, 6, hi - lo)   # heavy fading
    bins[55:58] = -115.0                      # three deep nulls inside it
    bins[70:73] = -115.0
    sw2 = Sweep(1, time.time(), 1, 2.4e9, 326171.875, bins.tolist(), -115.0, 40)
    frag = find_emitters(sw2, snr_db=8.0, merge_gap_bins=0)
    whole = find_emitters(sw2, snr_db=8.0, merge_gap_bins=3)
    check(f"gap bridging keeps a faded carrier as one emitter "
          f"({len(frag)} fragments -> {len(whole)})",
          len(whole) == 1 and len(frag) > 1)

    # Channel labels must pick the NEAREST channel, not the first that fits.
    for f_mhz, want in ((2412.0, "WiFi ch1"), (2460.8, "WiFi ch11"),
                        (2437.0, "WiFi ch6"), (2426.1, "BLE adv 38"),
                        (2480.0, "BLE adv 39")):
        got = label_for(f_mhz * 1e6)
        check(f"{f_mhz} MHz labelled {got!r}", got == want, f"want {want!r}")
    # 2404.5 MHz: 7.5 MHz from WiFi ch1 and 2.5 MHz from BLE adv 37, so
    # outside both tolerances. (2447 would NOT work - that is exactly ch8.)
    check("a frequency with nothing near it gets no label",
          label_for(2_404_500_000.0) == "", f"got {label_for(2_404_500_000.0)!r}")

    section("8. RSSI tracking and relative range")
    # A known walk: emitter power falls 20 dB then rises 20 dB. The relative
    # range must follow the path-loss law, not just wiggle in the right way.
    n = 2.5
    tr = None
    t0 = time.time()
    profile = [0, -5, -10, -15, -20, -15, -10, -5, 0]
    for k, drop in enumerate(profile):
        for rep in range(6):                    # 6 sweeps at each position
            t = t0 + k * 6 * 0.2 + rep * 0.2
            v = -50.0 + drop
            if tr is None:
                tr = EmitterTrack(2.437e9, v, t)
            else:
                tr.update(2.437e9, v, t)

    rel = tr.relative_range(n)
    check(f"back at the start, relative range returns to 1.0 ({rel:.3f})",
          abs(rel - 1.0) < 0.05)

    # Now check the middle of the walk against the closed form.
    tr2 = None
    for rep in range(12):
        t = t0 + rep * 0.2
        tr2 = EmitterTrack(2.437e9, -50.0, t) if tr2 is None else tr2
        tr2.update(2.437e9, -50.0, t)
    for rep in range(12):
        tr2.update(2.437e9, -70.0, t0 + 3 + rep * 0.2)
    expect = 10 ** (20.0 / (10 * n))            # 20 dB weaker
    got = tr2.relative_range(n)
    check(f"20 dB weaker reads as {expect:.2f}x further ({got:.2f}x)",
          abs(got - expect) < 0.1)

    # Trend direction.
    up = EmitterTrack(2.437e9, -80.0, t0)
    for k in range(20):
        up.update(2.437e9, -80.0 + k * 0.8, t0 + k * 0.25)
    check(f"a rising signal reads as warmer ({up.verdict().strip()}, "
          f"{up.trend_db_per_s():+.2f} dB/s)", up.trend_db_per_s() > 0.4)

    down = EmitterTrack(2.437e9, -50.0, t0)
    for k in range(20):
        down.update(2.437e9, -50.0 - k * 0.8, t0 + k * 0.25)
    check(f"a falling signal reads as colder ({down.verdict().strip()}, "
          f"{down.trend_db_per_s():+.2f} dB/s)", down.trend_db_per_s() < -0.4)

    # The median filter has to beat a plain mean under Rayleigh fading, or
    # there is no point paying for it.
    rng = np.random.default_rng(61)
    truth = -60.0
    fades = truth + 20 * np.log10(np.abs(rng.normal(0, .707, 400)
                                         + 1j * rng.normal(0, .707, 400)))
    tr3 = None
    for k, v in enumerate(fades):
        tr3 = EmitterTrack(2.4e9, v, t0 + k * .1) if tr3 is None \
            else (tr3.update(2.4e9, v, t0 + k * .1) or tr3)
    med_err = abs(float(np.median(list(tr3.med_hist))) - float(np.median(fades)))
    check("median filter tracks the fading median without bias",
          med_err < 1.5, f"error {med_err:.2f} dB")
    spread_raw = float(np.std(fades))
    spread_med = float(np.std(list(tr3.med_hist)))
    check(f"median filter cuts fading spread {spread_raw:.1f} -> "
          f"{spread_med:.1f} dB", spread_med < spread_raw * 0.8)


# ----------------------------------------------------------------- 9. loss

def test_loss_accounting():
    section("9. sequence gap / loss accounting")
    st = W.SeqTracker()
    for s in range(1, 101):
        st.observe(s)
    check("no loss reported on a clean run", st.loss_pct == 0.0)

    st = W.SeqTracker()
    seen = 0
    for s in range(1, 201):
        if s % 10 == 0:
            continue                            # drop every 10th
        st.observe(s)
        seen += 1
    check(f"10% drop measured as {st.loss_pct:.1f}%",
          9.0 < st.loss_pct < 11.0)

    st = W.SeqTracker()
    st.observe(65530)
    for s in [65533, 65534, 65535, 0, 1, 2]:
        st.observe(s)
    check(f"counter wrap at 65535 is not read as 65000 lost sweeps "
          f"({st.lost} lost)", st.lost == 2)

    st = W.SeqTracker()
    st.observe(5000)
    st.observe(1)                               # scanner restarted
    check("a scanner restart does not report a huge phantom loss",
          st.lost == 0, f"{st.lost} lost")

    # And the state object must survive out-of-order / duplicate frames.
    state = RxState()
    base = Sweep(1, time.time(), 1, 2.4e9, 326171.875,
                 [-100.0] * 256, -100.0, 40)
    for s in (1, 2, 2, 3, 7):
        base.seq = s
        state.add(base, 200)
    check("duplicate sequence numbers are not counted as loss",
          state.seqs.lost == 3, f"{state.seqs.lost} lost (expect 3 for 4,5,6)")


# ------------------------------------------------------ 11. bands config

def test_bands():
    section("11. band configuration")
    from bands import (BAND_PRESETS, DEFAULT_BANDS, gain_limits_db,
                       resolve_bands, validate)
    d = resolve_bands(None)
    check(f"defaults are {[b.name for b in d]}",
          [b.name for b in d] == ["ism24", "ism58", "ism915"])
    check("default ranges: 2400-2483.5 / 5725-5875 / 915-928 MHz",
          [(b.f_start_hz, b.f_stop_hz) for b in d] ==
          [(2400e6, 2483.5e6), (5725e6, 5875e6), (915e6, 928e6)])

    b = resolve_bands(["ism24", "gnss_l1,ism433"])
    check("--band takes presets, repeated and comma-separated",
          [x.name for x in b] == ["ism24", "gnss_l1", "ism433"])

    c = resolve_bands(["mine=433.05:434.79:64:30:2"])[0]
    check("custom spec name=start:stop:bins:gain:every parses",
          (c.name, c.f_start_hz, c.f_stop_hz, c.bins, c.gain_db, c.every) ==
          ("mine", 433.05e6, 434.79e6, 64, 30.0, 2))
    c2 = resolve_bands(["1240:1300"])[0]
    check(f"custom spec without a name is auto-named ({c2.name})",
          c2.name == "1240-1300" and c2.bins == 256)

    for bad, why in ((["ism24,ism24"], "duplicate"),
                     (["x=2500:2400"], "stop below start"),
                     (["nosuchband"], "unknown preset"),
                     (["x=1:2:3"], "3 bins")):
        try:
            resolve_bands(bad)
            ok = False
        except ValueError:
            ok = True
        check(f"rejects a bad spec ({why})", ok)

    check("presets were not mutated by parsing (deep copies)",
          BAND_PRESETS["ism915"].gain_db == 20)
    check("AD936x gain tables: 73 dB < 1.3 GHz, 71 to 4 GHz, 62 above",
          gain_limits_db(915e6)[1] == 73 and gain_limits_db(2.4e9)[1] == 71
          and gain_limits_db(5.8e9)[1] == 62)
    w = validate(resolve_bands(["ism58", "fpv58_full"]))
    check("overlapping bands produce a warning, not an error",
          any("overlap" in x for x in w))


# ------------------------------------------------ 12. multi-band sweeping

class _Args:
    """What argparse would hand build_engines, with sim defaults."""

    def __init__(self, **kw):
        from sdr_scan import (DEF_CAL_OFFSET_DB, DEF_DC_NOTCH, DEF_GAIN_DB,
                              DEF_NAVG, DEF_NFFT, DEF_SAMPLE_RATE, DEF_USABLE)
        self.band = None
        self.gain = DEF_GAIN_DB
        self.agc = False
        self.nfft = DEF_NFFT
        self.navg = DEF_NAVG
        self.sample_rate = DEF_SAMPLE_RATE
        self.usable = DEF_USABLE
        self.cal_offset = DEF_CAL_OFFSET_DB
        self.dc_notch = DEF_DC_NOTCH
        self.smooth = 0.0
        self.node_id = 1
        self.lo_mode = "offset"
        self.lo_guard = 500e3
        self.__dict__.update(kw)


def test_multiband():
    from sdr_scan import build_engines
    section("12. multi-band sweeping on one radio")
    src = SimSource(gain_db=40, noise_dbm=-115.0, seed=81, motion=False,
                    fading=False, emitters=[
                        (2_437_000_000.0, 1.0, -60.0, "cw", 1.0),
                        (5_806_000_000.0, 1.0, -65.0, "cw", 1.0),
                        (921_400_000.0, 1.0, -70.0, "cw", 1.0)])
    engines, skipped, bands = build_engines(_Args(), src, log=open(os.devnull, "w"))
    check(f"3 engines built, none skipped ({[e.name for e in engines]})",
          len(engines) == 3 and not skipped)
    check("band ids are 0, 1, 2 in config order",
          [e.band_id for e in engines] == [0, 1, 2])
    tiles = {e.name: e.plan.n_tiles for e in engines}
    check(f"tile counts {tiles}",
          tiles == {"ism24": 12, "ism58": 22, "ism915": 2})

    # Interleave the bands twice over, the way sdr_tx does, and make sure
    # each band's gain is the band's own and each tone lands in its band.
    want = {"ism24": (2437e6, -60.0, 40), "ism58": (5806e6, -65.0, 40),
            "ism915": (921.4e6, -70.0, 20)}
    for rnd in range(2):
        for eng in engines:
            sw = eng.sweep()
            f, lvl, g = want[eng.name]
            i, _ = sw.peak()
            fpk = sw.f0_hz + i * sw.df_hz
            # Integrate +/-2 bins, as the tracker does: in 100 kHz bins a CW
            # tone near a bin edge splits across two, so the single peak bin
            # can read up to 3 dB low while the total is right.
            b = np.asarray(sw.bins_dbm)
            p = 10 * math.log10((10 ** (b[max(0, i-2):i+3] / 10)).sum())
            if rnd == 1:
                check(f"{eng.name}: tone at {f/1e6:.1f} MHz found at "
                      f"{fpk/1e6:.2f} MHz, {p:.1f} dBm, gain {sw.gain_db}",
                      abs(fpk - f) <= sw.df_hz and abs(p - lvl) < 1.0
                      and sw.gain_db == g and sw.band_id == eng.band_id,
                      f"want {lvl} dBm @ gain {g}")

    # A band the radio cannot tune is skipped, and the others keep their ids.
    src2 = SimSource(motion=False, fading=False, lo_range=(325e6, 3.8e9))
    engines2, skipped2, _ = build_engines(_Args(), src2,
                                          log=open(os.devnull, "w"))
    check("stock AD9363 range (325-3800 MHz): ism58 skipped, not fatal",
          [e.name for e in engines2] == ["ism24", "ism915"]
          and [s[1].name for s in skipped2] == ["ism58"])
    check("ids survive a skip (ism915 is still band 2)",
          [e.band_id for e in engines2] == [0, 2])
    check("the skip reason says how to fix it",
          "ad9364" in skipped2[0][2])

    # Auto-gain on a WIDEBAND signal. The old rule aimed the strongest output
    # bin at -15 dBFS; a 20 MHz carrier's strongest bin is ~18 dB under what
    # the ADC sees, so it walked the gain up into clipping and back, forever.
    src_ag = SimSource(gain_db=40, noise_dbm=-115.0, seed=83, motion=False,
                       fading=False,
                       emitters=[(2_437_000_000.0, 20e6, -62.0, "ofdm", 1.0)])
    eng_ag = SweepEngine(src_ag, nbins=256, gain_db=40)
    hist, clips = [], 0
    for _ in range(12):
        sw = eng_ag.sweep()
        clips += int(sw.clipped)
        want = eng_ag.suggest_gain(sw)
        if want is not None:
            eng_ag.gain_db = float(want)
        hist.append(sw.gain_db)
    check(f"auto-gain on a 20 MHz carrier settles without clipping "
          f"(gain path {hist[:6]}..., {clips} clipped sweeps)",
          clips == 0 and len(set(hist[-5:])) == 1, f"final {hist[-1]} dB")
    quiet = SweepEngine(SimSource(gain_db=20, seed=84, motion=False,
                                  fading=False, emitters=[]),
                        nbins=256, gain_db=20)
    path = []
    for _ in range(8):
        sw = quiet.sweep()
        want = quiet.suggest_gain(sw)
        if want is not None:
            quiet.gain_db = float(want)
        path.append(sw.gain_db)
    check(f"and on an empty band it climbs in <=10 dB steps ({path})",
          path[-1] > 50 and all(b - a <= 10 for a, b in zip(path, path[1:])))

    # --agc must leave gain alone rather than fight the AGC.
    e3, _, _ = build_engines(_Args(agc=True), SimSource(motion=False),
                             log=open(os.devnull, "w"))
    check("with --agc no band sets a manual gain",
          all(e.gain_db is None for e in e3))


# -------------------------------------------- 13. the Pluto driver path

class _FakeIioDev:
    def __init__(self, owner):
        self.owner = owner

    def set_kernel_buffers_count(self, n):
        self.owner.kernel_buffers = n


class _FakePluto:
    """Stands in for adi.Pluto. Models the two things that matter here:

    * libiio's kernel buffer queue: rx() returns the OLDEST queued capture,
      and the queue refills at whatever the LO is now. With 4 buffers a
      retune leaves stale captures in the queue.
    * the AD936x gain table following the LO: a gain outside the current
      table's range is refused, like the driver.
    """

    def __init__(self, uri=None, tone_hz=2_437_000_000.0, lo_range=(70e6, 6e9)):
        self.kernel_buffers = 4
        self.sample_rate = 20_000_000
        self.rx_rf_bandwidth = 18_000_000
        self.rx_buffer_size = 8192
        self.gain_control_mode_chan0 = "manual"
        self._gain = 40
        self._lo = 2_400_000_000
        self.tone_hz = tone_hz
        self.lo_range = lo_range
        self._rxadc = _FakeIioDev(self)
        self.queue = []
        self.rng = np.random.default_rng(5)
        self.lo_writes = []

    @property
    def rx_lo(self):
        return self._lo

    @rx_lo.setter
    def rx_lo(self, v):
        self._lo = int(v)
        self.lo_writes.append(int(v))

    @property
    def rx_hardwaregain_chan0(self):
        return self._gain

    @rx_hardwaregain_chan0.setter
    def rx_hardwaregain_chan0(self, v):
        from bands import gain_limits_db
        lo, hi = gain_limits_db(self._lo)
        if not lo <= v <= hi:
            raise OSError(22, f"gain {v} invalid at LO {self._lo}")
        self._gain = v

    def _capture(self):
        n = self.rx_buffer_size
        k = np.arange(n)
        x = (self.rng.normal(0, 2, n) + 1j * self.rng.normal(0, 2, n))
        off = self.tone_hz - self._lo
        if abs(off) < self.sample_rate / 2:
            x += 400 * np.exp(2j * np.pi * off / self.sample_rate * k)
        return np.round(x.real) + 1j * np.round(x.imag)   # integer counts

    def rx(self):
        while len(self.queue) < self.kernel_buffers:
            self.queue.append(self._capture())
        out = self.queue.pop(0)
        self.queue.append(self._capture())
        return out

    def rx_destroy_buffer(self):
        self.queue = []

    def _get_iio_attr_str(self, ch, attr, out):
        lo, hi = self.lo_range
        return f"[{int(lo)} 1 {int(hi)}]"


def _pluto_with(fake, **kw):
    """A PlutoSource whose adi.Pluto is our fake."""
    import types
    import sdr_scan
    mod = types.ModuleType("adi")
    mod.Pluto = lambda uri=None: fake
    saved = sys.modules.get("adi")
    sys.modules["adi"] = mod
    try:
        return sdr_scan.PlutoSource(**kw)
    finally:
        if saved is None:
            del sys.modules["adi"]
        else:
            sys.modules["adi"] = saved


def test_pluto_path():
    import sdr_scan as sdr_scan_mod
    section("13. Pluto/AD9363 driver path (mocked radio)")
    fake = _FakePluto(tone_hz=2_437_000_000.0)
    src = _pluto_with(fake, gain_db=40)
    check("kernel buffer queue cut to 1 so retunes cannot return stale data",
          fake.kernel_buffers == 1)
    eng = SweepEngine(src, nbins=256, nfft=1024, navg=8)
    sw = eng.sweep()
    i, p = sw.peak()
    ghosts = np.sort(np.asarray(sw.bins_dbm))[-3] - sw.noise_dbm
    check(f"tone found at {(sw.f0_hz + i*sw.df_hz)/1e6:.2f} MHz, no ghosts "
          f"(3rd-highest bin {ghosts:.1f} dB over floor)",
          abs(sw.f0_hz + i * sw.df_hz - 2437e6) < 400e3 and ghosts < 15)

    # The old behaviour, reproduced: 4 kernel buffers, discard only 1.
    fake4 = _FakePluto(tone_hz=2_437_000_000.0)
    fake4._rxadc = type("NoSet", (), {})()          # cannot set the count
    src4 = _pluto_with(fake4, gain_db=40, flush_buffers=1)
    check("if the count cannot be set, --flush is raised to 4 automatically",
          src4.flush_buffers == 4)
    src4.flush_buffers = 1                            # force the old bug
    sw4 = SweepEngine(src4, nbins=256, nfft=1024, navg=8).sweep()
    i4, _ = sw4.peak()
    moved = (sw4.f0_hz + i4 * sw4.df_hz - 2437e6) / 1e6
    check(f"(model of the old setup - 4 queued buffers, flush 1 - puts the "
          f"tone {moved:+.1f} MHz from where it really is)", abs(moved) > 5)

    check("ADC counts detected as counts (x 1/2048), by integrality",
          src._scale == 1.0 / 2048.0)
    fake_q = _FakePluto()
    fake_q.rng = np.random.default_rng(1)
    fake_q._capture = lambda: (np.round(fake_q.rng.normal(0, 0.4, 8192))
                               + 1j * np.round(fake_q.rng.normal(0, 0.4, 8192)))
    src_q = _pluto_with(fake_q, gain_db=0)
    src_q.capture(8192)
    check("a near-silent first capture no longer flips the scale by 66 dB",
          src_q._scale == 1.0 / 2048.0)

    lr = src.lo_range_hz()
    check(f"LO range read from the radio ({lr[0]/1e6:.0f}-{lr[1]/1e6:.0f} MHz)",
          lr == (70e6, 6e9))

    # Gain ordering: 915 at 20 dB -> 5.8 GHz asking for 70 dB (max there 62).
    fake5 = _FakePluto(tone_hz=5_806_000_000.0)
    src5 = _pluto_with(fake5, gain_db=20)
    e915 = SweepEngine(src5, nbins=128, f_start=915e6, f_stop=928e6,
                       gain_db=20, band_id=2)
    e58 = SweepEngine(src5, nbins=256, f_start=5725e6, f_stop=5875e6,
                      gain_db=70, band_id=1)
    try:
        s1 = e915.sweep()
        s2 = e58.sweep()
        s3 = e915.sweep()
        ok = (s1.gain_db, s2.gain_db, s3.gain_db) == (20, 62, 20)
        detail = f"gains {s1.gain_db}/{s2.gain_db}/{s3.gain_db}"
    except OSError as e:
        ok, detail = False, str(e)
    check("band hop 915 -> 5.8 GHz -> 915: each band gets its own gain, "
          "tuned before set, clamped to the table", ok, detail)

    # The radio dropping off USB mid-run (what a Pi 5 USB brown-out does):
    # the next rx() is a broken pipe, the SDR takes a while to come back, and
    # when it does every setting has to be put back.
    import types
    made = []

    class _Dying(_FakePluto):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.calls = 0

        def rx(self):
            self.calls += 1
            if len(made) == 1 and self.calls > 30:
                raise BrokenPipeError(32, "Broken pipe")
            return super().rx()

    attempts = {"n": 0}

    def factory(uri=None):
        attempts["n"] += 1
        if attempts["n"] == 2:              # still rebooting on the first retry
            raise Exception("No device found with name cf-ad9361-lpc")
        f = _Dying(tone_hz=2_437_000_000.0)
        made.append(f)
        return f
    mod = types.ModuleType("adi")
    mod.Pluto = factory
    saved = sys.modules.get("adi")
    sys.modules["adi"] = mod
    try:
        srcd = sdr_scan_mod.PlutoSource(gain_db=33)
        engd = SweepEngine(srcd, nbins=256, nfft=1024, navg=8, gain_db=33)
        engd.sweep()
        dropped = False
        try:
            engd.sweep()
        except OSError:
            dropped = True
        took = srcd.reconnect(log=open(os.devnull, "w")) if dropped else 0
        sw_after = engd.sweep()
        i, _ = sw_after.peak()
        ok = (dropped and srcd.reconnects == 1 and len(made) == 2
              and made[1].kernel_buffers == 1 and sw_after.gain_db == 33
              and abs(sw_after.f0_hz + i * sw_after.df_hz - 2437e6) < 400e3)
        detail = f"back after {took:.1f} s, gain {sw_after.gain_db}"
    finally:
        if saved is None:
            del sys.modules["adi"]
        else:
            sys.modules["adi"] = saved
    check("a broken pipe mid-run: reconnects through a failed retry, puts "
          "every setting back, sweeps carry on", ok, detail)

    # The loop seen on the real board: after a drop the SDR takes the
    # connection and every setting, then hangs up when a buffer is opened.
    # Reconnect must not call that "back" - only once samples really flow.
    import io as _io
    stuck = {"left": 1}

    class _Stuck(_FakePluto):
        def rx(self):
            if stuck["left"] > 0:
                stuck["left"] -= 1
                raise BrokenPipeError(32, "Broken pipe")
            return super().rx()

    seq = {"n": 0}

    def factory2(uri=None):
        seq["n"] += 1
        return (_FakePluto if seq["n"] == 1 else _Stuck)(tone_hz=2_437_000_000.0)
    mod2 = types.ModuleType("adi")
    mod2.Pluto = factory2
    saved = sys.modules.get("adi")
    sys.modules["adi"] = mod2
    try:
        srcs = sdr_scan_mod.PlutoSource(gain_db=33)
        buf = _io.StringIO()
        srcs.reconnect(log=buf)
        sw_s = SweepEngine(srcs, nbins=256, nfft=1024, navg=8,
                           gain_db=33).sweep()
        ok = (seq["n"] == 3 and srcs.reconnects == 1
              and "will not stream" in buf.getvalue() and sw_s.nbins == 256)
        detail = buf.getvalue().strip().splitlines()[0][:90] if buf.getvalue() else ""

        stuck["left"] = 10 ** 6                      # never streams
        seq["n"] = 1
        try:
            sdr_scan_mod.PlutoSource(gain_db=33)
            refused = False
        except sdr_scan_mod.StreamStuck:
            refused = True
    finally:
        if saved is None:
            del sys.modules["adi"]
        else:
            sys.modules["adi"] = saved
    check("a reconnect that can't stream isn't reported as back; it retries "
          "until samples flow", ok, detail)
    check("an SDR that won't stream at start-up is refused with a clear error",
          refused)

    # Board temperatures, read the way libiio exposes them on a Pluto.
    class _A:
        def __init__(self, v):
            self.value = str(v)

    class _Ch:
        def __init__(self, **attrs):
            self.attrs = {k: _A(v) for k, v in attrs.items()}

    class _Dev:
        def __init__(self, **chans):
            self.chans = chans

        def find_channel(self, name, is_output=False):
            return self.chans.get(name)

    class _Ctx:
        def __init__(self, devs):
            self.devs = devs

        def find_device(self, name):
            return self.devs.get(name)

    fake_t = _FakePluto()
    src_t = _pluto_with(fake_t, gain_db=40)
    no_ctx = src_t.temps()
    fake_t.ctx = _Ctx({
        "ad9361-phy": _Dev(temp0=_Ch(input=51234)),
        "xadc": _Dev(temp0=_Ch(raw=2600, offset=-2219, scale=123.040771484))})
    both = src_t.temps()
    fake_t.ctx = _Ctx({"ad9361-phy": _Dev(temp0=_Ch(input=48000))})
    rf_only = src_t.temps()
    check(f"SDR temperatures read and scaled ({both})",
          no_ctx is None and both == {"rf": 51.2, "fpga": 46.9}
          and rf_only == {"rf": 48.0})

    # pyadi says only "No device found"; the real reason gets dug out.
    import types as _types
    fake_iio = _types.ModuleType("iio")

    def _ctx_fail(uri):
        raise TimeoutError(110, "Connection timed out")
    fake_iio.Context = _ctx_fail
    saved_iio = sys.modules.get("iio")
    sys.modules["iio"] = fake_iio
    try:
        why = src_t._why(Exception("No device found"))
        kept = src_t._why(Exception("No device found with name x"))
    finally:
        if saved_iio is None:
            del sys.modules["iio"]
        else:
            sys.modules["iio"] = saved_iio
    check(f"a failed reconnect says why ({why!r})",
          "timed out" in why and kept == "No device found with name x")

    import base_rx as _b
    h = _b.sdr_health({"temp": [51.2, 68.9], "sdr_drops": 3})
    check(f"the laptop shows SDR temps and drops ({h!r})",
          h == "SDR 51/69 C  drops 3" and _b.sdr_health({}) == ""
          and _b.sdr_health({"temp": None, "sdr_drops": 0}) == "")


# --------------------------------------------------- 14. wire format v2

def test_wire_v2():
    section("14. wire format v2 (band id) and v1 compatibility")
    check(f"v2 header is {W.HDR_LEN} B, v1 was {W.HDR_LEN_V1}",
          W.HDR_LEN == 27 and W.HDR_LEN_V1 == 26)
    rng = np.random.default_rng(3)
    sw = Sweep(seq=77, t=time.time(), node_id=1, f0_hz=5725292969.0,
               df_hz=585937.5, bins_dbm=(-110 + rng.normal(0, 3, 256)).tolist(),
               noise_dbm=-110.0, gain_db=40, band_id=1)
    back = decode_sweep(encode_sweep(sw))
    check("band id survives the round trip", back.band_id == 1)
    check("5.8 GHz f0 survives (kHz field has room)",
          abs(back.f0_hz - sw.f0_hz) < 1000)
    v1 = decode_sweep(encode_sweep(sw, version=1))
    check("a v1 frame (old Pi build) still decodes, as band 0",
          v1.band_id == 0 and v1.seq == 77)
    crashes = 0
    good = encode_sweep(sw)
    for _ in range(400):
        m = bytearray(good)
        for _ in range(rng.integers(1, 6)):
            m[rng.integers(0, len(m))] ^= 1 << rng.integers(0, 8)
        try:
            decode_sweep(bytes(m))
        except W.DecodeError:
            pass
        except Exception:
            crashes += 1
    check(f"400 corrupted v2 frames: {crashes} crashes", crashes == 0)
    check("wire_size counts 21 B per frame and splits at 217 B",
          W.wire_size(217) == 238 and W.wire_size(218) == 218 + 42)


# ------------------------------------------- 15. link sharing / headroom

class _FakeLink:
    """siklink.Link stand-in: real frames into a byte buffer, no serial."""

    max_payload = 220

    def __init__(self):
        import siklink
        self._sl = siklink
        self.buf = bytearray()
        self.seq = 0
        self.frames_sent = 0
        self.throttle_waits = 0

    def send_msg(self, mtype, body=b""):
        payload = bytes([mtype]) + body
        self.buf += self._sl.build_frame(self.seq, time.time(), payload)
        self.seq += 1
        self.frames_sent += 1


def test_link_sharing():
    import threading
    from aux_channels import AuxRunner, TestPatternSource
    from linkmux import LinkMux
    section("15. link sharing: spectrum budget, GNSS reserve, interleaving")

    total, reserve = 17_000 / 8, 3_000 / 8
    link = _FakeLink()
    mux = LinkMux(link, total, reserve)
    stop = threading.Event()

    # Aux offered at ~2x its reserve (10 Hz x 60 B = 810 B/s of wire).
    aux = AuxRunner(TestPatternSource(10.0, 60), mux, stop)
    aux.start()

    # Three bands of very different sizes, offered as fast as the pacer lets
    # them through. Real sweeps, so sizes are realistic.
    src = SimSource(gain_db=40, seed=91, motion=False)
    engs = [SweepEngine(src, nbins=nb, f_start=a, f_stop=b, band_id=k)
            for k, (a, b, nb) in enumerate(((2400e6, 2483.5e6, 256),
                                            (5725e6, 5875e6, 256),
                                            (915e6, 928e6, 128)))]
    blobs = [encode_sweep(e.sweep()) for e in engs]
    sent = [0, 0, 0]
    t0 = time.monotonic()
    warm = None
    while time.monotonic() - t0 < 5.0:
        if warm is None and time.monotonic() - t0 > 1.5:
            # Both buckets start full, so the first second is a burst by
            # design. Measure the steady state after it, which is what the
            # radio's buffer actually has to live with.
            warm = (time.monotonic(), mux.spectrum.spent, mux.aux.spent)
        for k, blob in enumerate(blobs):
            mux.wait_spectrum(mux.wire(len(blob)))
            mux.send_spectrum(blob, W.CH_SWEEP)
            sent[k] += 1
    el = time.monotonic() - warm[0]
    stop.set()
    aux.join(2)

    spec = mux.spectrum.spent - warm[1]
    auxb = mux.aux.spent - warm[2]
    spec_cap = (total - reserve) * el + 400          # + one sweep in flight
    aux_cap = reserve * el + 60
    check(f"spectrum stays inside total - reserve "
          f"({spec*8/el/1000:.2f} kbit/s of {(total-reserve)*8/1000:.1f})",
          spec <= spec_cap)
    check(f"aux stays inside its reserve even when offered 2x "
          f"({auxb*8/el/1000:.2f} kbit/s of {reserve*8/1000:.1f})",
          auxb <= aux_cap)
    check(f"total inside 17 kbit/s ({(spec+auxb)*8/el/1000:.2f})",
          spec + auxb <= spec_cap + aux_cap)
    check(f"excess aux is refused, not queued behind the spectrum "
          f"({aux.superseded} superseded by newer)", aux.superseded > 0)
    check(f"no band starves: {sent} sweeps each",
          min(sent) >= max(sent) - 1 and min(sent) > 3)

    # Everything that went out must come back out of the real parser and
    # reassembler intact, with aux frames interleaved between sweep halves.
    import siklink
    parser = siklink.FrameParser()
    partial: dict = {}
    got_sweeps = got_aux = bad = 0
    order = []
    for _s, _t, payload in parser.feed(bytes(link.buf)):
        if payload[0] != 1:
            continue
        chan, more, data = payload[1], payload[2], payload[3:]
        order.append(chan)
        partial.setdefault(chan, bytearray()).extend(data)
        if more:
            continue
        blob = bytes(partial.pop(chan))
        if chan == W.CH_SWEEP:
            try:
                decode_sweep(blob)
                got_sweeps += 1
            except W.DecodeError:
                bad += 1
        elif chan == 15:
            got_aux += int(blob[:2] == b"TP")
    check(f"all {got_sweeps} sweeps and {got_aux} aux messages reassemble, "
          f"{bad} bad, {parser.crc_errors} CRC errors",
          bad == 0 and parser.crc_errors == 0
          and got_sweeps == sum(sent) and got_aux == aux.sent)

    # Headroom holds even with nothing using it.
    link2 = _FakeLink()
    mux2 = LinkMux(link2, total, reserve)
    t0 = time.monotonic()
    while time.monotonic() - t0 < 1.0:                 # warm-up burst
        mux2.wait_spectrum(250)
        mux2.send_spectrum(blobs[0], W.CH_SWEEP)
    t1, s1 = time.monotonic(), mux2.spectrum.spent
    while time.monotonic() - t1 < 3.0:
        mux2.wait_spectrum(250)
        mux2.send_spectrum(blobs[0], W.CH_SWEEP)
    el = time.monotonic() - t1
    rate = (mux2.spectrum.spent - s1) / el
    check(f"with no aux running the reserve stays free "
          f"(spectrum {rate*8/1000:.2f} kbit/s, "
          f"{(total-rate)*8/1000:.2f} kbit/s left for GNSS)",
          rate <= (total - reserve) + 400 / el)


# -------------------------------------------- 16. base station, multi-band

def test_base_multiband():
    from aux_channels import TestPatternHandler, TestPatternSource
    section("16. base station: bands, names, aux, clocks, old logs")
    st = RxState(handlers={15: TestPatternHandler()})
    st.pin([2437e6, 921.4e6])
    skew = 3600.0                   # Pi clock an hour fast, no NTP in the field
    t0 = time.time() + skew
    geo = {0: (2400e6, 83.5e6, 256), 1: (5725e6, 150e6, 256),
           2: (915e6, 13e6, 128)}
    for k in range(12):
        for bid, (f0, span, nb) in geo.items():
            df = span / nb
            bins = np.full(nb, -120.0)
            bins[nb // 3] = -70.0
            st.add(Sweep(k + 1, t0 + k * 0.3, 1, f0 + df / 2, df,
                         bins.tolist(), -120.0, 40, band_id=bid), 200)
    check("sweeps are routed to 3 separate band views",
          sorted(st.bands) == [0, 1, 2])
    check("names guessed from the presets before any status arrives",
          [st.bands[i].name for i in range(3)] == ["ism24", "ism58", "ism915"])
    st.add_status({"bands": [[0, "ism24", 2400, 2483.5, 256, 40, "ok"],
                             [1, "fpv", 5725, 5875, 256, 40, "ok"],
                             [2, "ism915", 915, 928, 128, 20, "ok"]]})
    check("names from the Pi's band table win", st.bands[1].name == "fpv")
    check("per-band loss tracking sees no loss", st.seqs.lost == 0)
    check("pinned frequencies go to the band that contains them",
          [len(st.bands[i].tracker.pinned) for i in range(3)] == [1, 0, 1])
    drift = abs(st.pi_now() - (t0 + 11 * 0.3))
    check(f"display clock follows the Pi's clock, not the laptop's "
          f"(within {drift:.1f} s with the Pi {skew:.0f} s fast)", drift < 4.0)
    check("narrow 915 band gets a narrow track tolerance "
          f"({st.bands[2].tracker.tol/1e6:.2f} MHz vs "
          f"{st.bands[0].tracker.tol/1e6:.1f})",
          st.bands[2].tracker.tol < 1e6 and st.bands[0].tracker.tol == 5e6)

    src = TestPatternSource(20, 32)
    msgs = []
    while len(msgs) < 3:
        m = src.read()
        if m:
            msgs.append(m)
    for m in msgs:
        st.add_aux(15, m, time.time())
    st.add_aux(4, b"$GPGGA,fake", time.time())
    _, _, _, aux, _ = st.snapshot()
    check("aux test pattern decoded by its handler",
          any(a[0] == 15 and "loss 0.0%" in a[6] for a in aux))
    check("an aux channel with no handler still shows up (GNSS slot, ch4)",
          any(a[0] == 4 and a[2] == 1 for a in aux))

    for f_mhz, want in ((5806.0, "FPV R5"), (5785.0, "WiFi ch157"),
                        (1575.4, "GPS L1"), (2412.0, "WiFi ch1")):
        check(f"{f_mhz} MHz labelled {label_for(f_mhz*1e6)!r}",
              label_for(f_mhz * 1e6) == want)

    # Every log layout ever written must replay.
    layouts = {
        "current (with position)": (
            W.CSV_HEADER.split(","),
            ["1.5", "3", "1", "2", "-117.2", "20", "0", "-36.8485", "174.7633",
             "40.0", "915050781", "101562.5", "-110.0", "-111.0"]),
        "multi-band, no position": (
            "t,seq,node_id,band_id,noise_dbm,gain_db,clipped,f0_hz,df_hz,"
            "bins...".split(","),
            ["1.5", "3", "1", "2", "-117.2", "20", "0", "915050781",
             "101562.5", "-110.0", "-111.0"]),
        "base_rx v1": ("t,seq,node_id,noise_dbm,gain_db,clipped,f0_hz,df_hz,"
                       "bins...".split(","),
                       ["1.5", "3", "1", "-117.2", "20", "0", "915050781",
                        "101562.5", "-110.0", "-111.0"]),
        "sdr_tx v1": ("t,seq,noise_dbm,gain_db,clipped,f0_hz,df_hz,b0,b1"
                      .split(","),
                      ["1.5", "3", "-117.2", "20", "0", "915050781",
                       "101562.5", "-110.0", "-111.0"]),
        "sdr_scan v1": ("t,seq,noise_dbm,gain_db,f0_hz,df_hz,b0,b1".split(","),
                        ["1.5", "3", "-117.2", "20", "915050781", "101562.5",
                         "-110.0", "-111.0"]),
    }
    for name, (hdr, row) in layouts.items():
        s = W.parse_csv_row(hdr, row)
        check(f"replay reads {name} logs", s.seq == 3 and s.gain_db == 20
              and s.bins_dbm == [-110.0, -111.0] and s.f0_hz == 915050781)
    survey = os.path.join(HERE, "survey.csv")
    if os.path.exists(survey):
        import csv as csvmod
        with open(survey, newline="") as fh:
            rd = csvmod.reader(fh)
            hdr = next(rd)
            s = W.parse_csv_row(hdr, next(rd))
        check(f"your survey.csv replays ({s.nbins} bins, noise "
              f"{s.noise_dbm} dBm)", s.nbins == 256 and s.band_id == 0)


# ------------------------------------------------------------ 18. GNSS

def _nmea_epoch(t, lat, lon, alt=40.0, sats=9):
    from fake_gnss import epoch_sentences
    return [s.decode().strip() for s in
            epoch_sentences(t, lat, lon, alt, 5.0, 90.0, sats)]


def test_gnss():
    import threading
    from gnss_source import (MSG, GnssHandler, GnssSource, PositionTrack,
                             decode_fix, distance_m, encode_fix, fix_time)
    from max_m8q_gps import NMEAParser
    section("18. GNSS: your MAX-M8Q script on the link")

    # Your parser, fed what a MAX-M8Q sends, into the packed message and back.
    t = float(int(time.time()))
    p = NMEAParser()
    fix = None
    for line in _nmea_epoch(t, -36.8485123, 174.7633456):
        fix = p.feed(line) or fix
    check("max_m8q_gps parses a u-blox epoch (RMC, VTG, GGA, GSA, GSV, GLL)",
          fix is not None and fix.valid and abs(fix.lat + 36.8485123) < 1e-6)
    t_fix, from_gps = fix_time(fix, t + 0.15)
    check("a GGA 0.15 s after its epoch on a synced Pi -> exact GPS time",
          from_gps and t_fix == t)
    t_fix2, from_gps2 = fix_time(fix, t + 7.0)
    check("Pi clock 7 s off -> keeps the Pi's clock so fixes line up with "
          "sweeps", not from_gps2 and t_fix2 == t + 7.0)

    blob = encode_fix(fix, 42, t_fix, from_gps)
    d = decode_fix(blob)
    check(f"fix message is {len(blob)} B ({len(blob) + 21} B on air, "
          f"{(len(blob) + 21) * 8 / 1000:.2f} kbit/s at 1 Hz)",
          len(blob) == MSG.size == 31)
    err_m = distance_m((d["lat"], d["lon"]), (fix.lat, fix.lon))
    check(f"position survives to {err_m*100:.1f} cm, alt {d['alt_m']} m, "
          f"sats {d['sats_used']}, seq {d['seq']}",
          err_m < 0.02 and d["alt_m"] == 40.0 and d["sats_used"] == 9
          and d["seq"] == 42 and d["gps_time"])

    p2 = NMEAParser()
    nofix = None
    for line in _nmea_epoch(t, None, None, sats=2):
        nofix = p2.feed(line) or nofix
    dn = decode_fix(encode_fix(nofix, 1, t))
    check("a 'no fix yet' epoch is still sent, with its satellite count",
          not dn["valid"] and dn["lat"] is None and dn["sats_used"] == 2)

    # Where was I between fixes?
    tr = PositionTrack()
    tr.add(100.0, -36.0, 174.0, 10.0)
    tr.add(101.0, -36.001, 174.002, 20.0)
    mid = tr.at(100.25)
    check("between two fixes the position is interpolated",
          abs(mid[0] + 36.00025) < 1e-9 and abs(mid[1] - 174.0005) < 1e-9
          and abs(mid[2] - 12.5) < 1e-9)
    check("just after the last fix it holds that fix",
          tr.at(101.8) == (-36.001, 174.002, 20.0))
    check("long after the last fix there is no position (not a stale one)",
          tr.at(110.0) is None)
    tr.add(106.0, -36.01, 174.01, 5.0)
    check("across a 5 s gap in fixes it does not interpolate",
          tr.at(103.5) is None)

    # The whole Pi-side source against a fake receiver on a pseudo-terminal:
    # your MaxM8Q class, real serial I/O, the reconnect logic.
    try:
        import pty
        import tty
    except ImportError:
        check("pseudo-terminal test skipped (not Linux)", True)
        return
    from fake_gnss import epoch_sentences
    master, slave = pty.openpty()
    tty.setraw(slave)
    path = os.ttyname(slave)
    stop = threading.Event()

    def feeder():
        tt = float(int(time.time())) + 1
        k = 0
        while not stop.is_set():
            time.sleep(max(0, tt - time.time()) + 0.1)
            lat = -36.8485 + k * 1e-5 if k >= 1 else None
            lon = 174.7633 if k >= 1 else None
            for s in epoch_sentences(tt, lat, lon, 40.0, 1.0, 0.0,
                                     9 if lat else 2):
                try:
                    os.write(master, s)
                except OSError:
                    return
            tt += 1.0
            k += 1
    th = threading.Thread(target=feeder, daemon=True)
    th.start()
    src = GnssSource(port=path)
    handler = GnssHandler()
    got = []
    try:
        src.open()
        t_end = time.time() + 6
        while time.time() < t_end and sum(1 for g in got if g["valid"]) < 3:
            m = src.read()
            if m:
                got.append(decode_fix(m))
                handler.handle(m, time.time())
    finally:
        stop.set()
        src.close()
    valid = [g for g in got if g["valid"]]
    check(f"GnssSource reads your HAT script over a real serial port "
          f"({len(got)} epochs, {len(valid)} with a fix)",
          len(valid) >= 3 and not got[0]["valid"])
    check(f"handler shows it: {handler.handle(encode_fix(src.last, 99, time.time()), time.time())[:48]}...",
          handler.last is not None and handler.last["valid"])
    tv = valid[-1]["t"]
    check("both ends can say where the Pi was at a sweep's timestamp",
          src.position_at(tv) is not None
          and handler.position_at(tv) is not None)
    os.close(master)


# ------------------------------------------------------------------ 17. live

def test_live(timeout=45):
    section("17. live: 3 bands + GNSS + aux test -> virtual radio (5% loss) "
            "-> base_rx")
    tmp = tempfile.mkdtemp(prefix="sdrlink-")
    a, b = os.path.join(tmp, "radioA"), os.path.join(tmp, "radioB")
    gps = os.path.join(tmp, "gps")
    png = os.path.join(tmp, "base.png")
    rx_csv = os.path.join(tmp, "rx.csv")
    tx_csv = os.path.join(tmp, "tx.csv")
    env = dict(os.environ, PYTHONUNBUFFERED="1")

    fg = subprocess.Popen(
        [sys.executable, "fake_gnss.py", "--link", gps, "--quiet",
         "--no-fix-secs", "2"],
        cwd=HERE, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    vl = subprocess.Popen(
        [sys.executable, "vlink.py", "--link-a", a, "--link-b", b,
         "--loss", "5", "--corrupt", "0.05", "--latency", "0.05", "--quiet"],
        cwd=HERE, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True)

    for _ in range(100):
        if os.path.exists(a) and os.path.exists(b):
            break
        time.sleep(0.1)
    if not (os.path.exists(a) and os.path.exists(b)):
        vl.kill()
        return check("virtual radio pair came up", False)
    check("virtual radio pair came up", True)

    rx = subprocess.Popen(
        [sys.executable, "base_rx.py", "--port", b, "--save-png", png,
         "--save-after", "22", "--verbose", "--csv", rx_csv],
        cwd=HERE, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True)
    time.sleep(2.0)
    tx = subprocess.Popen(
        [sys.executable, "sdr_tx.py", "--source", "sim", "--port", a,
         "--kbps", "17", "--reserve-kbps", "3", "--aux-test", "5:32",
         "--gnss-port", gps, "--csv", tx_csv, "--sim-drop-after", "8"],
        cwd=HERE, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True)

    try:
        rx_out, _ = rx.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        rx.kill()
        rx_out, _ = rx.communicate()

    import signal
    tx.send_signal(signal.SIGINT)        # Ctrl-C, so it prints its summary
    try:
        tx_out, _ = tx.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        tx.kill()
        tx_out = ""
    fg.terminate()
    vl.terminate()
    try:
        vl_out, _ = vl.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        vl.kill()
        vl_out = ""

    print("\n  --- transmitter ---")
    for line in (tx_out or "").strip().splitlines()[-9:]:
        print("   ", line)
    print("  --- receiver ---")
    for line in (rx_out or "").strip().splitlines()[-9:]:
        print("   ", line)
    print("  --- virtual radio ---")
    for line in (vl_out or "").strip().splitlines()[-3:]:
        print("   ", line)
    print()

    got = 0
    per_band = {}
    aux_msgs = 0
    import re
    for line in (rx_out or "").splitlines():
        if line.strip().startswith("sweeps "):
            try:
                got = int(line.split()[1])
            except (ValueError, IndexError):
                pass
            m = re.search(r"\[(.*?)\]", line)
            if m:
                for part in m.group(1).split(","):
                    name, n = part.split()
                    per_band[name] = int(n)
        if line.startswith("aux ch15"):
            aux_msgs = int(line.split()[3])

    check(f"base station decoded sweeps over a lossy link ({got} received)",
          got > 10)
    check(f"all three bands arrived {per_band}",
          len(per_band) == 3 and min(per_band.values()) > 5)
    check(f"aux test stream arrived alongside ({aux_msgs} msgs)",
          aux_msgs > 20)
    check("a simulated SDR dropout 8 s in was survived (reconnected, kept "
          "sending)", "SDR back after" in (tx_out or "")
          and "dropped out 1 time" in (tx_out or ""))
    gnss_line = next((ln for ln in (rx_out or "").splitlines()
                      if ln.startswith("aux ch4 gnss")), "")
    check(f"GNSS fixes arrived on channel 4: {gnss_line[:70]}",
          "GPS" in gnss_line and "msgs" in gnss_line)

    def tagged(path):
        import csv as csvmod
        n = with_pos = 0
        try:
            with open(path, newline="") as fh:
                rd = csvmod.reader(fh)
                hdr = next(rd)
                for row in rd:
                    sw = W.parse_csv_row(hdr, row)
                    n += 1
                    with_pos += sw.lat is not None
        except (OSError, StopIteration):
            pass
        return n, with_pos
    n_rx, p_rx = tagged(rx_csv)
    n_tx, p_tx = tagged(tx_csv)
    check(f"base log: {p_rx} of {n_rx} sweeps carry a position",
          n_rx > 10 and p_rx > 0.6 * n_rx)
    check(f"Pi log: {p_tx} of {n_tx} sweeps carry a position",
          n_tx > 10 and p_tx > 0.6 * n_tx)
    check("PNG dashboard rendered", os.path.exists(png)
          and os.path.getsize(png) > 40_000,
          f"{os.path.getsize(png) if os.path.exists(png) else 0} bytes")

    kbps = None
    for line in (tx_out or "").splitlines():
        if "mean rate" in line:
            try:
                kbps = float(line.split()[2])
            except (ValueError, IndexError):
                pass
    if kbps is not None:
        check(f"transmitter stayed inside 17 kbit/s ({kbps:.2f} measured, "
              f"aux included)", kbps <= 17.0)

    if os.path.exists(png):
        dest = os.path.join(HERE, "base_station_demo.png")
        try:
            import shutil
            shutil.copy(png, dest)
            print(f"  dashboard saved to {dest}")
        except OSError:
            pass
    return True


# ------------------------------------------------------------------------ CLI

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true",
                    help="also run the tx/rx processes against a virtual radio")
    ap.add_argument("--only", default=None, help="substring of a test name")
    args = ap.parse_args()

    t0 = time.time()
    tests = [
        ("power", test_power_calibration),
        ("stitch", test_stitching),
        ("dc", test_dc_repair),
        ("codec", test_codec),
        ("bandwidth", test_bandwidth),
        ("detect", test_detection_tracking),
        ("loss", test_loss_accounting),
        ("bands", test_bands),
        ("multiband", test_multiband),
        ("pluto", test_pluto_path),
        ("wire2", test_wire_v2),
        ("linkshare", test_link_sharing),
        ("base", test_base_multiband),
        ("gnss", test_gnss),
    ]
    for name, fn in tests:
        if args.only and args.only not in name:
            continue
        fn()
    if args.live and not args.only:
        test_live()

    print("\n" + "=" * 66)
    print(f"{len(PASS)} passed, {len(FAIL)} failed  in {time.time()-t0:.1f}s")
    if FAIL:
        print("\nfailures:")
        for f in FAIL:
            print(f"  - {f}")
    print("=" * 66)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
