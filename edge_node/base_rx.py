#!/usr/bin/env python3
"""base_rx - receive spectrum sweeps on the laptop and show them live.

    python base_rx.py                          # autodetect the radio, open the GUI
    python base_rx.py --port COM5              # Windows, explicit port
    python base_rx.py --port /dev/pts/6        # against vlink.py
    python base_rx.py --no-gui                 # text dashboard, good over ssh
    python base_rx.py --csv run.csv            # log everything as it arrives
    python base_rx.py --replay run.csv         # play a log back through the GUI

LAYOUT - one column per band, however many bands the Pi is scanning
  spectrum   current sweep, max-hold, noise floor, known channels marked
  waterfall  last few hundred sweeps of that band, newest at top
  and across the bottom, for emitters in ALL bands:
  RSSI/time  the emitters being tracked, one line each
  range      relative range and warmer/colder for each

Band names come from the Pi's status messages, so this needs no config of its
own. Anything on channels 4-15 (GNSS, once it exists) is shown on the top line
and handed to whatever handler aux_channels.py registers for it.

READING RSSI WITHOUT GPS.  You cannot get an absolute distance from RSSI - that
needs a calibrated reference at a known range, and it changes every time you
turn a corner. What you CAN get is relative: how far away is this emitter now
compared to the closest I have been to it so far. That is the number in the
bearing panel, and it is honest about what it is.

The other half of the problem is fading. Move a receiver half a wavelength at
2.4 GHz - six centimetres - and multipath can swing RSSI by 20 dB with the
transmitter bolted to a wall. A mean would chase every one of those swings.
This uses a running MEDIAN, which is the standard treatment for Rayleigh
fading, and then fits a trend over several seconds. So the warmer/colder arrow
responds in about two seconds rather than instantly, and that is a deliberate
trade: an instant arrow points at multipath, not at the transmitter.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import threading
import time
from collections import deque

import numpy as np

from aux_channels import GenericHandler, build_handlers
from bands import BAND_PRESETS, KNOWN_SIGNALS, OVERLAYS
from rssi_wire import (CH_AUX_FIRST, CH_AUX_LAST, CH_LOG, CH_STATUS, CH_SWEEP,
                       CSV_HEADER, DecodeError, DelayedSweepLog, SeqTracker,
                       Sweep, decode_status, decode_sweep, parse_csv_row)

DEF_PATHLOSS_N = 2.5        # log-distance exponent: 2.0 free space, 3-4 indoors
DEF_TREND_SECS = 4.0
DEF_MEDIAN_N = 5


# ----------------------------------------------------------------- detection

def _close_gaps(mask: np.ndarray, gap: int) -> np.ndarray:
    """Bridge runs of False shorter than `gap`, so one signal stays one signal."""
    if gap <= 0:
        return mask
    out = mask.copy()
    idx = np.flatnonzero(mask)
    if idx.size < 2:
        return out
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= gap + 1:
            out[a:b] = True
    return out


def find_emitters(sweep: Sweep, snr_db: float = 8.0, max_n: int = 8,
                  merge_gap_bins: int = 3):
    """Contiguous runs of bins above the noise floor, as (freq, dBm, bw, bins).

    Runs rather than single peaks: a WiFi carrier is 20 MHz wide and occupies
    60 bins, and reporting 60 separate emitters would be useless. The reported
    power is the SUM of the run in linear terms, so a wide signal and a narrow
    one at the same total power compare correctly - which is what you need
    when the thing you are hunting might be either.

    `merge_gap_bins` bridges short dropouts before runs are cut. Without it a
    faded 20 MHz carrier breaks into three or four fragments, each becomes its
    own track, and the RSSI history you are trying to navigate by keeps
    getting split between tracks that appear and vanish.
    """
    b = np.asarray(sweep.bins_dbm, dtype=float)
    thresh = sweep.noise_dbm + snr_db
    above = _close_gaps(b > thresh, merge_gap_bins)
    if not above.any():
        return []

    out = []
    i = 0
    n = len(b)
    while i < n:
        if not above[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and above[j + 1]:
            j += 1

        idx = np.arange(i, j + 1)
        lin = 10.0 ** (b[idx] / 10.0)
        total = float(lin.sum())
        centroid = float((idx * lin).sum() / lin.sum())
        out.append({
            "f_hz": sweep.f0_hz + centroid * sweep.df_hz,
            "dbm": 10.0 * math.log10(total),
            "bw_hz": (j - i + 1) * sweep.df_hz,
            "peak_dbm": float(b[idx].max()),
            "lo": i, "hi": j,
        })
        i = j + 1

    out.sort(key=lambda d: d["dbm"], reverse=True)
    return out[:max_n]


def label_for(f_hz: float) -> str:
    """Name the frequency if a well-known service lives there.

    Nearest match, not first match. WiFi channels are 5 MHz apart and overlap
    heavily, so any tolerance wide enough to be useful catches two or three of
    them - iterating and taking the first gives you whichever happens to come
    first in the dict, which is how 2460.8 MHz ends up labelled ch10 when ch11
    is four times closer. The table is bands.KNOWN_SIGNALS. These are guesses
    from frequency alone: an analog FPV transmitter on R5 and WiFi ch161 sit
    1 MHz apart and only the spectrum shape tells them apart.
    """
    best, best_df = "", float("inf")
    for name, f, tol in KNOWN_SIGNALS:
        df = abs(f_hz - f)
        if df < tol and df < best_df:
            best, best_df = name, df
    return best


# ------------------------------------------------------------------ tracking

class EmitterTrack:
    """One emitter followed over time, with the localisation state."""

    _next_id = 1

    def __init__(self, f_hz, dbm, t, history=600, band_id=0, noise_dbm=None):
        self.id = EmitterTrack._next_id
        EmitterTrack._next_id += 1
        self.band_id = band_id
        self.noise_dbm = noise_dbm      # of its band, for ranking across bands
        self.f_hz = f_hz
        self.label = label_for(f_hz)
        self.t_hist = deque(maxlen=history)
        self.raw_hist = deque(maxlen=history)
        self.med_hist = deque(maxlen=history)
        self.best_dbm = -999.0
        self.last_seen = t
        self.misses = 0
        self.update(f_hz, dbm, t)

    def update(self, f_hz, dbm, t, median_n=DEF_MEDIAN_N):
        # Ease the frequency rather than jumping: the centroid of a fading
        # wideband signal wanders a bin or two and we do not want the track
        # label wandering with it.
        self.f_hz = 0.9 * self.f_hz + 0.1 * f_hz
        self.last_seen = t
        self.misses = 0

        self.t_hist.append(t)
        self.raw_hist.append(dbm)

        window = list(self.raw_hist)[-median_n:]
        med = statistics.median(window)
        self.med_hist.append(med)
        # Only trust the reference once the median has a full window behind
        # it, or a single fading spike on the first sample sets an
        # unreachable "closest approach" and every later reading looks distant.
        if len(self.raw_hist) >= median_n:
            self.best_dbm = max(self.best_dbm, med)

    @property
    def rssi(self) -> float:
        return self.med_hist[-1] if self.med_hist else float("nan")

    def relative_range(self, n=DEF_PATHLOSS_N) -> float:
        """How far away compared to the closest this emitter has ever been.

        1.0 means you are at your best-ever position for it. 2.0 means twice
        that distance. Unitless on purpose - turning it into metres needs a
        calibration you do not have.
        """
        if not self.med_hist or self.best_dbm < -998:
            return float("nan")
        return 10.0 ** ((self.best_dbm - self.rssi) / (10.0 * n))

    def trend_db_per_s(self, secs=DEF_TREND_SECS) -> float:
        """Positive means the signal is growing, i.e. you are closing on it."""
        if len(self.t_hist) < 3:
            return 0.0
        t_now = self.t_hist[-1]
        ts, vs = [], []
        for t, v in zip(self.t_hist, self.med_hist):
            if t_now - t <= secs:
                ts.append(t)
                vs.append(v)
        if len(ts) < 3 or (ts[-1] - ts[0]) < 0.5:
            return 0.0
        return float(np.polyfit(np.array(ts) - ts[0], vs, 1)[0])

    def verdict(self, secs=DEF_TREND_SECS) -> str:
        s = self.trend_db_per_s(secs)
        if s > 1.5:
            return "WARMER  ^^"
        if s > 0.4:
            return "warmer   ^"
        if s < -1.5:
            return "COLDER  vv"
        if s < -0.4:
            return "colder   v"
        return "steady   -"


class Tracker:
    """Associates detections with tracks, frame to frame."""

    def __init__(self, tol_hz=3e6, max_tracks=8, drop_after=6, band_id=0):
        self.band_id = band_id
        self.tol = tol_hz
        self.max_tracks = max_tracks
        self.drop_after = drop_after
        self.tracks: list[EmitterTrack] = []
        self.pinned: list[float] = []

    def pin(self, freqs):
        """Follow these exact frequencies whether or not they look like peaks."""
        self.pinned = list(freqs)

    def update(self, sweep: Sweep, dets):
        t = sweep.t

        if self.pinned:
            b = np.asarray(sweep.bins_dbm)
            dets = []
            for f in self.pinned:
                i = sweep.bin_of(f)
                lo, hi = max(0, i - 30), min(len(b) - 1, i + 30)
                lin = 10.0 ** (b[lo:hi + 1] / 10.0)
                dets.append({"f_hz": f, "dbm": 10 * math.log10(lin.sum()),
                             "bw_hz": (hi - lo + 1) * sweep.df_hz,
                             "peak_dbm": float(b[lo:hi + 1].max()),
                             "lo": lo, "hi": hi})

        used = set()
        for d in dets:
            best, best_df = None, self.tol
            for tr in self.tracks:
                if id(tr) in used:
                    continue
                df = abs(tr.f_hz - d["f_hz"])
                if df < best_df:
                    best, best_df = tr, df
            if best is not None:
                best.update(d["f_hz"], d["dbm"], t)
                best.noise_dbm = sweep.noise_dbm
                used.add(id(best))
            elif (len(self.tracks) < self.max_tracks) or self.pinned:
                self.tracks.append(EmitterTrack(d["f_hz"], d["dbm"], t,
                                                band_id=self.band_id,
                                                noise_dbm=sweep.noise_dbm))

        for tr in self.tracks:
            if id(tr) not in used and tr.last_seen < t:
                tr.misses += 1
        if not self.pinned:
            self.tracks = [tr for tr in self.tracks if tr.misses < self.drop_after]
        self.tracks.sort(key=lambda tr: tr.rssi, reverse=True)
        return dets


# --------------------------------------------------------------- shared state

def guess_band_name(f_lo: float, f_hi: float, band_id: int) -> str:
    """Until the Pi's band table arrives (or when replaying a log), name a
    band by matching its edges against the presets."""
    for name, b in BAND_PRESETS.items():
        if abs(b.f_start_hz - f_lo) < 2e6 and abs(b.f_stop_hz - f_hi) < 2e6:
            return name
    return f"band{band_id}"


class MultiSeq:
    """A SeqTracker per band (each band counts its own sweeps), plus totals."""

    def __init__(self):
        self.per: dict[int, SeqTracker] = {}

    def observe(self, band_id: int, seq: int) -> int:
        return self.per.setdefault(band_id, SeqTracker()).observe(seq)

    @property
    def lost(self) -> int:
        return sum(t.lost for t in self.per.values())

    @property
    def received(self) -> int:
        return sum(t.received for t in self.per.values())

    @property
    def loss_pct(self) -> float:
        miss = sum(sum(t._recent) for t in self.per.values())
        got = sum(len(t._recent) for t in self.per.values())
        return 100.0 * miss / max(1, got + miss) if got else 0.0

    def band_loss_pct(self, band_id: int) -> float:
        t = self.per.get(band_id)
        return t.loss_pct if t else 0.0


class BandView:
    """Everything the display needs for one band."""

    def __init__(self, band_id, name, waterfall_rows, max_tracks, tol_hz,
                 pinned=()):
        self.band_id = band_id
        self.name = name
        self.sweep: Sweep | None = None
        self.maxhold: np.ndarray | None = None
        self.waterfall = deque(maxlen=waterfall_rows)
        self.geometry: tuple | None = None
        self.tracker = Tracker(tol_hz=tol_hz, max_tracks=max_tracks,
                               band_id=band_id)
        self.pinned_all = list(pinned)
        self.dets: list = []
        self.n_sweeps = 0
        self.n_clipped = 0
        self.t_rx: deque = deque(maxlen=30)     # arrival times, for rate
        self.gain_note = ""

    def rate(self) -> float:
        if len(self.t_rx) < 2:
            return 0.0
        return (len(self.t_rx) - 1) / max(1e-6, self.t_rx[-1] - self.t_rx[0])


class AuxView:
    def __init__(self, chan, handler):
        self.chan = chan
        self.handler = handler
        self.msgs = 0
        self.bytes = 0
        self.t_first = None
        self.t_last = None
        self.text = ""


class RxState:
    """What the reader thread produces and the display consumes."""

    def __init__(self, waterfall_rows=300, snr_db=8.0, max_emitters=8,
                 pathloss_n=DEF_PATHLOSS_N, median_n=DEF_MEDIAN_N,
                 merge_gap_bins=3, track_tol_hz=5e6, handlers=None):
        self.lock = threading.Lock()
        self.bands: dict[int, BandView] = {}
        self.band_table: dict[int, list] = {}   # from the Pi's status
        self.waterfall_rows = waterfall_rows
        self.max_emitters = max_emitters
        self.track_tol_hz = track_tol_hz
        self.pinned: list[float] = []
        self.seqs = MultiSeq()
        self.status: dict = {}
        self.snr_db = snr_db
        self.pathloss_n = pathloss_n
        self.median_n = median_n
        self.merge_gap_bins = merge_gap_bins
        self.handlers = handlers if handlers is not None else {}
        self.aux: dict[int, AuxView] = {}

        self.n_sweeps = 0
        self.n_bad = 0
        self.n_clipped = 0
        self.bytes_rx = 0
        self.t_first: float | None = None
        self.t_last = 0.0
        self.notices: deque = deque(maxlen=8)
        self._offsets: deque = deque(maxlen=25)

    # -- time ---------------------------------------------------------------

    def pi_now(self) -> float:
        """'Now' on the Pi's clock.

        Sweep timestamps come from the Pi, and with no network in the field
        its clock can be minutes or days off the laptop's. Plotting Pi times
        against the laptop's time.time() (as this used to) pushes every line
        off the edge of the RSSI plot - and did the same on --replay, where
        the log is days old. The median arrival offset fixes both.
        """
        if not self._offsets:
            return time.time()
        return time.time() - statistics.median(self._offsets)

    # -- feeding ------------------------------------------------------------

    def pin(self, freqs):
        self.pinned = list(freqs)

    def _band(self, sw: Sweep) -> BandView:
        bv = self.bands.get(sw.band_id)
        if bv is None:
            name = self._name_for(sw.band_id) or guess_band_name(
                sw.f_lo_hz, sw.f_hi_hz, sw.band_id)
            span = sw.nbins * sw.df_hz
            # Tracks may drift a few MHz in a 83 MHz band, but in 13 MHz of
            # 915 that tolerance would merge every LoRa channel into one.
            tol = min(self.track_tol_hz, max(4 * sw.df_hz, span / 16.0))
            bv = BandView(sw.band_id, name, self.waterfall_rows,
                          self.max_emitters, tol)
            pins = [f for f in self.pinned if sw.f_lo_hz <= f <= sw.f_hi_hz]
            if pins:
                bv.tracker.pin(pins)
            self.bands[sw.band_id] = bv
        return bv

    def _name_for(self, band_id) -> str:
        row = self.band_table.get(band_id)
        return row[1] if row and len(row) > 1 else ""

    def add(self, sw: Sweep, nbytes: int) -> None:
        with self.lock:
            now = time.time()
            self._offsets.append(now - sw.t)
            bv = self._band(sw)
            geom = (sw.nbins, round(sw.f0_hz), round(sw.df_hz))
            if geom != bv.geometry:
                # The scanner reconfigured mid-run. Old rows do not line up
                # with the new grid, so throw them away rather than draw a
                # waterfall that silently stretches history across the band.
                if bv.geometry is not None:
                    self.notices.append(
                        f"{bv.name}: grid changed to {sw.nbins} bins @ "
                        f"{sw.df_hz/1e3:.0f} kHz")
                bv.geometry = geom
                bv.waterfall.clear()
                bv.maxhold = None

            bv.sweep = sw
            b = np.asarray(sw.bins_dbm, dtype=float)
            bv.maxhold = b.copy() if bv.maxhold is None else \
                np.maximum(bv.maxhold, b)
            bv.waterfall.append(b)
            bv.n_sweeps += 1
            bv.n_clipped += int(sw.clipped)
            bv.t_rx.append(now)

            self.n_sweeps += 1
            self.bytes_rx += nbytes
            self.n_clipped += int(sw.clipped)
            self.seqs.observe(sw.band_id, sw.seq)
            if self.t_first is None:
                self.t_first = now
            self.t_last = now

            # A clipped sweep reads low across the board. Feeding it to the
            # tracker would poison the median and the best-ever reference,
            # so it is displayed but not tracked.
            dets = find_emitters(sw, self.snr_db, bv.tracker.max_tracks,
                                 self.merge_gap_bins)
            bv.dets = dets
            if not sw.clipped:
                bv.tracker.update(sw, dets)

    def add_status(self, d: dict) -> None:
        with self.lock:
            self.status = d
            for row in d.get("bands", []) or []:
                try:
                    bid = int(row[0])
                except (TypeError, ValueError, IndexError):
                    continue
                self.band_table[bid] = row
                if bid in self.bands and len(row) > 1:
                    self.bands[bid].name = str(row[1])
                if len(row) > 6 and row[6] == "skip":
                    note = f"Pi skipped band {row[1]} (radio cannot tune it)"
                    if note not in self.notices:
                        self.notices.append(note)

    def add_log(self, blob: bytes) -> None:
        with self.lock:
            try:
                d = json.loads(blob.decode("utf-8"))
                ev = d.get("ev") if isinstance(d, dict) else None
                if ev == "gain":
                    msg = f"{d.get('band', '?')}: gain -> {d.get('db')} dB"
                elif ev == "sdr_lost":
                    msg = f"SDR DROPPED OUT on the Pi ({d.get('err', '')})"
                elif ev == "sdr_back":
                    msg = (f"SDR back after {d.get('s')} s "
                           f"(drop #{d.get('n')})")
                else:
                    msg = str(d)
            except (UnicodeDecodeError, json.JSONDecodeError):
                msg = blob.decode("utf-8", "replace")
            self.notices.append(msg[:70])

    def add_aux(self, chan: int, blob: bytes, t_rx: float) -> None:
        with self.lock:
            av = self.aux.get(chan)
            if av is None:
                h = self.handlers.get(chan) or GenericHandler(chan)
                av = self.aux[chan] = AuxView(chan, h)
            av.msgs += 1
            av.bytes += len(blob)
            av.t_first = av.t_first or t_rx
            av.t_last = t_rx
            try:
                txt = av.handler.handle(blob, t_rx)
            except Exception as e:              # a handler bug must not kill rx
                txt = f"handler error: {e}"
            if txt:
                av.text = txt

    def position_at(self, t: float):
        """Where the Pi was at time t (its clock), from any aux handler that
        tracks position - the GNSS one. None if nothing knows."""
        for h in self.handlers.values():
            fn = getattr(h, "position_at", None)
            if fn:
                p = fn(t)
                if p:
                    return p
        return None

    # -- reading ------------------------------------------------------------

    def all_tracks(self):
        """Tracks from every band, strongest above ITS OWN floor first.

        Ranking on raw dBm would let whichever band has the highest noise
        floor hog the list; SNR compares like with like."""
        out = []
        for bv in self.bands.values():
            out.extend(bv.tracker.tracks)

        def snr(tr):
            return tr.rssi - (tr.noise_dbm if tr.noise_dbm is not None else -120)
        out.sort(key=snr, reverse=True)
        return out

    def snapshot(self):
        with self.lock:
            bands = []
            for bid in sorted(self.bands):
                bv = self.bands[bid]
                bands.append({
                    "id": bid, "name": bv.name, "sweep": bv.sweep,
                    "maxhold": None if bv.maxhold is None else bv.maxhold.copy(),
                    "wf": np.array(bv.waterfall) if bv.waterfall else None,
                    "dets": list(bv.dets), "rate": bv.rate(),
                    "loss": self.seqs.band_loss_pct(bid),
                    "n": bv.n_sweeps, "clipped": bv.n_clipped,
                })
            aux = [(ch, av.handler.name, av.msgs, av.bytes, av.t_first,
                    av.t_last, av.text) for ch, av in sorted(self.aux.items())]
            return bands, self.all_tracks(), dict(self.status), aux, \
                list(self.notices)


# -------------------------------------------------------------- reader thread

def reader(link, state: RxState, stop: threading.Event, csv=None,
           verbose=False, aux_csv=None):
    """csv, if given, is a DelayedSweepLog: rows go out ~1.5 s late so each
    sweep can be tagged with a position interpolated between GNSS fixes."""
    while not stop.is_set():
        try:
            if csv:
                csv.flush()
            for mtype, body in link.poll_raw(0.1):
                if mtype != 1:                    # siklink DATA
                    continue
                got = link.reassemble(body)
                if got is None:
                    continue
                chan, blob = got
                t_rx = time.time()

                if chan == CH_SWEEP:
                    try:
                        sw = decode_sweep(blob)
                    except DecodeError as e:
                        state.n_bad += 1
                        if verbose:
                            print(f"bad sweep: {e}", file=sys.stderr)
                        continue
                    state.add(sw, len(blob))
                    if csv:
                        csv.add(sw)

                elif chan == CH_STATUS:
                    try:
                        state.add_status(decode_status(blob))
                    except DecodeError:
                        pass

                elif chan == CH_LOG:
                    state.add_log(blob)

                elif CH_AUX_FIRST <= chan <= CH_AUX_LAST:
                    state.add_aux(chan, blob, t_rx)
                    if aux_csv:
                        aux_csv.write(f"{t_rx:.3f},{chan},{len(blob)},"
                                      f"{blob.hex()}\n")
        except Exception as e:                    # keep the display alive
            if verbose:
                print(f"reader: {e}", file=sys.stderr)
            time.sleep(0.2)


# ------------------------------------------------------------ shared text bits

def sdr_health(status: dict) -> str:
    """'SDR 52/68 C  drops 2' from the Pi's status, or '' if it did not say.
    Temperatures are AD936x / Zynq; drops are SDR reconnects on the Pi."""
    out = []
    t = status.get("temp") if isinstance(status, dict) else None
    if isinstance(t, (list, tuple)):
        vals = [f"{v:.0f}" for v in t if isinstance(v, (int, float))]
        if vals:
            out.append(f"SDR {'/'.join(vals)} C")
    d = status.get("sdr_drops") if isinstance(status, dict) else None
    if isinstance(d, int) and d:
        out.append(f"drops {d}")
    return "  ".join(out)


def link_line(state: RxState, link) -> str:
    el = max(1e-9, time.time() - (state.t_first or time.time()))
    if not state.t_last:
        return "waiting for the first sweep..."
    health = sdr_health(state.status)
    return (f"{state.n_sweeps} sweeps  {state.bytes_rx*8/el/1000:4.1f} kbit/s "
            f"spectrum  loss {state.seqs.loss_pct:4.1f}%  "
            f"crc {link.parser.crc_errors}  bad {state.n_bad}  "
            f"age {time.time() - state.t_last:4.1f}s"
            + (f"  |  {health}" if health else ""))


def aux_lines(aux) -> list[str]:
    out = []
    now = time.time()
    for ch, name, msgs, nbytes, t_first, t_last, text in aux:
        el = max(1e-9, (t_last or now) - (t_first or now))
        rate = (msgs - 1) / el if msgs > 1 else 0.0
        out.append(f"ch{ch} {name}: {rate:4.1f} msg/s  "
                   f"{nbytes*8/max(el,1e-9)/1000 if msgs > 1 else 0:4.2f} kbit/s  "
                   f"last {now - (t_last or now):4.1f}s ago  {text}")
    return out


def band_title(b) -> str:
    sw = b["sweep"]
    if sw is None:
        return b["name"]
    clip = "  ** ADC CLIPPING **" if sw.clipped else ""
    return (f"{b['name']}  {sw.f_lo_hz/1e6:.0f}-{sw.f_hi_hz/1e6:.0f} MHz  "
            f"{b['rate']:.1f}/s  loss {b['loss']:.0f}%  gain {sw.gain_db}  "
            f"floor {sw.noise_dbm:.0f}{clip}")


# ------------------------------------------------------------------ text mode

def run_text(state: RxState, stop: threading.Event, link, interval=1.0):
    print("waiting for sweeps... (Ctrl-C to stop)")
    try:
        while not stop.is_set():
            time.sleep(interval)
            bands, tracks, status, aux, notices = state.snapshot()
            if not bands:
                continue
            print("\n" + "=" * 78)
            print(link_line(state, link))
            for line in aux_lines(aux):
                print("  " + line)
            for b in bands:
                sw = b["sweep"]
                print(f"  [{b['id']}] {band_title(b)}  "
                      f"({sw.nbins} bins x {sw.df_hz/1e3:.0f} kHz)")
            if not tracks:
                print(f"  no emitters above noise + {state.snr_db:.0f} dB")
            names = {b["id"]: b["name"] for b in bands}
            for tr in tracks[:8]:
                rel = tr.relative_range(state.pathloss_n)
                lbl = f" ({tr.label})" if tr.label else ""
                # Bar shows SNR above the band's floor, not absolute dBm - the
                # floor moves with gain and band, so a fixed scale means nothing.
                floor = tr.noise_dbm if tr.noise_dbm is not None else -120
                bar = "#" * max(0, min(24, int((tr.rssi - floor) / 2)))
                # A track needs a full median window before its best-ever
                # reference means anything; until then there is no range yet.
                rng = "   --  " if math.isnan(rel) else f"{rel:5.2f}x "
                print(f"  {names.get(tr.band_id, '?'):<8}"
                      f"{tr.f_hz/1e6:9.2f} MHz{lbl:<13} "
                      f"{tr.rssi:7.1f} dBm {bar:<24} "
                      f"rel.range {rng} "
                      f"{tr.trend_db_per_s():+5.1f} dB/s  {tr.verdict()}")
            for n in notices[-3:]:
                print(f"  note: {n}")
    except KeyboardInterrupt:
        stop.set()


# ------------------------------------------------------------------- GUI mode

def run_gui(state: RxState, stop: threading.Event, link, args):
    import matplotlib
    if args.save_png:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(16, 9.5))
    try:
        fig.canvas.manager.set_window_title("RSSI survey - base station")
    except Exception:
        pass
    ui = {"ids": None}

    def build(ids):
        """(Re)create the axes: one spectrum + waterfall column per band."""
        fig.clf()
        n = max(1, len(ids))
        gs = GridSpec(3, n, figure=fig, height_ratios=[1.0, 1.25, 1.1],
                      hspace=0.45, wspace=0.18, top=0.885, bottom=0.06,
                      left=0.05, right=0.98)
        cols = {}
        for i, bid in enumerate(ids):
            ax_s = fig.add_subplot(gs[0, i])
            ax_w = fig.add_subplot(gs[1, i])
            ax_s.grid(alpha=0.25)
            ax_s.set_ylabel("dBm" if i == 0 else "")
            (l_spec,) = ax_s.plot([], [], lw=1.1, color="#1f77b4", label="now")
            (l_hold,) = ax_s.plot([], [], lw=0.7, color="#999999", alpha=0.8,
                                  label="max hold")
            l_noise = ax_s.axhline(-110, ls="--", lw=0.8, color="#d62728",
                                   label="noise floor")
            marks = ax_s.scatter([], [], marker="v", s=30, color="#d62728",
                                 zorder=5)
            if i == 0:
                ax_s.legend(loc="upper right", fontsize=7, ncol=3)
            im = ax_w.imshow(np.zeros((2, 2)), aspect="auto", origin="upper",
                             cmap="viridis", interpolation="nearest")
            cb = fig.colorbar(im, ax=ax_w, pad=0.01, fraction=0.05)
            cb.ax.tick_params(labelsize=7)
            ax_w.set_xlabel("MHz", fontsize=8)
            ax_w.set_ylabel("sweeps ago" if i == 0 else "")
            ax_s.tick_params(labelsize=7)
            ax_w.tick_params(labelsize=7)
            cols[bid] = dict(ax_s=ax_s, ax_w=ax_w, l_spec=l_spec,
                             l_hold=l_hold, l_noise=l_noise, marks=marks,
                             im=im, overlays_for=None)
        bottom = gs[2, :].subgridspec(1, 2, wspace=0.22)
        ui.update(ids=list(ids), cols=cols,
                  ax_rssi=fig.add_subplot(bottom[0]),
                  ax_loc=fig.add_subplot(bottom[1]),
                  txt=fig.text(0.01, 0.985, "", family="monospace",
                               fontsize=8, va="top"))

    def draw_overlays(col, f_lo, f_hi):
        key = (round(f_lo), round(f_hi))
        if col["overlays_for"] == key:
            return
        col["overlays_for"] = key
        ax = col["ax_s"]
        for fc, w, lbl in OVERLAYS:
            if not f_lo <= fc <= f_hi:
                continue
            if w:
                ax.axvspan((fc - w / 2) / 1e6, (fc + w / 2) / 1e6,
                           color="#1f77b4", alpha=0.06)
                if lbl:
                    ax.text(fc / 1e6, 0.96, lbl, transform=
                            ax.get_xaxis_transform(), ha="center", va="top",
                            fontsize=7, color="#1f77b4")
            else:
                ax.axvline(fc / 1e6, color="#2ca02c", lw=0.7, alpha=0.5)

    def update(_frame):
        bands, tracks, status, aux, notices = state.snapshot()
        ids = [b["id"] for b in bands]
        if ui["ids"] != ids:
            build(ids)
        if not bands:
            ui["txt"].set_text("waiting for the first sweep...")
            return ()

        top = [link_line(state, link)] + aux_lines(aux)
        if notices:
            top.append("note: " + notices[-1])
        ui["txt"].set_text("\n".join(top[:4]))

        for b in bands:
            col = ui["cols"][b["id"]]
            sw = b["sweep"]
            f = np.array(sw.freqs_hz()) / 1e6
            y = np.asarray(sw.bins_dbm)
            ax = col["ax_s"]
            ax.set_title(band_title(b), loc="left", fontsize=8,
                         color="#d62728" if sw.clipped else "black")
            col["l_spec"].set_data(f, y)
            if b["maxhold"] is not None and len(b["maxhold"]) == len(f):
                col["l_hold"].set_data(f, b["maxhold"])
            col["l_noise"].set_ydata([sw.noise_dbm, sw.noise_dbm])
            ax.set_ylim(sw.noise_dbm - 12,
                        max(y.max(), sw.noise_dbm + 25) + 5)
            ax.set_xlim(f[0], f[-1])
            draw_overlays(col, sw.f_lo_hz, sw.f_hi_hz)
            dets = b["dets"]
            col["marks"].set_offsets(
                np.c_[[d["f_hz"] / 1e6 for d in dets],
                      [d["peak_dbm"] + 2 for d in dets]]
                if dets else np.empty((0, 2)))
            wf = b["wf"]
            if wf is not None and len(wf) > 1:
                col["im"].set_data(wf[::-1])
                col["im"].set_extent([f[0], f[-1], len(wf), 0])
                vmin = sw.noise_dbm - 5
                vmax = max(np.percentile(wf, 99.5), vmin + 20)
                col["im"].set_clim(vmin, vmax)

        names = {b["id"]: b["name"] for b in bands}
        now = state.pi_now()

        # RSSI vs time
        ax_r = ui["ax_rssi"]
        ax_r.clear()
        ax_r.set_title("tracked emitters, all bands - RSSI vs time "
                       "(median filtered)", loc="left", fontsize=9)
        ax_r.set_xlabel("seconds ago", fontsize=8)
        ax_r.set_ylabel("dBm", fontsize=8)
        ax_r.grid(alpha=0.25)
        ax_r.tick_params(labelsize=7)
        drawn = 0
        for tr in tracks[:6]:
            if len(tr.t_hist) < 2:
                continue
            ts = np.array(tr.t_hist) - now
            lbl = f"{tr.f_hz/1e6:.1f} MHz"
            if tr.label:
                lbl += f" {tr.label}"
            ax_r.plot(ts, list(tr.med_hist), lw=1.4, label=lbl)
            ax_r.plot(ts, list(tr.raw_hist), lw=0.5, alpha=0.25,
                      color=ax_r.lines[-1].get_color())
            drawn += 1
        if drawn:
            ax_r.legend(fontsize=7, loc="lower left", ncol=2)
            ax_r.set_xlim(-args.history, 0)

        # relative range
        ax_l = ui["ax_loc"]
        ax_l.clear()
        ax_l.set_title("relative range - 1.0x = closest you have been",
                       loc="left", fontsize=9)
        ax_l.set_xlabel("relative range (unitless)", fontsize=8)
        ax_l.grid(alpha=0.25, axis="x")
        ax_l.tick_params(labelsize=7)
        shown = [tr for tr in tracks[:6]
                 if not math.isnan(tr.relative_range(state.pathloss_n))]
        if shown:
            ylabels, vals, cols, notes = [], [], [], []
            for tr in shown:
                rel = tr.relative_range(state.pathloss_n)
                ylabels.append(f"{tr.f_hz/1e6:.1f}"
                               + (f"\n{tr.label}" if tr.label else
                                  f"\n{names.get(tr.band_id, '')}"))
                vals.append(rel)
                s = tr.trend_db_per_s()
                cols.append("#2ca02c" if s > 0.4 else
                            "#d62728" if s < -0.4 else "#7f7f7f")
                notes.append(f"{tr.verdict()}  {tr.rssi:.0f} dBm")
            yy = np.arange(len(shown))
            ax_l.barh(yy, vals, color=cols, alpha=0.75)
            ax_l.set_yticks(yy)
            ax_l.set_yticklabels(ylabels, fontsize=7)
            ax_l.invert_yaxis()
            ax_l.axvline(1.0, color="#333333", lw=1.0, ls="--")
            ax_l.set_xlim(0, max(3.0, max(vals) * 1.25))
            for yi, (v, note) in enumerate(zip(vals, notes)):
                ax_l.text(v + max(vals) * 0.02, yi, note, va="center",
                          fontsize=7)
            ax_l.text(0.99, 0.02, f"path-loss n={state.pathloss_n}",
                      transform=ax_l.transAxes, ha="right", fontsize=7,
                      color="#666666")
        return ()

    if args.save_png:
        # Headless: collect for a while, draw one frame, write it out. No
        # animation object here - an unrendered one just warns at exit.
        deadline = time.time() + args.save_after
        while time.time() < deadline and not stop.is_set():
            time.sleep(0.2)
        update(0)
        fig.savefig(args.save_png, dpi=110)
        print(f"wrote {args.save_png}", file=sys.stderr)
    else:
        anim = FuncAnimation(fig, update, interval=args.refresh,
                             blit=False, cache_frame_data=False)
        fig._anim = anim              # keep a reference or it gets collected
        plt.show()
    stop.set()


# --------------------------------------------------------------------- replay

def replay(path, state: RxState, stop: threading.Event, speed=1.0):
    """Feed a CSV log back through the same pipeline as a live link.

    Reads every layout this chain has written (see rssi_wire.parse_csv_row),
    including logs from before multi-band, which replay as one band."""
    import csv as csvmod
    with open(path, newline="") as fh:
        rows = csvmod.reader(fh)
        hdr = next(rows, None)
        if not hdr or hdr[0] != "t":
            print(f"{path}: no header row, cannot replay", file=sys.stderr)
            return
        prev_t = None
        for row in rows:
            if stop.is_set():
                return
            try:
                sw = parse_csv_row(hdr, row)
            except (ValueError, IndexError):
                continue
            if not sw.bins_dbm:
                continue
            state.add(sw, 0)
            if prev_t is not None and speed > 0:
                time.sleep(min(1.0, max(0.0, (sw.t - prev_t) / speed)))
            prev_t = sw.t


# ------------------------------------------------------------------------ CLI

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default=None,
                    help="serial port (default: autodetect - on Windows pass "
                         "the COM port, autodetect can pick a Bluetooth one)")
    ap.add_argument("--baud", type=int, default=57600)
    ap.add_argument("--bps", type=float, default=600,
                    help="our own write budget - we mostly listen (default: 600)")
    ap.add_argument("--no-gui", action="store_true", help="text dashboard instead")
    ap.add_argument("--csv", default=None, help="log every sweep as it arrives")
    ap.add_argument("--aux-csv", default=None,
                    help="log every aux message (GNSS etc) as t,chan,len,hex")
    ap.add_argument("--aux-handler", action="append", default=[],
                    metavar="CH:MOD:CLASS",
                    help="decoder for an aux channel, e.g. "
                         "4:gnss_source:GnssHandler (in addition to "
                         "BASE_HANDLERS in aux_channels.py)")
    ap.add_argument("--replay", default=None, help="replay a CSV instead of a radio")
    ap.add_argument("--replay-speed", type=float, default=1.0)
    ap.add_argument("--snr", type=float, default=8.0,
                    help="dB above noise floor to call something an emitter")
    ap.add_argument("--max-emitters", type=int, default=8,
                    help="tracked emitters per band (default: 8)")
    ap.add_argument("--track", default=None,
                    help="comma-separated MHz to follow instead of auto-picking, "
                         "e.g. --track 2412,5806 - applies in whichever band "
                         "contains each frequency")
    ap.add_argument("--pathloss-n", type=float, default=DEF_PATHLOSS_N,
                    help="log-distance exponent: 2 free space, 3-4 indoors "
                         f"(default: {DEF_PATHLOSS_N})")
    ap.add_argument("--median", type=int, default=DEF_MEDIAN_N,
                    help="sweeps in the median filter (default: 5)")
    ap.add_argument("--merge-gap", type=int, default=3,
                    help="bridge dropouts up to N bins wide so a faded "
                         "wideband carrier stays one emitter (default: 3)")
    ap.add_argument("--track-tol", type=float, default=5.0,
                    help="MHz a track may drift and still be the same "
                         "emitter; narrow bands use less (default: 5)")
    ap.add_argument("--waterfall", type=int, default=300, help="waterfall rows")
    ap.add_argument("--history", type=float, default=120.0,
                    help="seconds shown in the RSSI plot")
    ap.add_argument("--refresh", type=int, default=250, help="GUI refresh ms")
    ap.add_argument("--save-png", default=None,
                    help="render one frame to this file and exit (headless)")
    ap.add_argument("--save-after", type=float, default=15.0,
                    help="seconds to collect before saving the PNG")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    state = RxState(waterfall_rows=args.waterfall, snr_db=args.snr,
                    max_emitters=args.max_emitters,
                    pathloss_n=args.pathloss_n, median_n=args.median,
                    merge_gap_bins=args.merge_gap,
                    track_tol_hz=args.track_tol * 1e6,
                    handlers=build_handlers(args.aux_handler))
    if args.track:
        state.pin([float(x) * 1e6 for x in args.track.split(",")])

    stop = threading.Event()
    csv = None
    csv_fh = None
    aux_csv = None
    link = None

    if args.replay:
        threading.Thread(target=replay,
                         args=(args.replay, state, stop, args.replay_speed),
                         daemon=True).start()

        class _NullParser:
            crc_errors = 0
            resyncs = 0

        class _NullLink:
            parser = _NullParser()
            frames_recv = 0
        link = _NullLink()
        print(f"replaying {args.replay}", file=sys.stderr)
    else:
        from linkmux import pick_port
        from siklink import Link
        link = Link(pick_port(args.port), args.baud, bps=args.bps)
        print(f"listening on {link.ser.port} @ {args.baud} baud", file=sys.stderr)

        if args.csv:
            csv_fh = open(args.csv, "a", buffering=1)
            if csv_fh.tell() == 0:
                csv_fh.write(CSV_HEADER + "\n")
            csv = DelayedSweepLog(csv_fh, state.position_at)
        if args.aux_csv:
            aux_csv = open(args.aux_csv, "a", buffering=1)
            if aux_csv.tell() == 0:
                aux_csv.write("t_rx,chan,len,hex\n")
        threading.Thread(target=reader,
                         args=(link, state, stop, csv, args.verbose, aux_csv),
                         daemon=True).start()

    try:
        if args.no_gui:
            run_text(state, stop, link)
        else:
            run_gui(state, stop, link, args)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        time.sleep(0.2)
        el = max(1e-9, time.time() - (state.t_first or time.time()))
        per = ", ".join(f"{bv.name} {bv.n_sweeps}"
                        for _, bv in sorted(state.bands.items()))
        print(f"\nsweeps {state.n_sweeps} ({state.n_sweeps/el:.1f}/s)"
              f"{' [' + per + ']' if per else ''}  "
              f"undecodable {state.n_bad}  clipped {state.n_clipped}  "
              f"loss {state.seqs.loss_pct:.1f}%  "
              f"crc errors {link.parser.crc_errors}", file=sys.stderr)
        for ch, av in sorted(state.aux.items()):
            print(f"aux ch{ch} {av.handler.name}: {av.msgs} msgs, "
                  f"{av.bytes} B  {av.text}", file=sys.stderr)
        if csv:
            csv.close()                 # writes the rows still held back
        for f in (csv_fh, aux_csv):
            if f:
                f.close()
        if hasattr(link, "close"):
            link.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
