#!/usr/bin/env python3
"""sdr_scan - sweep one or more bands on an AD9363 and turn IQ into RSSI bins.

Runs on the Pi against the OpenSourceSDRLab Zynq7020+AD9363 board over USB or
Ethernet (it speaks PlutoSDR's libiio interface). Also runs with no hardware at
all via the simulated backend, which is how the test suite exercises the DSP.

    python3 sdr_scan.py --source sim --sweeps 3        # no hardware needed
    python3 sdr_scan.py --uri ip:192.168.2.1 --sweeps 5
    python3 sdr_scan.py --uri ip:192.168.2.1 --band ism24 --csv survey.csv

Which bands: bands.py (DEFAULT_BANDS), or --band on the command line. Each band
gets its own tile plan, bin count and gain; they share the one radio and are
swept one after another, so a "cycle" is one sweep of every band.

WHY IT SWEEPS.  The AD9363 gives you about 20 MHz of instantaneous bandwidth.
The ISM band is 83.5 MHz. So the band is cut into tiles, each tile is captured
and FFT'd separately, and the pieces are stitched onto one output grid. Only
the middle ~75% of each tile is kept, because the analog anti-alias filter rolls
off at the edges and would otherwise put a fake dip into your spectrum every
15 MHz. Tiles are spaced so their kept regions abut exactly - no gaps, no
double-counting.

THE THING THAT WILL RUIN YOUR RSSI.  Leave the AGC on and the radio silently
re-gains between tiles, so a strong signal in tile 2 changes the apparent noise
floor in tile 5, and your "signal got stronger" reading is really "the AGC
backed off". Manual gain is the default here and the code warns loudly if it
finds AGC enabled. Every sweep also carries the gain it was taken at, so the
base station can spot it if it happens anyway.
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import numpy as np

from bands import Band, gain_limits_db, resolve_bands, validate
from rssi_wire import BAND_START_HZ, BAND_STOP_HZ, CSV_HEADER, Sweep, csv_row

# ------------------------------------------------------------------ defaults

DEF_SAMPLE_RATE = 20_000_000     # AD9363 tops out here; more needs an AD9364
DEF_NBINS = 256                  # 83.5 MHz / 256 = 326 kHz per bin
DEF_NFFT = 1024
DEF_NAVG = 8                     # Welch segments; variance goes as 1/navg
DEF_USABLE = 0.75                # fraction of each tile that clears the filter
DEF_GAIN_DB = 40
DEF_DC_NOTCH = 2                 # fine bins either side of DC to repair
CLIP_THRESHOLD = 0.98            # |sample| at or above this = saturated ADC

# dBm = dBFS - hardware_gain + CAL_OFFSET_DB.
# This default is a plausible starting point, NOT a calibration. See calibrate()
# for how to measure yours. Localisation only needs *relative* RSSI, so an
# uncalibrated offset shifts every number by a constant and changes nothing
# about which direction is warmer.
DEF_CAL_OFFSET_DB = -10.0


# -------------------------------------------------------------- tile planning

class TilePlan:
    """How the band is cut up, which slice of each tile is trusted, and where
    the local oscillator goes.

    Centres are spaced by `step`, and each tile owns exactly [fc-step/2,
    fc+step/2). Because step <= usable bandwidth, every owned frequency sits
    inside the flat part of that tile's filter response. Tiles abut exactly,
    so no frequency is measured twice and none is missed.

    THE DC PROBLEM AND WHY lo_mode EXISTS.  Every direct-conversion receiver -
    the AD9363 included - puts a large spike at whatever frequency the LO is
    tuned to, from LO leakage and ADC offset. It is an artefact. If you leave
    it in, you get a fake emitter at every tile centre. If you notch it out,
    you get a BLIND SPOT at every tile centre instead, and a real narrowband
    signal sitting there vanishes completely. Neither is acceptable if you are
    hunting a specific transmitter and have no idea what frequency it is on.

    'offset' (default) fixes it properly: the LO is parked to one side so that
    DC lands outside the region the tile is responsible for. Nothing is
    notched, nothing is blind. The cost is that only one side of the baseband
    spectrum gets used, so the band needs about twice as many tiles.

    'notch' is the fast version: LO at the tile centre, DC interpolated away,
    ~40 kHz blind at each tile centre. Use it if you are sweep-rate limited and
    you know your target is wideband - a 40 kHz notch in a 20 MHz WiFi carrier
    costs 0.2% of its power and does not matter.

    blind_spots() tells you exactly which frequencies are affected in 'notch'
    mode, and returns nothing in 'offset' mode.
    """

    def __init__(self, f_start=BAND_START_HZ, f_stop=BAND_STOP_HZ,
                 sample_rate=DEF_SAMPLE_RATE, usable_frac=DEF_USABLE,
                 lo_mode="offset", lo_guard_hz=500e3, dc_notch=DEF_DC_NOTCH):
        if f_stop <= f_start:
            raise ValueError("f_stop must be above f_start")
        if not 0.1 <= usable_frac <= 1.0:
            raise ValueError("usable_frac must be in 0.1..1.0")
        if lo_mode not in ("offset", "notch"):
            raise ValueError("lo_mode must be 'offset' or 'notch'")

        self.f_start = float(f_start)
        self.f_stop = float(f_stop)
        self.sample_rate = float(sample_rate)
        self.usable_frac = float(usable_frac)
        self.lo_mode = lo_mode
        self.lo_guard_hz = float(lo_guard_hz)
        self.dc_notch = int(dc_notch)

        span = self.f_stop - self.f_start
        flat_half_bw = self.sample_rate / 2.0 * self.usable_frac

        if lo_mode == "offset":
            # Only the positive half of baseband is used, and the guard keeps
            # DC clear of the owned region, so the widest tile is one-sided.
            max_step = flat_half_bw - self.lo_guard_hz
            if max_step <= 0:
                raise ValueError("lo_guard_hz leaves no usable bandwidth")
        else:
            max_step = 2.0 * flat_half_bw

        self.n_tiles = max(1, math.ceil(span / max_step))
        self.step = span / self.n_tiles
        self.centres = [self.f_start + self.step * (i + 0.5)
                        for i in range(self.n_tiles)]
        self.lo_offset = (self.step / 2.0 + self.lo_guard_hz
                          if lo_mode == "offset" else 0.0)

    def lo_freq(self, i: int) -> float:
        """Where to actually tune for tile i - not the same as its centre."""
        return self.centres[i] - self.lo_offset

    def edges(self, i: int) -> tuple[float, float]:
        """Frequency range tile i is responsible for."""
        return (self.centres[i] - self.step / 2.0,
                self.centres[i] + self.step / 2.0)

    def blind_spots(self) -> list[float]:
        """Frequencies lost to the DC notch. Empty in 'offset' mode."""
        if self.lo_mode == "offset" or self.dc_notch <= 0:
            return []
        return list(self.centres)

    def blind_width_hz(self, nfft: int) -> float:
        if self.lo_mode == "offset" or self.dc_notch <= 0:
            return 0.0
        return (2 * self.dc_notch + 1) * self.sample_rate / nfft

    def __repr__(self):
        extra = (f", LO offset {self.lo_offset/1e6:.2f} MHz"
                 if self.lo_mode == "offset" else ", LO at tile centre")
        return (f"<TilePlan {self.f_start/1e6:.1f}-{self.f_stop/1e6:.1f} MHz, "
                f"{self.n_tiles} tiles of {self.step/1e6:.2f} MHz "
                f"@ {self.sample_rate/1e6:.1f} MSPS, {self.lo_mode}{extra}>")


# ------------------------------------------------------------------- sources

class SdrSource:
    """Interface a capture backend has to provide."""

    name = "abstract"

    def tune(self, freq_hz: float) -> None:
        raise NotImplementedError

    def capture(self, nsamples: int) -> np.ndarray:
        """Return `nsamples` complex samples normalised so |x| <= ~1.0."""
        raise NotImplementedError

    @property
    def gain_db(self) -> float:
        return 0.0

    def set_gain(self, gain_db: float) -> None:
        """Manual RX gain. Called between sweeps, never inside one."""

    def lo_range_hz(self) -> tuple[float, float] | None:
        """(min, max) LO the radio accepts, or None if it cannot say."""
        return None

    def temps(self) -> dict | None:
        """Board temperatures in deg C, e.g. {"rf": 52.1, "fpga": 68.4}.
        None if the radio cannot say. Costs a network round trip or two, so
        call it every few seconds, not every sweep."""
        return None

    def reconnect(self, stop=None, log=sys.stderr) -> float:
        """Reopen after the radio drops out. Returns seconds it took."""
        raise NotImplementedError

    def close(self) -> None:
        pass


class StreamStuck(OSError):
    """The SDR took the connection and every setting, but hung up the moment
    a sample buffer was opened (libiio prints 'Open unlocked: -32'). Its
    control side is alive and its streaming side is wedged; so far only a
    reboot of the SDR has cleared it."""


def _host(uri: str) -> str:
    return uri.split(":", 1)[1] if uri.startswith("ip:") else "192.168.2.1"


class PlutoSource(SdrSource):
    """The real board, over libiio/pyadi-iio.

    Install on the Pi with:
        sudo apt install libiio-utils python3-libiio
        pip3 install pyadi-iio --break-system-packages

    `uri` is usually 'ip:192.168.2.1' over the USB ethernet gadget, or
    'ip:<address>' over real ethernet. 'usb:' works too if libusb is set up.
    """

    name = "pluto"

    def __init__(self, uri="ip:192.168.2.1", sample_rate=DEF_SAMPLE_RATE,
                 gain_db=DEF_GAIN_DB, agc=False, buffer_size=8192,
                 flush_buffers=1, rf_bw=None, kernel_buffers=1):
        self.uri = uri
        self._cfg = dict(sample_rate=sample_rate, gain_db=gain_db,
                         buffer_size=buffer_size, rf_bw=rf_bw,
                         kernel_buffers=kernel_buffers)
        self.flush_buffers = int(flush_buffers)
        self.agc = bool(agc)
        self._scale = None
        self.sdr = None
        self.reconnects = 0
        self._connect(first=True)

    def _connect(self, first: bool = False) -> None:
        """Open the radio and put every setting back. Used at start-up and
        again by reconnect(), so a radio that dropped off USB comes back
        configured exactly as it was."""
        try:
            import adi
        except ImportError:
            sys.exit("pyadi-iio missing. "
                     "pip3 install pyadi-iio --break-system-packages")
        c = self._cfg
        sample_rate = c["sample_rate"]

        sdr = adi.Pluto(uri=self.uri)
        sdr.sample_rate = int(sample_rate)
        sdr.rx_rf_bandwidth = int(c["rf_bw"] or sample_rate * 0.9)
        sdr.rx_buffer_size = int(c["buffer_size"])

        # STALE DATA AFTER A RETUNE.  libiio queues 4 kernel buffers by
        # default and the DMA keeps filling them in the background, so right
        # after a retune up to 4 buffers can still hold samples from the OLD
        # LO. Discarding one (--flush 1) would then leave up to three stale
        # captures that get stitched into the wrong tile - a real signal then
        # shows up whole tile-steps (7 MHz multiples) from where it really
        # is, and vanishes from where it should be. ADI's own multi-tune
        # examples drop the queue to one buffer for this reason. Must happen
        # before the first rx() creates the buffer.
        try:
            sdr._rxadc.set_kernel_buffers_count(int(c["kernel_buffers"]))
        except Exception as e:                       # libiio v1, odd builds
            if self.flush_buffers < 4:
                print(f"note: could not set the kernel buffer count ({e}). "
                      f"Raising --flush to 4 so retunes cannot read stale "
                      f"samples.", file=sys.stderr)
                self.flush_buffers = 4

        if self.agc:
            sdr.gain_control_mode_chan0 = "slow_attack"
            if first:
                print("WARNING: AGC is on. The radio will re-gain between "
                      "tiles and your RSSI comparisons become meaningless. "
                      "Use manual gain for anything you intend to localise "
                      "with.", file=sys.stderr)
        else:
            sdr.gain_control_mode_chan0 = "manual"
            # Clamp to the gain table for wherever the LO was left last run,
            # or the driver can refuse it before we have tuned anywhere.
            lo, hi = gain_limits_db(float(sdr.rx_lo))
            sdr.rx_hardwaregain_chan0 = int(max(lo, min(hi, c["gain_db"])))
        self._gain = float(sdr.rx_hardwaregain_chan0)

        # Read back rather than trust: the driver quietly clamps to what the
        # part can actually do, and a 20 MSPS request can land at 19.99.
        self.actual_sample_rate = float(sdr.sample_rate)
        if first and abs(self.actual_sample_rate - sample_rate) > \
                sample_rate * 0.02:
            print(f"note: asked for {sample_rate/1e6:.2f} MSPS, "
                  f"got {self.actual_sample_rate/1e6:.2f} MSPS",
                  file=sys.stderr)

        # Prove it streams before calling it connected. After some drops the
        # SDR accepts the connection and all of the above, then hangs up the
        # moment a buffer is opened. Without this check reconnect() reported
        # "back after 1.0 s" and the very next sweep dropped again, forever.
        try:
            sdr.rx()
        except Exception as e:
            try:
                sdr.rx_destroy_buffer()
            except Exception:
                pass
            raise StreamStuck(getattr(e, "errno", None) or 32,
                              f"answers, but will not stream ({e})") from e
        self.sdr = sdr

    def reconnect(self, stop=None, log=sys.stderr) -> float:
        """Keep trying to reopen the radio until it answers. Returns seconds
        spent. A dropped USB link, an SDR that rebooted on a brown-out, a
        restarted iiod: all look like a broken pipe from here, and all of
        them come back if you wait. Ctrl-C (or `stop`) gives up."""
        t0 = time.monotonic()
        try:
            if self.sdr is not None:
                self.sdr.rx_destroy_buffer()
        except Exception:
            pass
        self.sdr = None
        delay, tries, hinted = 1.0, 0, False
        while True:
            try:
                self._connect()
                self.reconnects += 1
                return time.monotonic() - t0
            except Exception as e:
                tries += 1
                if isinstance(e, StreamStuck) and not hinted and \
                        time.monotonic() - t0 > 15:
                    hinted = True
                    print(f"  The SDR answers but will not stream - that has "
                          f"needed a reboot of the SDR so far:\n"
                          f"    ssh root@{_host(self.uri)} reboot     "
                          f"(or unplug it). Still trying meanwhile.",
                          file=log)
                if tries == 1 or tries % 10 == 0:
                    print(f"  SDR not back yet ({self._why(e)[:90]}) - still "
                          f"trying, {time.monotonic() - t0:.0f} s so far",
                          file=log)
                if stop is not None:
                    if stop.wait(delay):
                        raise KeyboardInterrupt
                else:
                    time.sleep(delay)
                delay = min(5.0, delay * 1.5)

    def _why(self, e: Exception) -> str:
        """pyadi turns every connection failure into 'No device found', which
        hides the one thing you want to know. Ask libiio directly: 'timed out'
        means the Pi can route to the SDR but nothing answers (SDR hung);
        'unreachable' / 'no route' means the network link itself is gone."""
        msg = str(e)
        if msg.strip() != "No device found":
            return msg
        try:
            import iio
            iio.Context(self.uri)
        except Exception as e2:
            return f"{msg}: {e2}"
        return msg

    def temps(self) -> dict | None:
        """AD936x die temperature and the Zynq's own (XADC) temperature.
        Either can be missing on odd firmware; both missing -> None."""
        ctx = getattr(self.sdr, "ctx", None) or getattr(self.sdr, "_ctx", None)
        if ctx is None:
            return None
        out = {}
        try:
            phy = ctx.find_device("ad9361-phy")
            ch = phy.find_channel("temp0", False)
            out["rf"] = round(float(ch.attrs["input"].value) / 1000.0, 1)
        except Exception:
            pass
        try:
            x = ctx.find_device("xadc")
            a = x.find_channel("temp0", False).attrs
            mc = (float(a["raw"].value) + float(a["offset"].value)) * \
                float(a["scale"].value)
            out["fpga"] = round(mc / 1000.0, 1)
        except Exception:
            pass
        return out or None

    def tune(self, freq_hz: float) -> None:
        self.sdr.rx_lo = int(freq_hz)
        # Buffers already in flight were filled at the OLD frequency. Reading
        # them as if they were the new one smears signals across the band and
        # is a genuinely nasty bug to chase, because the spectrum looks fine.
        for _ in range(self.flush_buffers):
            self.sdr.rx()

    def capture(self, nsamples: int) -> np.ndarray:
        if int(self.sdr.rx_buffer_size) != nsamples:
            self.sdr.rx_destroy_buffer()
            self.sdr.rx_buffer_size = int(nsamples)
        x = np.asarray(self.sdr.rx())

        # pyadi hands back 12-bit ADC counts, so roughly +/-2048 (as complex
        # floats - it builds I + jQ itself, so the dtype does not tell you).
        # This used to decide by whether the first capture peaked above 4
        # counts, which a quiet first capture on a low-gain band can fail -
        # and then every dBm figure is out by 66 dB for the whole run. Counts
        # are whole numbers; normalised floats are not. Test that instead.
        if self._scale is None and x.size:
            integral = bool(np.all(x.real == np.round(x.real)) and
                            np.all(x.imag == np.round(x.imag)))
            self._scale = 1.0 / 2048.0 if integral else 1.0
        return x.astype(np.complex128) * (self._scale or 1.0 / 2048.0)

    @property
    def gain_db(self) -> float:
        # Cached: reading it back over the network costs a round trip per
        # sweep, and nothing else changes it in manual mode.
        if self.agc:
            return float(self.sdr.rx_hardwaregain_chan0)
        return self._gain

    def set_gain(self, gain_db: float) -> None:
        if self.agc:
            return
        lo, hi = gain_limits_db(float(self.sdr.rx_lo))
        g = int(round(max(lo, min(hi, gain_db))))
        if g == int(round(self._gain)):
            return
        self.sdr.rx_hardwaregain_chan0 = g
        self._gain = float(self.sdr.rx_hardwaregain_chan0)
        for _ in range(self.flush_buffers):          # settle, like a retune
            self.sdr.rx()

    def lo_range_hz(self):
        try:
            txt = self.sdr._get_iio_attr_str("altvoltage0",
                                             "frequency_available", True)
            lo, _, hi = txt.strip().strip("[]").split()
            return float(lo), float(hi)
        except Exception:
            return None

    def close(self) -> None:
        try:
            self.sdr.rx_destroy_buffer()
        except Exception:
            pass


class SimSource(SdrSource):
    """Synthetic IQ with known emitters, for testing the chain without a radio.

    Emitters move: their power follows a log-distance path loss curve as the
    simulated receiver walks a slow circuit, with Rayleigh fading layered on
    top. That matters - it means the base station's warmer/colder display gets
    exercised against something that behaves like a real walk, not a constant.
    """

    name = "sim"

    def __init__(self, sample_rate=DEF_SAMPLE_RATE, gain_db=DEF_GAIN_DB,
                 noise_dbm=-115.0, cal_offset_db=DEF_CAL_OFFSET_DB,
                 emitters=None, seed=0, motion=True, fading=True,
                 lo_range=(70e6, 6e9), drop_after=None):
        self.sample_rate = float(sample_rate)
        self._gain = float(gain_db)
        self.cal_offset_db = float(cal_offset_db)
        self.noise_dbm = float(noise_dbm)
        self.rng = np.random.default_rng(seed)
        self.motion = motion
        self.fading = fading
        self.freq = 2_440_000_000.0
        self.t0 = time.monotonic()
        self.lo_range = lo_range
        # Testing only: raise a broken pipe once, this many seconds in, the way
        # the real radio does when it drops off USB.
        self.drop_at = None if drop_after is None else self.t0 + drop_after
        self.reconnects = 0

        # (centre Hz, bandwidth Hz, power dBm at closest approach, kind, period s)
        # Levels chosen to sit ~20 dB below full scale at the default 40 dB
        # gain, so the default demo is not permanently clipping.
        self.emitters = emitters if emitters is not None else [
            (2_412_000_000.0, 20_000_000.0, -72.0, "ofdm", 37.0),   # WiFi ch1
            (2_437_000_000.0, 20_000_000.0, -81.0, "ofdm", 61.0),   # WiFi ch6
            (2_462_000_000.0, 20_000_000.0, -90.0, "ofdm", 23.0),   # WiFi ch11
            (2_426_000_000.0, 1_000_000.0, -86.0, "cw", 17.0),      # BLE adv 38
            (2_480_000_000.0, 1_000_000.0, -94.0, "cw", 29.0),      # BLE adv 39
            # 5.8 GHz
            (5_785_000_000.0, 20_000_000.0, -86.0, "ofdm", 41.0),   # WiFi ch157
            (5_806_000_000.0, 8_000_000.0, -76.0, "ofdm", 19.0),    # FPV R5
            # 915-928 MHz
            (916_800_000.0, 125_000.0, -95.0, "ofdm", 13.0),        # LoRa 125k
            (921_400_000.0, 500_000.0, -84.0, "ofdm", 31.0),        # LoRa 500k
        ]

    def tune(self, freq_hz: float) -> None:
        self.freq = float(freq_hz)

    @property
    def gain_db(self) -> float:
        return self._gain

    def set_gain(self, gain_db: float) -> None:
        lo, hi = gain_limits_db(self.freq)
        self._gain = float(int(round(max(lo, min(hi, gain_db)))))

    def lo_range_hz(self):
        return self.lo_range

    def temps(self):
        # Warms up over ~10 min like a real board would, so the logging
        # path gets exercised. Obviously not a measurement.
        k = 1.0 - math.exp(-(time.monotonic() - self.t0) / 600.0)
        return {"rf": round(40 + 12 * k, 1), "fpga": round(48 + 20 * k, 1)}

    def reconnect(self, stop=None, log=sys.stderr) -> float:
        self.reconnects += 1
        time.sleep(0.5)
        return 0.5

    def _dbm_to_amp(self, dbm: float) -> float:
        """Power in dBm -> linear amplitude in full-scale units."""
        dbfs = dbm - self.cal_offset_db + self._gain
        return math.sqrt(10.0 ** (dbfs / 10.0))

    def _range_loss_db(self, period: float, t: float) -> float:
        """Extra path loss from the receiver being further away right now."""
        if not self.motion:
            return 0.0
        # Distance ratio swings between 1x and 6x on a slow sine.
        ratio = 1.0 + 2.5 * (1.0 - math.cos(2 * math.pi * t / period))
        return 10.0 * 2.5 * math.log10(ratio)      # n = 2.5

    def capture(self, nsamples: int) -> np.ndarray:
        if self.drop_at is not None and time.monotonic() >= self.drop_at:
            self.drop_at = None
            raise BrokenPipeError(32, "Broken pipe (simulated)")
        t = time.monotonic() - self.t0
        fs = self.sample_rate

        # Thermal noise: complex Gaussian at the stated density.
        namp = self._dbm_to_amp(self.noise_dbm)
        x = (self.rng.normal(0, namp / math.sqrt(2), nsamples)
             + 1j * self.rng.normal(0, namp / math.sqrt(2), nsamples))

        n_idx = np.arange(nsamples)
        fr = np.fft.fftfreq(nsamples, 1.0 / fs)
        for fc, bw, p_dbm, kind, period in self.emitters:
            off = fc - self.freq
            amp = self._dbm_to_amp(p_dbm - self._range_loss_db(period, t))
            if self.fading:
                # Rayleigh: one complex Gaussian draw per emitter per capture.
                f = abs(complex(self.rng.normal(0, 0.7071),
                                self.rng.normal(0, 0.7071)))
                amp *= max(0.1, f)

            if kind == "cw":
                # Outside Nyquist it would alias to a fake frequency. A real
                # receiver's analog filter removes it, so drop it here too.
                if abs(off) > fs / 2:
                    continue
                ph = self.rng.uniform(0, 2 * math.pi)
                x += amp * np.exp(1j * (2 * math.pi * off / fs * n_idx + ph))
            else:
                keep = np.abs(fr - off) <= bw / 2.0
                n_kept = int(keep.sum())
                if n_kept == 0:
                    continue
                # A wideband emitter spans several tiles, and each tile sees
                # only part of it. Scale by the FRACTION of the emitter's
                # bandwidth this tile can see, or every tile gets the full
                # power of the emitter and a 20 MHz carrier comes out looking
                # like a series of humps with the wrong total power.
                frac = (n_kept * fs / nsamples) / bw
                s = (self.rng.normal(0, 1, nsamples)
                     + 1j * self.rng.normal(0, 1, nsamples))
                S = np.fft.fft(s)
                S[~keep] = 0.0
                s = np.fft.ifft(S)
                rms = np.sqrt(np.mean(np.abs(s) ** 2))
                if rms > 0:
                    x += s * (amp * math.sqrt(min(1.0, frac)) / rms)

        # LO leakage, because the real part does this and the DSP has to cope.
        x += 0.002 * self._dbm_to_amp(self.noise_dbm) * 300

        # A real ADC saturates. Simulating that means the clip detector and the
        # gain suggestion get exercised instead of being dead code we hope works.
        np.clip(x.real, -1.0, 1.0, out=x.real)
        np.clip(x.imag, -1.0, 1.0, out=x.imag)
        return x


# ----------------------------------------------------------------- DSP core

def welch_psd(x: np.ndarray, nfft: int, navg: int,
              window: np.ndarray) -> np.ndarray:
    """Averaged power spectrum, fftshifted, normalised so total power is right.

    The normalisation is |X|^2 / (nfft * sum(w^2)). That is the density-correct
    convention: summing bins over a band recovers the true power in that band
    for noise AND for tones. The alternative (tone-correct) convention reads
    CW peaks right but under-reads noise by the window's noise bandwidth, and
    since the next step is summing fine bins into coarse ones, density-correct
    is the one that survives.
    """
    usable = (len(x) // nfft) * nfft
    if usable == 0:
        raise ValueError(f"need at least {nfft} samples, got {len(x)}")
    segs = x[:usable].reshape(-1, nfft)[:navg]

    X = np.fft.fft(segs * window, axis=1)
    p = np.mean(np.abs(X) ** 2, axis=0)
    p /= (nfft * np.sum(window ** 2))
    return np.fft.fftshift(p)


def repair_dc(p: np.ndarray, notch: int = DEF_DC_NOTCH) -> np.ndarray:
    """Interpolate across the DC spike left by LO leakage and ADC offset.

    Zynq/AD9363 boards always show a spike at the tune frequency that is an
    artefact, not a signal. Left alone it becomes a fake emitter in every
    single tile, which on a stitched sweep means a comb of fake peaks every
    13.9 MHz - very convincing and completely wrong.
    """
    if notch <= 0:
        return p
    p = p.copy()
    mid = len(p) // 2
    lo, hi = max(0, mid - notch), min(len(p) - 1, mid + notch)
    if lo == 0 or hi >= len(p) - 1:
        return p
    left, right = p[lo - 1], p[hi + 1]
    n = hi - lo + 1
    p[lo:hi + 1] = np.linspace(left, right, n + 2)[1:-1]
    return p


DEF_NOISE_PCTL = 20.0


def robust_noise_floor(dbm: np.ndarray,
                       pctl: float = DEF_NOISE_PCTL) -> tuple[float, float]:
    """(floor, spread) in dB, from a low percentile of the bins.

    Mean and standard deviation are obviously wrong here - two WiFi carriers
    drag a mean up by 10 dB and then everything looks like noise.

    The median is the usual robust answer, but it fails too, and it fails
    quietly. The median only finds the noise floor if emitters occupy less
    than half the band, and 2.4 GHz routinely breaks that: three overlapping
    WiFi carriers cover 60 of the 83.5 MHz. The median then sits INSIDE a
    carrier and reports a noise floor 15 dB too high, which pushes the
    detection threshold above the weak signals you most wanted to see.

    A low percentile survives that - it only needs a fifth of the band to be
    quiet. The cost is a slight low bias on a genuinely empty band, well under
    1 dB with any reasonable navg, which is harmless for a reference level.
    """
    finite = dbm[np.isfinite(dbm)]
    if finite.size == 0:
        return -120.0, 1.0
    floor = float(np.percentile(finite, pctl))
    # Spread from the quiet half only, so emitters cannot inflate it.
    quiet = finite[finite <= np.percentile(finite, 50.0)]
    mad = float(np.median(np.abs(quiet - np.median(quiet)))) if quiet.size else 0.5
    return floor, max(0.1, mad * 1.4826)    # 1.4826 -> sigma for a Gaussian


class SweepEngine:
    """Captures every tile of ONE band, stitches them, hands back one Sweep.

    Several engines can share one source - that is how multi-band works. Each
    engine owns its band's gain and re-applies it before its own sweep, so a
    low gain chosen for one band never leaks into the next.

    gain_db=None leaves the radio's gain alone (use that with --agc).
    """

    def __init__(self, source: SdrSource, nbins=DEF_NBINS, nfft=DEF_NFFT,
                 navg=DEF_NAVG, f_start=BAND_START_HZ, f_stop=BAND_STOP_HZ,
                 sample_rate=DEF_SAMPLE_RATE, usable_frac=DEF_USABLE,
                 cal_offset_db=DEF_CAL_OFFSET_DB, dc_notch=DEF_DC_NOTCH,
                 smooth_alpha=0.0, node_id=1, lo_mode="offset",
                 lo_guard_hz=500e3, band_id=0, name="", gain_db=None,
                 every=1):
        self.src = source
        self.band_id = int(band_id)
        self.name = name or f"{f_start/1e6:.0f}-{f_stop/1e6:.0f}"
        self.gain_db = None if gain_db is None else float(gain_db)
        self.every = max(1, int(every))
        self.nbins = int(nbins)
        self.nfft = int(nfft)
        self.navg = int(navg)
        self.cal_offset_db = float(cal_offset_db)
        self.dc_notch = int(dc_notch)
        self.smooth_alpha = float(smooth_alpha)
        self.node_id = int(node_id)

        rate = getattr(source, "actual_sample_rate", sample_rate)
        self.plan = TilePlan(f_start, f_stop, rate, usable_frac,
                             lo_mode=lo_mode, lo_guard_hz=lo_guard_hz,
                             dc_notch=dc_notch)
        self.sample_rate = self.plan.sample_rate

        self.df_out = (self.plan.f_stop - self.plan.f_start) / self.nbins
        self.f0_hz = self.plan.f_start + self.df_out / 2.0
        self.window = np.hanning(self.nfft + 1)[:-1]   # periodic, not symmetric
        self.nsamples = self.nfft * self.navg

        self.auto_gain = True           # sdr_tx --auto-gain may touch this band
        self._peak_fs = 0.0             # largest |I| or |Q| in the last sweep
        self._rms_dbfs = -200.0         # loudest tile's total power, dBFS
        self.last_peak_fs: float | None = None
        self.last_rms_dbfs: float | None = None
        self.seq = 0
        self._smoothed: np.ndarray | None = None
        self.last_sweep_secs = 0.0
        self.clip_count = 0

        if self.df_out < self.sample_rate / self.nfft:
            print(f"warning: output bins ({self.df_out/1e3:.0f} kHz) are "
                  f"narrower than the FFT resolution "
                  f"({self.sample_rate/self.nfft/1e3:.0f} kHz). Some bins will "
                  f"be interpolated rather than measured. Raise --nfft or "
                  f"lower --bins.", file=sys.stderr)

    def _tile_power(self, i: int):
        """Capture one tile. Returns (abs freqs, linear power, clipped)."""
        f_lo = self.plan.lo_freq(i)
        self.src.tune(f_lo)
        x = self.src.capture(self.nsamples)

        # Check saturation on the RAW samples, before the mean is removed -
        # afterwards the peak has moved and a clipped capture can look clean.
        # Per rail: I and Q are separate ADCs, and a sample at |I|=|Q|=0.7 has
        # magnitude 0.99 without either of them being anywhere near clipping.
        peak = float(max(np.max(np.abs(x.real)), np.max(np.abs(x.imag)))) \
            if x.size else 0.0
        clipped = peak >= CLIP_THRESHOLD

        x = x - np.mean(x)                       # kill the DC offset

        # How hard this tile drives the ADC, for suggest_gain(). This is the
        # TOTAL power in the ~18 MHz the filter passes, which is what clips -
        # not the strongest output bin, which for a 20 MHz carrier spread over
        # 60 bins is ~18 dB lower than what the ADC actually sees.
        ms = float(np.mean(x.real ** 2 + x.imag ** 2)) if x.size else 0.0
        self._peak_fs = max(self._peak_fs, peak)
        self._rms_dbfs = max(self._rms_dbfs,
                             10.0 * math.log10(ms) if ms > 0 else -200.0)

        p = welch_psd(x, self.nfft, self.navg, self.window)
        # In offset mode DC sits outside the owned region and gets discarded by
        # the keep mask below, so notching it would only destroy real signal.
        if self.plan.lo_mode == "notch":
            p = repair_dc(p, self.dc_notch)

        rel = np.fft.fftshift(np.fft.fftfreq(self.nfft, 1.0 / self.sample_rate))
        freqs = f_lo + rel                       # absolute, from the ACTUAL LO

        lo, hi = self.plan.edges(i)
        keep = (freqs >= lo) & (freqs < hi)
        return freqs[keep], p[keep], clipped

    def reachable(self, lo_range) -> bool:
        """Can the radio tune every LO this band needs?"""
        if not lo_range:
            return True
        lo, hi = lo_range
        return all(lo <= self.plan.lo_freq(i) <= hi
                   for i in range(self.plan.n_tiles))

    def _apply_gain(self) -> None:
        if self.gain_db is None:
            return
        if abs(float(self.src.gain_db) - self.gain_db) < 0.5:
            return
        # Tune first: which gain values are legal depends on the LO (the
        # driver swaps gain tables at 1.3 and 4 GHz), so setting a 5.8 GHz
        # band's gain while still parked at 915 MHz can be refused.
        self.src.tune(self.plan.lo_freq(0))
        self.src.set_gain(self.gain_db)

    def sweep(self) -> Sweep:
        t_start = time.monotonic()
        self._apply_gain()

        acc = np.zeros(self.nbins)
        cnt = np.zeros(self.nbins, dtype=np.int64)
        clipped = False
        self._peak_fs, self._rms_dbfs = 0.0, -200.0

        for i in range(self.plan.n_tiles):
            freqs, p, tile_clipped = self._tile_power(i)
            clipped |= tile_clipped
            idx = ((freqs - self.plan.f_start) / self.df_out).astype(np.int64)
            ok = (idx >= 0) & (idx < self.nbins)
            idx, p = idx[ok], p[ok]
            # Summing linear power is what makes a coarse bin mean "all the
            # energy in this 326 kHz", regardless of whether it holds one
            # carrier or a spread of them.
            acc += np.bincount(idx, weights=p, minlength=self.nbins)
            cnt += np.bincount(idx, minlength=self.nbins)

        gain = float(self.src.gain_db)

        with np.errstate(divide="ignore", invalid="ignore"):
            dbfs = 10.0 * np.log10(acc)
        dbfs[cnt == 0] = np.nan
        dbfs[~np.isfinite(dbfs)] = np.nan
        dbm = dbfs - gain + self.cal_offset_db
        dbm = _fill_gaps(dbm)

        if self.smooth_alpha > 0.0:
            # Exponential smoothing across sweeps. Keep this small on a moving
            # receiver: it trades the fading noise you want gone against a lag
            # in the RSSI trend, which is the exact signal you are navigating by.
            a = self.smooth_alpha
            self._smoothed = dbm if self._smoothed is None else \
                (1 - a) * self._smoothed + a * dbm
            dbm = self._smoothed.copy()

        noise, _ = robust_noise_floor(dbm)
        self.last_peak_fs, self.last_rms_dbfs = self._peak_fs, self._rms_dbfs
        self.last_sweep_secs = time.monotonic() - t_start
        self.seq += 1
        self.clip_count += int(clipped)

        return Sweep(seq=self.seq, t=time.time(), node_id=self.node_id,
                     f0_hz=self.f0_hz, df_hz=self.df_out,
                     bins_dbm=dbm.tolist(), noise_dbm=noise,
                     gain_db=int(round(gain)), clipped=clipped,
                     band_id=self.band_id)

    # Loudest tile's total power target. OFDM peaks run 10-12 dB over its RMS
    # and fading adds more, so -25 dBFS RMS leaves room before the rails.
    AGC_TARGET_RMS_DBFS = -25.0
    AGC_PEAK_MARGIN_DB = 6.0        # never push the largest sample closer
    AGC_MAX_STEP_UP_DB = 10.0       # climb slowly, drop fast

    def suggest_gain(self, sweep: Sweep) -> int | None:
        """Gain the radio should be at, or None if it is already right.

        Two failure modes, opposite ends. Too much gain and a close emitter
        saturates the ADC, which makes RSSI read LOW exactly when you are
        getting warm - the worst possible time to be wrong. Too little and
        weak emitters vanish into the quantisation noise.

        This used to aim the strongest OUTPUT BIN at -15 dBFS. That is fine
        for a CW tone, but a 20 MHz WiFi carrier spreads over ~60 bins, so its
        strongest bin sits ~18 dB under what the ADC actually sees - and the
        gain walked straight up into clipping, backed off, and did it again.
        Now it steers on the loudest tile's measured total power and peak
        sample, which is what actually clips.
        """
        g = sweep.gain_db
        lo, hi = gain_limits_db(self.plan.f_start)
        lo = max(0.0, lo)
        mine = (self.last_rms_dbfs is not None and self.seq == sweep.seq)
        if mine:
            adjust = self.AGC_TARGET_RMS_DBFS - self.last_rms_dbfs
            if self.last_peak_fs and self.last_peak_fs > 0:
                headroom = -20.0 * math.log10(self.last_peak_fs)
                adjust = min(adjust, headroom - self.AGC_PEAK_MARGIN_DB)
            adjust = min(adjust, self.AGC_MAX_STEP_UP_DB)
        else:
            # A sweep this engine did not just take: fall back to the bins.
            _, peak_dbm = sweep.peak()
            adjust = -15.0 - (peak_dbm + g - self.cal_offset_db)
        if sweep.clipped:
            adjust = min(adjust, -10.0)         # always back off on a clip
        elif abs(adjust) < 6.0:
            return None                         # inside the deadband, leave it
        want = int(max(lo, min(hi, round(g + adjust))))
        return None if want == g else want

    def describe(self) -> dict:
        return {
            "band": self.name,
            "band_id": self.band_id,
            "source": self.src.name,
            "lo_mode": self.plan.lo_mode,
            "blind_spots": len(self.plan.blind_spots()),
            "blind_khz_each": round(self.plan.blind_width_hz(self.nfft) / 1e3, 1),
            "tiles": self.plan.n_tiles,
            "tile_step_mhz": round(self.plan.step / 1e6, 3),
            "sample_rate_msps": round(self.sample_rate / 1e6, 3),
            "nbins": self.nbins,
            "bin_khz": round(self.df_out / 1e3, 2),
            "nfft": self.nfft,
            "navg": self.navg,
            "fft_res_khz": round(self.sample_rate / self.nfft / 1e3, 2),
            "gain_db": round(float(self.src.gain_db), 1),
            "cal_offset_db": self.cal_offset_db,
            "f_start_mhz": self.plan.f_start / 1e6,
            "f_stop_mhz": self.plan.f_stop / 1e6,
        }


def _fill_gaps(a: np.ndarray) -> np.ndarray:
    """Linear-interpolate any NaN bins so downstream code never sees one."""
    bad = ~np.isfinite(a)
    if not bad.any():
        return a
    if bad.all():
        return np.full_like(a, -120.0)
    idx = np.arange(len(a))
    a = a.copy()
    a[bad] = np.interp(idx[bad], idx[~bad], a[~bad])
    return a


# --------------------------------------------------------------- calibration

def calibrate(measured_dbm: float, known_dbm: float, cal_offset_db: float) -> float:
    """Work out the real cal offset from one known reference.

    Feed a signal generator at a known level (say -40 dBm) into the SDR, run a
    sweep, read the peak, and call this. Then pass the result as --cal-offset.

        new = calibrate(measured_dbm=-31.4, known_dbm=-40.0, cal_offset_db=-10.0)

    Do it at mid-band and mid-gain. The AD9363 is not flat across 83 MHz nor
    linear across 70 dB of gain, so one point gets you within a couple of dB
    over the useful range, not exact everywhere. For RSSI localisation that is
    fine: you are comparing the same emitter against itself over time, and a
    constant error cancels out entirely.
    """
    return cal_offset_db + (known_dbm - measured_dbm)


# ---------------------------------------------------------------------- CLI

def build_source(args) -> SdrSource:
    if args.source == "sim":
        return SimSource(sample_rate=args.sample_rate, gain_db=args.gain,
                         cal_offset_db=args.cal_offset, seed=args.seed,
                         motion=not args.sim_static,
                         drop_after=getattr(args, "sim_drop_after", None))
    try:
        return PlutoSource(uri=args.uri, sample_rate=args.sample_rate,
                           gain_db=args.gain, agc=args.agc,
                           buffer_size=args.nfft * args.navg,
                           flush_buffers=args.flush)
    except StreamStuck as e:
        sys.exit(f"SDR at {args.uri}: {e}.\nReboot it and try again: "
                 f"ssh root@{_host(args.uri)} reboot   (or unplug it)")


def add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--source", choices=["pluto", "sim"], default="pluto",
                    help="'sim' needs no hardware (default: pluto)")
    ap.add_argument("--uri", default="ip:192.168.2.1", help="libiio URI")
    ap.add_argument("--band", action="append", default=None, metavar="SPEC",
                    help="band(s) to scan, replacing the defaults in bands.py. "
                         "A preset name (ism24, ism58, ism915, gnss_l1, ...), "
                         "or name=START:STOP[:BINS[:GAIN[:EVERY]]] in MHz. "
                         "Repeat or comma-separate for several")
    ap.add_argument("--list-bands", action="store_true",
                    help="print the band presets and exit")
    ap.add_argument("--sample-rate", type=float, default=DEF_SAMPLE_RATE)
    ap.add_argument("--gain", type=float, default=DEF_GAIN_DB,
                    help="manual RX gain in dB for bands that do not set "
                         "their own in bands.py (default: 40)")
    ap.add_argument("--agc", action="store_true",
                    help="enable AGC - breaks RSSI comparability, avoid")
    ap.add_argument("--nfft", type=int, default=DEF_NFFT)
    ap.add_argument("--navg", type=int, default=DEF_NAVG)
    ap.add_argument("--usable", type=float, default=DEF_USABLE,
                    help="trusted fraction of each tile (default: 0.75)")
    ap.add_argument("--cal-offset", type=float, default=DEF_CAL_OFFSET_DB,
                    help="dBm = dBFS - gain + this, for bands without their "
                         "own cal_offset_db. See calibrate()")
    ap.add_argument("--dc-notch", type=int, default=DEF_DC_NOTCH,
                    help="fine bins repaired at DC, 'notch' mode only")
    ap.add_argument("--lo-mode", choices=["offset", "notch"], default="offset",
                    help="'offset' parks the LO to one side so the DC spike "
                         "lands outside the measured region - no artefacts and "
                         "no blind spots, but roughly twice the tiles. 'notch' "
                         "is faster and goes blind for ~98 kHz at each tile "
                         "centre (default: offset)")
    ap.add_argument("--lo-guard", type=float, default=500e3,
                    help="Hz between DC and the measured region (default: 500k)")
    ap.add_argument("--smooth", type=float, default=0.0,
                    help="EMA across sweeps, 0-1. 0 = off (default)")
    ap.add_argument("--flush", type=int, default=1,
                    help="buffers discarded after each retune (default: 1)")
    ap.add_argument("--node-id", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0, help="sim RNG seed")
    ap.add_argument("--sim-static", action="store_true",
                    help="sim: hold emitters still instead of simulating a walk")
    ap.add_argument("--sim-drop-after", type=float, default=None,
                    help=argparse.SUPPRESS)     # testing: fake a USB drop


def list_band_presets(out=sys.stdout) -> None:
    from bands import BAND_PRESETS, DEFAULT_BANDS
    for name, b in BAND_PRESETS.items():
        mark = "*" if name in DEFAULT_BANDS else " "
        print(f" {mark} {b}", file=out)
        if b.note:
            print(f"      {b.note}", file=out)
    print("\n * = scanned by default (DEFAULT_BANDS in bands.py)", file=out)


def build_engine(args, src: SdrSource, band: Band | None = None,
                 band_id: int = 0) -> SweepEngine:
    """One engine for one band. band=None means the first configured band."""
    if band is None:
        band = resolve_bands(getattr(args, "band", None))[0]
    gain = None if args.agc else (band.gain_db if band.gain_db is not None
                                  else args.gain)
    cal = band.cal_offset_db if band.cal_offset_db is not None \
        else args.cal_offset
    eng = SweepEngine(src, nbins=band.bins, nfft=args.nfft, navg=args.navg,
                       f_start=band.f_start_hz, f_stop=band.f_stop_hz,
                       sample_rate=args.sample_rate, usable_frac=args.usable,
                       cal_offset_db=cal, dc_notch=args.dc_notch,
                       smooth_alpha=args.smooth, node_id=args.node_id,
                       lo_mode=band.lo_mode or args.lo_mode,
                       lo_guard_hz=args.lo_guard, band_id=band_id,
                       name=band.name, gain_db=gain, every=band.every)
    eng.auto_gain = band.auto_gain
    return eng


def build_engines(args, src: SdrSource, log=sys.stderr):
    """Every configured band -> (engines, skipped, bands).

    band_id is the band's position in the CONFIGURED list, so it stays the
    same whether or not an earlier band had to be skipped - the base station
    keys its displays on it.

    A band the radio cannot tune is skipped with a warning rather than
    stopping the run. 5.8 GHz on an AD9363 that has not been switched to
    AD9364 mode is the case this is for: you still get the other bands.
    """
    try:
        bands = resolve_bands(getattr(args, "band", None))
    except ValueError as e:
        raise SystemExit(f"--band: {e}")
    for w in validate(bands):
        print(f"warning: {w}", file=log)

    lo_range = src.lo_range_hz()
    engines, skipped = [], []
    for bid, b in enumerate(bands):
        eng = build_engine(args, src, b, bid)
        if not eng.reachable(lo_range):
            lo, hi = lo_range
            why = (f"needs LO {eng.plan.lo_freq(0)/1e6:.1f}-"
                   f"{eng.plan.lo_freq(eng.plan.n_tiles-1)/1e6:.1f} MHz, radio "
                   f"tunes {lo/1e6:.0f}-{hi/1e6:.0f} MHz")
            if hi < 4e9 <= b.f_stop_hz:
                why += (". Above 3.8 GHz needs AD9364 mode: on the SDR run "
                        "fw_setenv attr_name compatible; "
                        "fw_setenv attr_val ad9364; reboot")
            print(f"WARNING: skipping band {b.name} - {why}", file=log)
            skipped.append((bid, b, why))
            continue
        engines.append(eng)
    if not engines:
        raise SystemExit("none of the configured bands can be tuned by this "
                         "radio - see the warnings above")
    return engines, skipped, bands


def _ascii_spectrum(sw: Sweep, width=72, rows=16) -> str:
    """A quick look at the band without needing a GUI on the Pi."""
    b = np.array(sw.bins_dbm)
    groups = np.array_split(b, width)
    vals = np.array([g.max() for g in groups])
    lo = sw.noise_dbm - 5
    hi = max(vals.max(), lo + 20)
    lines = []
    for r in range(rows, 0, -1):
        thresh = lo + (hi - lo) * r / rows
        lines.append(f"{thresh:6.0f} |" +
                     "".join("#" if v >= thresh else " " for v in vals))
    lines.append("       +" + "-" * width)
    lines.append(f"        {sw.f0_hz/1e6:.0f} MHz"
                 + " " * (width - 22)
                 + f"{(sw.f0_hz + sw.nbins*sw.df_hz)/1e6:.0f} MHz")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--sweeps", type=int, default=0,
                    help="cycles to run (one cycle = every band once), "
                         "0 = run forever")
    ap.add_argument("--interval", type=float, default=0.0,
                    help="minimum seconds between cycles")
    ap.add_argument("--csv", default=None, help="append every sweep to this file")
    ap.add_argument("--plot", action="store_true", help="ASCII spectrum per sweep")
    args = ap.parse_args()
    if args.list_bands:
        list_band_presets()
        return 0

    src = build_source(args)
    engines, skipped, _ = build_engines(args, src)
    for eng in engines:
        print(f"{eng.name}: {eng.plan}", file=sys.stderr)

    csv = open(args.csv, "a", buffering=1) if args.csv else None
    if csv and csv.tell() == 0:
        csv.write(CSV_HEADER + "\n")

    n = 0
    drops = []                     # [minutes-in, seconds-to-recover or None]
    t_run = time.monotonic()
    last_t, next_t = None, 0.0

    def fmt_t(d):
        if not d:
            return "n/a"
        return "  ".join(f"{k} {v:.0f} C" for k, v in
                         (("AD936x", d.get("rf")), ("Zynq", d.get("fpga")))
                         if v is not None)
    try:
        while args.sweeps == 0 or n < args.sweeps:
            t0 = time.monotonic()
            if t0 >= next_t:
                # Heat is a prime suspect when an SDR hangs after a while, so
                # log it: if drops line up with the temperature climbing,
                # it wants a heatsink / fan, not a software fix.
                next_t = t0 + 30.0
                try:
                    tt = src.temps()
                except OSError:
                    tt = None
                if tt:
                    last_t = tt
                    print(f"        temps  {fmt_t(tt)}  "
                          f"({(t0 - t_run)/60:.1f} min in)", file=sys.stderr)
            for eng in engines:
                if n % eng.every:
                    continue
                try:
                    sw = eng.sweep()
                except OSError as e:
                    # Radio stopped answering. Log when, get it back, carry
                    # on - so a long soak test counts drops instead of ending
                    # at the first one.
                    el = time.monotonic() - t_run
                    print(f"\n*** SDR connection lost at "
                          f"{time.strftime('%H:%M:%S')} ({el/60:.1f} min in, "
                          f"cycle {n+1}): {e} - reconnecting"
                          + (f"\n*** last temps: {fmt_t(last_t)}"
                             if last_t else ""), file=sys.stderr)
                    drops.append([el, None])   # counted even if it never returns
                    took = src.reconnect()
                    drops[-1][1] = took
                    print(f"*** back after {took:.1f} s\n", file=sys.stderr)
                    next_t = 0.0
                    continue
                i, p = sw.peak()
                print(f"[{n+1:5d}] {eng.name:<10} "
                      f"{eng.last_sweep_secs*1000:6.1f} ms  "
                      f"noise {sw.noise_dbm:7.2f} dBm  gain {sw.gain_db:2d}  "
                      f"peak {p:7.2f} dBm @ {(sw.f0_hz + i*sw.df_hz)/1e6:9.3f} MHz"
                      + ("  [CLIP]" if sw.clipped else ""))
                if args.plot:
                    print(_ascii_spectrum(sw))
                if csv:
                    csv.write(csv_row(sw, precision=2) + "\n")
            n += 1
            if args.interval:
                time.sleep(max(0.0, args.interval - (time.monotonic() - t0)))
    except KeyboardInterrupt:
        print("\nstopped", file=sys.stderr)
    finally:
        el = time.monotonic() - t_run
        print(f"\n{n} cycles in {el/60:.1f} min, SDR dropped out "
              f"{len(drops)} time(s)"
              + (": at " + ", ".join(
                  f"{d/60:.1f} min" + (f" (back in {t:.0f} s)" if t is not None
                                       else " (never came back)")
                  for d, t in drops) if drops else ""), file=sys.stderr)
        if last_t:
            print(f"last temps: {fmt_t(last_t)}", file=sys.stderr)
        src.close()
        if csv:
            csv.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
