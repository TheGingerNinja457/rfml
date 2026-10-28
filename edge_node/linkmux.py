"""linkmux - share one SiK link between the spectrum and everything else.

Runs on the Pi. sdr_tx.py owns one of these; anything else that wants airtime
(the GNSS hat, via aux_channels.py) goes through the same object, because only
one process can hold the serial port.

THE BUDGET SPLIT

    --kbps          everything this Pi may put on the air, framing included
    --reserve-kbps  held back for auxiliary channels (GNSS). The spectrum
                    never touches it, whether or not anything is using it

    spectrum + status   = kbps - reserve        (paced, see below)
    aux (GNSS etc)     <= reserve               (non-blocking, latest wins)

So with the defaults (17 and 3) the scanner uses at most 14 kbit/s and there
are always 3 kbit/s free for GNSS, even before any GNSS code exists.

PACING, NOT DROPPING.  The single-band version captured a sweep, then threw it
away if the budget said no. With three bands round-robin that starves the big
band: the small 915 MHz sweep always fits in the leftover tokens and the big
5.8 GHz one never does. So now the scanner waits for budget BEFORE it captures,
then captures and sends immediately. Nothing is ever queued (what arrives is
still the newest thing measured), no SDR time is wasted on sweeps that get
binned, and every band gets its turn.

THREAD SAFETY.  siklink.Link is not thread-safe - two threads writing at once
can interleave bytes mid-frame. Every frame goes out under one lock here. The
lock is taken per FRAGMENT, not per message, so a small GNSS message can slip
between the two halves of a big sweep instead of waiting behind it. siklink
reassembles per channel, so that interleaving is harmless.
"""

from __future__ import annotations

import sys
import threading
import time
from collections import defaultdict

from rssi_wire import SIK_DATA_HDR, wire_size

# How much a budget may bank when idle. Small on purpose: a big burst
# allowance lets a backlog fire in one go, and a SiK radio answers a sudden
# burst by overflowing its internal buffer and corrupting frames mid-flight.
BURST_SECS = 0.25

DATA = 0x01                     # siklink DATA message type


class ByteBudget:
    """Token bucket in wire bytes per second.

    Non-blocking on purpose: `allow()` answers yes or no and never sleeps.
    """

    def __init__(self, bps: float, burst_secs: float = BURST_SECS):
        self.bps = float(bps)
        self.capacity = self.bps * burst_secs
        self._tokens = self.capacity
        self._last = time.monotonic()
        self.spent = 0
        self.denied = 0

    def _refill(self) -> None:
        now = time.monotonic()
        self._tokens = min(self.capacity,
                           self._tokens + (now - self._last) * self.bps)
        self._last = now

    def allow(self, nbytes: int) -> bool:
        self._refill()
        if self._tokens >= nbytes:
            self._tokens -= nbytes
            self.spent += nbytes
            return True
        self.denied += 1
        return False


class Pacer(ByteBudget):
    """Token bucket that waits for room instead of refusing.

    wait(estimate) blocks until an item of about that size would fit;
    charge(actual) then takes the real size, which may push the bucket
    negative. The debt is paid back before the next wait() returns, so over
    any window T the bytes sent are at most bps*T + capacity + one item.
    """

    def wait(self, est_bytes: int, stop: threading.Event | None = None) -> float:
        """Sleep until est_bytes fits. Returns seconds spent waiting."""
        need = min(float(est_bytes), self.capacity)
        waited = 0.0
        while True:
            self._refill()
            if self._tokens >= need:
                return waited
            dt = min(0.25, (need - self._tokens) / self.bps)
            if stop is not None and stop.wait(dt):
                return waited
            if stop is None:
                time.sleep(dt)
            waited += dt

    def charge(self, nbytes: int) -> None:
        self._refill()
        self._tokens -= nbytes
        self.spent += nbytes


class LinkMux:
    """One SiK link, a paced spectrum budget, and a reserved aux budget."""

    def __init__(self, link, total_bps: float, reserve_bps: float,
                 burst_secs: float = BURST_SECS):
        if reserve_bps < 0 or reserve_bps >= total_bps:
            raise ValueError("reserve must be >= 0 and below the total budget")
        self.link = link
        self.total_bps = float(total_bps)
        self.reserve_bps = float(reserve_bps)
        self.spectrum = Pacer(total_bps - reserve_bps, burst_secs)
        # Aux gets a slightly deeper bucket: GNSS tends to arrive as a burst
        # of sentences once a second, not as a smooth trickle.
        self.aux = ByteBudget(reserve_bps, burst_secs=1.0) if reserve_bps else None
        self.max_payload = int(getattr(link, "max_payload", 220))
        self._lock = threading.Lock()
        self.t_start = time.monotonic()
        # per-channel counters: messages, wire bytes, refused
        self.chan_msgs = defaultdict(int)
        self.chan_bytes = defaultdict(int)
        self.chan_refused = defaultdict(int)

    # -- sizes ---------------------------------------------------------------

    def wire(self, nbytes: int) -> int:
        return wire_size(nbytes, self.max_payload)

    # -- sending -------------------------------------------------------------

    def _send(self, data: bytes, chan: int) -> None:
        room = self.max_payload - SIK_DATA_HDR
        if not data:
            with self._lock:
                self.link.send_msg(DATA, bytes([chan, 0]))
            return
        for off in range(0, len(data), room):
            piece = data[off:off + room]
            more = 1 if off + room < len(data) else 0
            with self._lock:
                self.link.send_msg(DATA, bytes([chan, more]) + piece)

    def wait_spectrum(self, est_bytes: int,
                      stop: threading.Event | None = None) -> float:
        """Block until the spectrum budget has room for about est_bytes."""
        return self.spectrum.wait(est_bytes, stop)

    def send_spectrum(self, data: bytes, chan: int) -> int:
        """Send on the spectrum budget. Call wait_spectrum() first if you
        want pacing; this charges the real cost either way and returns it."""
        w = self.wire(len(data))
        self.spectrum.charge(w)
        self._send(data, chan)
        self.chan_msgs[chan] += 1
        self.chan_bytes[chan] += w
        return w

    def try_send_aux(self, data: bytes, chan: int) -> bool:
        """Send on the reserved budget if there is room; never blocks on it.

        Safe to call from any thread. Returns False (and sends nothing) if
        the reserve is used up right now - the caller decides whether to
        retry, keep only the newest, or drop.
        """
        w = self.wire(len(data))
        if self.aux is None or not self.aux.allow(w):
            self.chan_refused[chan] += 1
            return False
        self._send(data, chan)
        self.chan_msgs[chan] += 1
        self.chan_bytes[chan] += w
        return True

    # -- reporting -----------------------------------------------------------

    def rates_kbps(self) -> dict:
        el = max(1e-9, time.monotonic() - self.t_start)
        aux = self.aux.spent if self.aux else 0
        return {"spectrum": self.spectrum.spent * 8 / el / 1000,
                "aux": aux * 8 / el / 1000,
                "total": (self.spectrum.spent + aux) * 8 / el / 1000}


# ------------------------------------------------------------ port picking

# USB-serial bridges SiK radios are built on.
_SIK_VIDS = {0x0403: "FTDI", 0x10C4: "SiLabs CP210x", 0x1A86: "WCH CH340",
             0x067B: "Prolific"}
# Things that also show up as serial ports and must never get spectrum data
# written into them: the Pluto's own Linux console, u-blox GNSS receivers.
_NOT_SIK_VIDS = {0x0456: "Analog Devices (the SDR's console)",
                 0x1546: "u-blox (a GNSS receiver)"}


def pick_port(explicit: str | None = None, log=sys.stderr) -> str:
    """The radio's serial port: explicit if given, else a careful guess.

    siklink's own guess takes the first ttyUSB/ttyACM - on the Pi that can be
    the SDR's USB console (ttyACM0) - and on Windows, where nothing matches
    those names, simply the first COM port, which is often a Bluetooth one.
    This prefers USB-serial chips SiK radios actually use, never picks the
    SDR or a u-blox GNSS, and says so when it is guessing between several.

    Best fix: pass --port. On the Pi, /dev/serial/by-id/usb-... never changes.
    """
    if explicit:
        return explicit
    try:
        from serial.tools import list_ports
    except ImportError:
        sys.exit("pyserial missing: pip install pyserial")

    ports = list(list_ports.comports())
    good, maybe = [], []
    for p in ports:
        vid = getattr(p, "vid", None)
        if vid in _NOT_SIK_VIDS:
            continue
        if vid in _SIK_VIDS:
            good.append(p)
        elif any(k in p.device for k in ("ttyUSB", "usbserial")) or \
                (vid is not None and "Bluetooth" not in (p.description or "")):
            maybe.append(p)
    cands = good or maybe
    if not cands:
        listing = "\n".join(f"  {p.device}  {p.description}" for p in ports)
        sys.exit("No likely radio port found. Pass --port. Ports seen:\n"
                 + (listing or "  (none)"))
    if len(cands) > 1:
        print("note: several possible radio ports - using the first. Pass "
              "--port to choose:", file=log)
        for p in cands:
            print(f"    {p.device}  {p.description}", file=log)
    return cands[0].device
