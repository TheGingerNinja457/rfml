"""aux_channels - put other data (the GNSS hat first) on the spectrum link.

Needed on BOTH ends. This is the generic slot; the GNSS HAT itself lives in
gnss_source.py and is registered below. There is also a test pattern, for
proving the reserved bandwidth works with any size and rate of message.

HOW IT FITS TOGETHER

    Pi:     AuxSource.read() -> AuxRunner thread -> LinkMux.try_send_aux()
            (own thread, own channel, reserved budget - never slows a sweep)
    link:   siklink DATA on channel 4..15
    laptop: base_rx reader -> AuxHandler.handle(blob) -> status line on screen

ADDING ANOTHER SOURCE (the GNSS hat is already in: gnss_source.py)

  1. Make a module next to these files with two classes, e.g. for GNSS:

        from aux_channels import AuxSource, AuxHandler
        from rssi_wire import CH_GNSS

        class GnssSource(AuxSource):          # runs on the Pi
            name = "gnss"
            chan = CH_GNSS
            def open(self):   ...  # open /dev/ttyAMA0 (or gpsd), etc.
            def read(self):   ...  # block <= ~0.5 s; return bytes to send,
                                   # or None if there is nothing new
            def close(self):  ...

        class GnssHandler(AuxHandler):        # runs on the laptop
            name = "gnss"
            def handle(self, blob, t_rx):
                ...                            # decode what read() sent
                return "fix 3D  -36.8523 174.7691  12 sats"   # shown on screen

  2. Add it to PI_SOURCES and BASE_HANDLERS below.
     (Or, without editing anything: sdr_tx.py --aux mymod:MySource
      and base_rx.py --aux-handler 5:mymod:MyHandler)

  That is all. Threading, budgeting, the lock on the serial port, dropping
  stale fixes and the on-screen stats are handled here.

SIZING GNSS AGAINST THE RESERVE.  The default reserve is 3 kbit/s = 375 B/s
of wire. Every message costs its payload + 21 B of framing.
    raw NMEA GGA+RMC at 1 Hz     ~150 B + 2x21  ~190 B/s   fits
    raw NMEA GGA+RMC at 5 Hz                    ~950 B/s   does NOT fit
    packed binary fix, ~24 B, at 5 Hz           ~225 B/s   fits
So either send NMEA at 1 Hz or pack the fix into a struct - the second is
what you want anyway once rates go up. The runner keeps only the newest
message when the reserve is full (keep_latest), which is the right call for
position: a stale fix is worse than a missing one.

TIMESTAMPS.  Sweeps are stamped with the Pi's clock. With no network in the
field the Pi's clock is only as good as its last NTP sync - which is the other
thing the GNSS hat fixes: run gpsd + chrony with the PPS pin and the Pi's
clock becomes GPS time, so every sweep timestamp is GPS-accurate for free, and
the base station can line positions up with sweeps by timestamp.
"""

from __future__ import annotations

import importlib
import struct
import sys
import threading
import time
from collections import deque

from rssi_wire import CH_AUX_FIRST, CH_AUX_LAST, CH_GNSS  # noqa: F401

CH_TEST = CH_AUX_LAST           # 15: the built-in test pattern


# ============================================================== Pi side

class AuxSource:
    """Base class for anything on the Pi that wants to send over the link.

    Subclass it and fill in open/read/close. read() runs in its own thread,
    so it is fine for it to block on a serial port - just return within about
    half a second (return None on a timeout) so shutdown is prompt.
    """

    name = "aux"
    chan = CH_GNSS
    keep_latest = True          # reserve full: keep only the newest message
    max_queue = 16              # ...or, if False, queue up to this many

    def open(self) -> None:
        pass

    def read(self) -> bytes | None:
        raise NotImplementedError

    def close(self) -> None:
        pass


# Register Pi-side sources here, as "module:Class" strings. They are imported
# only when sdr_tx.py starts, so a missing module does not break anything else.
PI_SOURCES: list[str] = [
    "gnss_source:GnssSource",          # MAX-M8Q HAT on /dev/ttyAMA0
]


class AuxRunner(threading.Thread):
    """Runs one AuxSource: read, then send on the reserved budget."""

    def __init__(self, source: AuxSource, mux, stop: threading.Event,
                 log=sys.stderr):
        super().__init__(daemon=True, name=f"aux-{source.name}")
        if not CH_AUX_FIRST <= source.chan <= CH_AUX_LAST:
            raise ValueError(f"{source.name}: channel {source.chan} is outside "
                             f"the aux range {CH_AUX_FIRST}-{CH_AUX_LAST}")
        self.src = source
        self.mux = mux
        self.stop = stop
        self.log = log
        self.pending: deque[bytes] = deque(
            maxlen=1 if source.keep_latest else source.max_queue)
        self.read_count = 0
        self.sent = 0
        self.superseded = 0     # replaced by a newer message before it went
        self.errors = 0
        self._last_msg = ""
        self._last_print = 0.0

    def _log(self, what: str, e: Exception) -> None:
        """Print an error, but not the same one every second forever."""
        msg = f"aux {self.src.name}: {what}: {e}"
        now = time.monotonic()
        if msg != self._last_msg or now - self._last_print > 30.0:
            print(msg, file=self.log)
            self._last_msg, self._last_print = msg, now

    def _flush(self) -> None:
        while self.pending:
            if not self.mux.try_send_aux(self.pending[0], self.src.chan):
                return
            self.pending.popleft()
            self.sent += 1

    def run(self) -> None:
        # Keep trying to open: a GNSS HAT can take a moment after boot, and
        # a missing one must not stop the spectrum - it just stays quiet.
        while not self.stop.is_set():
            try:
                self.src.open()
                break
            except Exception as e:
                self.errors += 1
                self._log("open failed (retrying every 5 s)", e)
                self.stop.wait(5.0)
        if self.stop.is_set():
            return
        try:
            while not self.stop.is_set():
                try:
                    msg = self.src.read()
                except Exception as e:
                    self.errors += 1
                    self._log("read failed", e)
                    self.stop.wait(1.0)
                    continue
                if msg:
                    self.read_count += 1
                    if len(self.pending) == self.pending.maxlen:
                        self.superseded += 1
                    self.pending.append(bytes(msg))
                self._flush()
                if self.pending and not msg:
                    self.stop.wait(0.05)    # budget full: brief back-off
        finally:
            try:
                self.src.close()
            except Exception:
                pass

    def stats(self) -> dict:
        return {"ch": self.src.chan, "read": self.read_count, "sent": self.sent,
                "superseded": self.superseded, "err": self.errors}


class TestPatternSource(AuxSource):
    """Stand-in for GNSS: fixed-size messages at a fixed rate on channel 15.

        sdr_tx.py --aux-test 5        # 5 msgs/s of 32 B, about a packed fix
        sdr_tx.py --aux-test 1:150    # 1 msg/s of 150 B, about raw NMEA

    base_rx recognises it and shows rate, loss and latency, so you can see
    the reserved budget doing its job while the spectrum runs flat out.
    """

    name = "test"
    chan = CH_TEST
    MAGIC = b"TP"
    HDR = struct.Struct("<2sId")        # magic, seq, send time

    def __init__(self, rate_hz: float = 5.0, size: int = 32):
        self.period = 1.0 / max(0.01, float(rate_hz))
        self.size = max(self.HDR.size, int(size))
        self.seq = 0
        self._next = time.monotonic()

    def read(self) -> bytes | None:
        now = time.monotonic()
        if now < self._next:
            time.sleep(min(0.5, self._next - now))
            return None
        self._next += self.period
        if self._next < now:                # fell behind; do not burst
            self._next = now + self.period
        self.seq += 1
        body = self.HDR.pack(self.MAGIC, self.seq, time.time())
        return body + bytes(self.size - len(body))


# ========================================================== laptop side

class AuxHandler:
    """Base class for decoding one aux channel at the base station.

    handle() gets every reassembled message on its channel and returns a
    short string to show on screen (or None to keep showing the last one).
    """

    name = "aux"

    def handle(self, blob: bytes, t_rx: float) -> str | None:
        return None

    def close(self) -> None:
        pass


# Register base-station handlers here: channel -> "module:Class".
BASE_HANDLERS: dict[int, str] = {
    CH_GNSS: "gnss_source:GnssHandler",
}


class TestPatternHandler(AuxHandler):
    name = "test"

    def __init__(self):
        self.last_seq = None
        self.got = 0
        self.lost = 0
        self.lat = deque(maxlen=50)

    def handle(self, blob, t_rx):
        if len(blob) < TestPatternSource.HDR.size or \
                blob[:2] != TestPatternSource.MAGIC:
            return "not a test pattern"
        _, seq, t_tx = TestPatternSource.HDR.unpack_from(blob)
        if self.last_seq is not None and seq > self.last_seq:
            self.lost += seq - self.last_seq - 1
        self.last_seq = seq
        self.got += 1
        self.lat.append(t_rx - t_tx)
        lat = sorted(self.lat)[len(self.lat) // 2]
        loss = 100.0 * self.lost / max(1, self.got + self.lost)
        return (f"test #{seq}  {len(blob)} B  loss {loss:.1f}%  "
                f"latency {lat*1000:.0f} ms (needs synced clocks)")


class GenericHandler(AuxHandler):
    """Used for any aux channel nobody registered a handler for yet."""

    def __init__(self, chan):
        self.name = f"ch{chan}"

    def handle(self, blob, t_rx):
        head = blob[:24]
        try:
            txt = head.decode("ascii")
            shown = txt if txt.isprintable() else head.hex()
        except UnicodeDecodeError:
            shown = head.hex()
        return f"{len(blob)} B  {shown}"


BUILTIN_HANDLERS = {CH_TEST: TestPatternHandler}


# =============================================================== loading

def load(spec: str):
    """'module:Class' -> an instance of Class (no constructor arguments)."""
    if ":" not in spec:
        raise ValueError(f"'{spec}' should look like module:Class")
    mod, cls = spec.rsplit(":", 1)
    return getattr(importlib.import_module(mod), cls)()


def build_handlers(extra: list[str] | None = None, log=sys.stderr) -> dict:
    """chan -> AuxHandler, from BUILTIN + BASE_HANDLERS + --aux-handler."""
    out = {ch: cls() for ch, cls in BUILTIN_HANDLERS.items()}
    specs = dict(BASE_HANDLERS)
    for item in extra or []:
        ch, spec = item.split(":", 1)
        specs[int(ch)] = spec
    for ch, spec in specs.items():
        try:
            out[ch] = load(spec)
        except Exception as e:
            print(f"aux handler {spec} for ch{ch} failed to load: {e}",
                  file=log)
    return out
