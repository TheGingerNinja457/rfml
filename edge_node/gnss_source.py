"""gnss_source - the MAX-M8Q GNSS HAT on the spectrum link (channel 4).

Needed on BOTH ends. The Pi half (GnssSource) wraps max_m8q_gps.py; the laptop
half (GnssHandler) only needs this file - it never imports the serial code.

    Pi:     max_m8q_gps.MaxM8Q -> GnssSource.read() -> 31 B fix on channel 4
    laptop: GnssHandler.handle() -> position track -> lat/lon on every sweep
                                                      row of base_rx --csv

Registered in aux_channels.py, so sdr_tx.py and base_rx.py pick it up with no
flags. On the Pi:

    sdr_tx.py                          uses /dev/ttyAMA0 (the Pi 5 GPIO UART)
    sdr_tx.py --gnss-port /tmp/GPS     somewhere else (e.g. fake_gnss.py)
    sdr_tx.py --no-gnss                leave it out

WHY A PACKED MESSAGE, NOT NMEA.  One fix is 31 bytes here, 52 on the air with
framing: 0.4 kbit/s at the M8Q's default 1 Hz, 2.1 kbit/s even at 5 Hz. The
same fix as raw GGA+RMC is ~150 B and does not fit at 5 Hz.

WHERE EACH FIX GOES IN TIME.  A fix is stamped with the Pi's clock at the
moment its GGA sentence arrived (~0.1-0.2 s after the fix epoch at 9600 baud).
If the Pi's clock agrees with GPS time to within half a second - which it will
once chrony is disciplined by the HAT's PPS pin, see README - the exact GPS
epoch time is used instead. Sweeps are stamped with the same Pi clock, so the
two line up without any clock sync between Pi and laptop.

PORT.  Always pass the port explicitly (it defaults to /dev/ttyAMA0 here).
max_m8q_gps's autodetect probes USB serial ports too, and on the drone
/dev/ttyUSB0 is the SiK radio: probing it at 9600 baud reconfigures the port
the spectrum link is using.
"""

from __future__ import annotations

import bisect
import calendar
import math
import os
import struct
import threading
import time
from collections import deque

from aux_channels import AuxHandler, AuxSource
from rssi_wire import CH_GNSS, _restore_time

PORT = os.environ.get("GNSS_PORT", "/dev/ttyAMA0")
BAUD = int(os.environ.get("GNSS_BAUD", "9600"))

# ------------------------------------------------------------- wire format
#
#   magic u8 'N' | version u8 | flags u8 | quality u8 | seq u16 |
#   t_ms u32 (Pi clock, ms, truncated like sweeps) |
#   utc_s u32 (GNSS UTC, unix seconds, 0 = not known yet) |
#   lat i32 (1e-7 deg) | lon i32 (1e-7 deg) | alt i16 (dm, MSL) |
#   speed u16 (cm/s) | course u16 (0.01 deg) |
#   hdop u8 (0.1) | sats used u8 | sats in view u8          -> 31 bytes

MAGIC = 0x4E
VERSION = 1
MSG = struct.Struct("<BBBBHIIiihHHBBB")

F_VALID = 0x01
F_ALT = 0x02
F_SPEED = 0x04
F_COURSE = 0x08
F_HDOP = 0x10
F_GPS_TIME = 0x20       # t_ms is the GNSS epoch, not the arrival time

FIX_NAMES = {0: "no fix", 1: "GPS", 2: "DGPS/SBAS", 4: "RTK fixed",
             5: "RTK float", 6: "dead reckoning"}


def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def utc_seconds(fix) -> int | None:
    """The fix's GNSS time as unix seconds, if the date is known yet."""
    utc = getattr(fix, "utc", None)
    if not utc or "T" not in utc:
        return None                     # no RMC date yet
    try:
        return calendar.timegm(time.strptime(utc, "%Y-%m-%dT%H:%M:%SZ"))
    except ValueError:
        return None


def fix_time(fix, t_arrival: float) -> tuple[float, bool]:
    """(when this fix was taken on the Pi's clock, came-from-GPS?)."""
    u = utc_seconds(fix)
    # At 9600 baud GGA lands ~0.1-0.4 s after its epoch. Arriving inside that
    # window means the Pi's clock agrees with GPS, so the epoch itself is the
    # better timestamp. Outside it the Pi's clock is off, and mixing GPS time
    # into a Pi-clock timeline would misplace fixes against sweeps - so use
    # the arrival time, which is on the same (wrong) clock as the sweeps.
    if u is not None and 0.0 <= t_arrival - u < 0.6:
        return float(u), True
    return t_arrival, False


def encode_fix(fix, seq: int, t_fix: float, from_gps: bool = False) -> bytes:
    flags = 0
    if fix.valid:
        flags |= F_VALID
    if fix.alt_m is not None:
        flags |= F_ALT
    if fix.speed_kmh is not None:
        flags |= F_SPEED
    if fix.course_deg is not None:
        flags |= F_COURSE
    if fix.hdop is not None:
        flags |= F_HDOP
    if from_gps:
        flags |= F_GPS_TIME
    return MSG.pack(
        MAGIC, VERSION, flags, _clamp(int(fix.fix_quality or 0), 0, 255),
        seq & 0xFFFF,
        int(t_fix * 1000.0) & 0xFFFFFFFF,
        (utc_seconds(fix) or 0) & 0xFFFFFFFF,
        int(round((fix.lat or 0.0) * 1e7)), int(round((fix.lon or 0.0) * 1e7)),
        _clamp(int(round((fix.alt_m or 0.0) * 10)), -32768, 32767),
        _clamp(int(round((fix.speed_kmh or 0.0) / 3.6 * 100)), 0, 65535),
        _clamp(int(round((fix.course_deg or 0.0) * 100)), 0, 65535),
        _clamp(int(round((fix.hdop or 0.0) * 10)), 0, 255),
        _clamp(int(fix.sats_used or 0), 0, 255),
        _clamp(int(fix.sats_in_view or 0), 0, 255))


def decode_fix(blob: bytes, now: float | None = None) -> dict:
    if len(blob) < MSG.size or blob[0] != MAGIC or blob[1] != VERSION:
        raise ValueError("not a GNSS fix message")
    (_, _, fl, q, seq, t_ms, utc_s, lat, lon, alt, spd, crs, hdop, used,
     view) = MSG.unpack_from(blob)
    valid = bool(fl & F_VALID)
    return {
        "seq": seq, "t": _restore_time(t_ms, now), "utc_s": utc_s or None,
        "valid": valid, "quality": q, "fix_type": FIX_NAMES.get(q, f"q{q}"),
        "lat": lat / 1e7 if valid else None,
        "lon": lon / 1e7 if valid else None,
        "alt_m": alt / 10.0 if valid and fl & F_ALT else None,
        "speed_ms": spd / 100.0 if fl & F_SPEED else None,
        "course_deg": crs / 100.0 if fl & F_COURSE else None,
        "hdop": hdop / 10.0 if fl & F_HDOP else None,
        "sats_used": used, "sats_in_view": view,
        "gps_time": bool(fl & F_GPS_TIME),
    }


# --------------------------------------------------------- position track

class PositionTrack:
    """Recent fixes, and where you were at any moment between them.

    Sweeps come ~3 times a second, fixes once a second, so a sweep almost
    never lands on a fix. Between two fixes (no more than `max_gap` apart)
    the position is interpolated; otherwise the nearest fix is used if it is
    within `max_age`; otherwise there is no position, and the CSV cell is
    left empty rather than filled with somewhere you no longer are.
    """

    def __init__(self, keep_s: float = 120.0, max_gap: float = 3.0,
                 max_age: float = 1.5):
        self.keep_s = keep_s
        self.max_gap = max_gap
        self.max_age = max_age
        self._t: list[float] = []
        self._p: list[tuple] = []
        self._lock = threading.Lock()

    def add(self, t: float, lat: float, lon: float, alt: float | None) -> None:
        with self._lock:
            i = bisect.bisect(self._t, t)
            self._t.insert(i, t)
            self._p.insert(i, (lat, lon, alt))
            cut = bisect.bisect(self._t, self._t[-1] - self.keep_s)
            if cut:
                del self._t[:cut]
                del self._p[:cut]

    def at(self, t: float):
        """(lat, lon, alt_m) at time t, or None."""
        with self._lock:
            if not self._t:
                return None
            i = bisect.bisect(self._t, t)
            if 0 < i < len(self._t):
                t0, t1 = self._t[i - 1], self._t[i]
                if t1 - t0 <= self.max_gap:
                    a = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
                    p0, p1 = self._p[i - 1], self._p[i]
                    alt = None if p0[2] is None or p1[2] is None else \
                        p0[2] + a * (p1[2] - p0[2])
                    return (p0[0] + a * (p1[0] - p0[0]),
                            p0[1] + a * (p1[1] - p0[1]), alt)
            near = min((j for j in (i - 1, i) if 0 <= j < len(self._t)),
                       key=lambda j: abs(self._t[j] - t))
            if abs(self._t[near] - t) <= self.max_age:
                return self._p[near]
            return None

    def latest(self):
        with self._lock:
            return (self._t[-1], self._p[-1]) if self._t else None


# ================================================================ Pi side

class GnssSource(AuxSource):
    """Reads the HAT with max_m8q_gps.MaxM8Q and sends one packed fix per
    navigation epoch (1 Hz by default), valid or not - a 'no fix, 3 sats'
    message is how the base station knows the GPS is alive but still
    acquiring."""

    name = "gnss"
    chan = CH_GNSS
    keep_latest = True          # a stale fix is worse than a missing one

    def __init__(self, port: str | None = None, baud: int | None = None):
        self.port = port or os.environ.get("GNSS_PORT", PORT)
        self.baud = int(baud or os.environ.get("GNSS_BAUD", BAUD))
        self.gps = None
        self._it = None
        self.seq = 0
        self.track = PositionTrack()
        self.last = None

    def open(self) -> None:
        from max_m8q_gps import MaxM8Q      # Pi only; the laptop never needs it
        if not os.path.exists(self.port):
            raise FileNotFoundError(
                f"{self.port} does not exist - enable the UART (raspi-config > "
                f"Interface Options > Serial Port: login shell No, hardware "
                f"Yes) and reboot, or pass --gnss-port")
        self.gps = MaxM8Q(self.port, self.baud).open()
        self._it = self.gps.fixes()

    def read(self) -> bytes | None:
        if self._it is None:
            self.close()
            self.open()
        try:
            fix = next(self._it)
        except StopIteration:
            self._it = None
            return None
        except Exception:
            # NoDataError (5 s of silence), a serial error, a yanked cable:
            # drop the port; the runner logs it and read() reopens next time.
            self._it = None
            raise
        t_fix, from_gps = fix_time(fix, time.time())
        if fix.valid:
            self.track.add(t_fix, fix.lat, fix.lon, fix.alt_m)
        self.last = fix
        self.seq = (self.seq + 1) & 0xFFFF
        return encode_fix(fix, self.seq, t_fix, from_gps)

    def position_at(self, t: float):
        return self.track.at(t)

    def close(self) -> None:
        if self.gps is not None:
            try:
                self.gps.close()
            except Exception:
                pass
            self.gps = None


# =========================================================== laptop side

class GnssHandler(AuxHandler):
    name = "gnss"

    def __init__(self):
        self.track = PositionTrack()
        self.last: dict | None = None
        self.got = 0
        self.lost = 0
        self._seq = None

    def handle(self, blob: bytes, t_rx: float) -> str | None:
        try:
            d = decode_fix(blob)
        except ValueError as e:
            return f"bad GNSS message: {e}"
        if self._seq is not None:
            gap = (d["seq"] - self._seq) & 0xFFFF
            if 1 < gap < 1000:
                self.lost += gap - 1
        self._seq = d["seq"]
        self.got += 1
        self.last = d
        if d["valid"]:
            self.track.add(d["t"], d["lat"], d["lon"], d["alt_m"])
            alt = "" if d["alt_m"] is None else f"  alt {d['alt_m']:.1f} m"
            hd = "" if d["hdop"] is None else f"  HDOP {d['hdop']:.1f}"
            spd = "" if d["speed_ms"] is None else f"  {d['speed_ms']:.1f} m/s"
            clk = ""
            if d["utc_s"]:
                off = d["t"] - d["utc_s"] if not d["gps_time"] else 0.0
                clk = ("  Pi clock = GPS" if d["gps_time"] else
                       f"  Pi clock {off:+.1f}s vs GPS")
            return (f"{d['fix_type']}  {d['lat']:.6f}, {d['lon']:.6f}{alt}  "
                    f"sats {d['sats_used']}/{d['sats_in_view']}{hd}{spd}{clk}")
        return (f"no fix yet - sats {d['sats_used']} used, "
                f"{d['sats_in_view']} in view")

    def position_at(self, t: float):
        return self.track.at(t)


def distance_m(a, b) -> float:
    """Metres between two (lat, lon[, alt]) points. Equirectangular - plenty
    for the few hundred metres a survey covers."""
    lat0 = math.radians((a[0] + b[0]) / 2)
    dx = math.radians(b[1] - a[1]) * math.cos(lat0) * 6_371_000
    dy = math.radians(b[0] - a[0]) * 6_371_000
    return math.hypot(dx, dy)
