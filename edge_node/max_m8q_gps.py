#!/usr/bin/env python3
"""
max_m8q_gps.py - read GPS coordinates from a u-blox MAX-M8Q GNSS HAT on a Raspberry Pi 5.

The HAT streams NMEA-0183 over the Pi's GPIO UART (GPIO14/15) at 9600 baud.
On a Pi 5 that UART is /dev/ttyAMA0 -- there is no /dev/ttyS0 like on a Pi 3/4,
which is what the Waveshare wiki assumes.

One-time setup on the Pi:
    sudo raspi-config   -> Interface Options -> Serial Port
                           "login shell over serial?"   No
                           "serial port hardware?"      Yes
    (equivalent: add  dtparam=uart0=on  to /boot/firmware/config.txt)
    sudo reboot
    sudo apt install python3-serial
    sudo usermod -aG dialout $USER      # log out/in after, then no sudo needed

Usage:
    python3 max_m8q_gps.py                  # stream fixes, one line per second
    python3 max_m8q_gps.py --once           # wait for a fix, print "lat,lon", exit
    python3 max_m8q_gps.py --once --json    # same, as JSON
    python3 max_m8q_gps.py --json           # stream JSON lines
    python3 max_m8q_gps.py --raw            # dump raw NMEA (debugging)
    python3 max_m8q_gps.py --port /dev/ttyAMA0 --baud 9600

From your own code:
    from max_m8q_gps import MaxM8Q
    with MaxM8Q() as gps:
        fix = gps.wait_for_fix(timeout=120)
        if fix:
            print(fix.lat, fix.lon, fix.alt_m)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, replace
from typing import Iterator

try:
    import serial  # pyserial
except ImportError:
    sys.exit("pyserial isn't installed:  sudo apt install python3-serial   (or pip install pyserial in a venv)")

# Pi 5 GPIO UART first; the rest cover older Pis and the HAT's micro-USB port.
CANDIDATE_PORTS = [
    "/dev/ttyAMA0",  # Pi 5: UART0 on GPIO14/15
    "/dev/serial0",  # symlink to the primary UART
    "/dev/ttyS0",    # Pi 3/4 mini-UART
    "/dev/ttyACM0",  # module plugged in over USB (u-blox CDC-ACM)
    # /dev/ttyUSB0 deliberately NOT probed: on the drone that is the SiK radio,
    # and opening it at 9600 baud re-programs the port the spectrum link is
    # using. Pass --port /dev/ttyUSB0 if your GNSS really is there.
]
DEFAULT_BAUD = 9600
SILENCE_LIMIT_S = 5.0  # no valid NMEA for this long -> something's wrong

FIX_TYPES = {
    0: "no fix",
    1: "GPS",
    2: "DGPS/SBAS",
    4: "RTK fixed",
    5: "RTK float",
    6: "dead reckoning",
}
KNOTS_TO_KMH = 1.852

TROUBLESHOOTING = """
Things to check:
  - UART enabled? sudo raspi-config -> Interface Options -> Serial Port
    (login shell: No, hardware: Yes), then reboot. On a Pi 5 the GPIO UART is /dev/ttyAMA0.
  - "Permission denied"? sudo usermod -aG dialout $USER, then log out and back in.
  - Something else reading the port? If you installed gpsd: sudo systemctl stop gpsd.socket gpsd
  - HAT jumpers set so the module's UART goes to the Pi header (not the USB side).
  - Changed the module's baud rate at some point? Try --baud 38400 or --baud 115200.
"""


class NoDataError(RuntimeError):
    """Raised when the port can't be found or goes silent."""


# ----------------------------------------------------------------- NMEA helpers

def checksum_ok(sentence: str) -> bool:
    """Validate the *HH XOR checksum of an NMEA sentence."""
    if not sentence.startswith("$"):
        return False
    body, star, cs = sentence[1:].partition("*")
    if not star or len(cs) < 2:
        return False
    try:
        expected = int(cs[:2], 16)
    except ValueError:
        return False
    calc = 0
    for ch in body:
        calc ^= ord(ch)
    return calc == expected


def dm_to_deg(value: str, hemi: str) -> float | None:
    """NMEA (d)ddmm.mmmm + N/S/E/W  ->  signed decimal degrees."""
    if not value or not hemi:
        return None
    dot = value.find(".")
    if dot == -1:
        dot = len(value)
    degrees = int(value[: dot - 2])
    minutes = float(value[dot - 2:])
    dd = degrees + minutes / 60.0
    return -dd if hemi in ("S", "W") else dd


def _float(s: str) -> float | None:
    try:
        return float(s) if s else None
    except ValueError:
        return None


def _int(s: str) -> int | None:
    try:
        return int(s) if s else None
    except ValueError:
        return None


# ------------------------------------------------------------------- fix state

@dataclass
class Fix:
    lat: float | None = None        # decimal degrees, + north / - south
    lon: float | None = None        # decimal degrees, + east / - west
    alt_m: float | None = None      # altitude above mean sea level, metres
    fix_quality: int = 0            # GGA quality indicator (0 = no fix)
    fix_type: str = "no fix"
    sats_used: int = 0
    sats_in_view: int = 0           # summed across GPS/GLONASS/Galileo/BeiDou
    hdop: float | None = None
    speed_kmh: float | None = None
    course_deg: float | None = None  # true course over ground
    utc: str | None = None          # receiver time, ISO-8601 UTC

    @property
    def valid(self) -> bool:
        return self.fix_quality > 0 and self.lat is not None and self.lon is not None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["valid"] = self.valid
        return d


class NMEAParser:
    """Keeps a running Fix from RMC / GGA / GSV sentences (any talker: GP, GN, GL, GA, GB)."""

    def __init__(self) -> None:
        self.fix = Fix()
        self._date: str | None = None
        self._in_view: dict[str, int] = {}

    def feed(self, sentence: str) -> Fix | None:
        """Feed one checksum-valid sentence. Returns a snapshot after each GGA,
        i.e. once per navigation epoch (1 Hz by default on the M8)."""
        fields = sentence[1: sentence.index("*")].split(",")
        talker, kind = fields[0][:2], fields[0][2:]
        try:
            if kind == "RMC":
                self._rmc(fields)
            elif kind == "GSV":
                self._gsv(talker, fields)
            elif kind == "GGA":
                self._gga(fields)
                return replace(self.fix)
        except (ValueError, IndexError):
            pass  # malformed sentence that still passed checksum; skip it
        return None

    def _timestamp(self, hhmmss: str) -> str | None:
        if len(hhmmss) < 6:
            return None
        t = f"{hhmmss[0:2]}:{hhmmss[2:4]}:{hhmmss[4:6]}"
        return f"{self._date}T{t}Z" if self._date else f"{t}Z"

    def _rmc(self, f: list[str]) -> None:
        # $xxRMC,time,status,lat,N/S,lon,E/W,speed_kn,course,ddmmyy,...
        d = f[9]
        if len(d) == 6:
            self._date = f"20{d[4:6]}-{d[2:4]}-{d[0:2]}"
        if f[2] == "A":
            kn = _float(f[7])
            self.fix.speed_kmh = round(kn * KNOTS_TO_KMH, 2) if kn is not None else None
            self.fix.course_deg = _float(f[8])
        else:
            self.fix.speed_kmh = self.fix.course_deg = None

    def _gsv(self, talker: str, f: list[str]) -> None:
        # $xxGSV,num_msgs,msg_num,sats_in_view,...  (one set per constellation)
        n = _int(f[3])
        if n is not None:
            self._in_view[talker] = n

    def _gga(self, f: list[str]) -> None:
        # $xxGGA,time,lat,N/S,lon,E/W,quality,num_sats,hdop,alt,M,geoid_sep,M,...
        q = _int(f[6]) or 0
        fx = self.fix
        fx.fix_quality = q
        fx.fix_type = FIX_TYPES.get(q, f"quality {q}")
        fx.sats_used = _int(f[7]) or 0
        fx.hdop = _float(f[8])
        fx.utc = self._timestamp(f[1])
        fx.sats_in_view = sum(self._in_view.values())
        if q > 0:
            fx.lat = dm_to_deg(f[2], f[3])
            fx.lon = dm_to_deg(f[4], f[5])
            fx.alt_m = _float(f[9])
        else:
            fx.lat = fx.lon = fx.alt_m = None


# ---------------------------------------------------------------- serial layer

def _extract_sentence(raw: bytes) -> str | None:
    line = raw.decode("ascii", errors="ignore").strip()
    i = line.rfind("$")  # drop any UBX binary / partial line stuck on the front
    if i < 0:
        return None
    line = line[i:]
    return line if checksum_ok(line) else None


def find_port(baud: int = DEFAULT_BAUD, candidates: list[str] = CANDIDATE_PORTS,
              probe_s: float = 2.5) -> str:
    """Return the first candidate port that's producing valid NMEA."""
    problems, seen = [], set()
    for port in candidates:
        if not os.path.exists(port):
            continue
        real = os.path.realpath(port)
        if real in seen:
            continue
        seen.add(real)
        try:
            with serial.Serial(port, baud, timeout=0.5) as s:
                s.reset_input_buffer()
                end = time.monotonic() + probe_s
                while time.monotonic() < end:
                    if _extract_sentence(s.readline()):
                        return port
            problems.append(f"{port}: opened OK but no NMEA at {baud} baud")
        except (serial.SerialException, OSError) as e:
            problems.append(f"{port}: {e}")
    if not problems:
        problems.append("none of " + ", ".join(candidates) + " exist (UART not enabled?)")
    raise NoDataError("Couldn't find the GNSS HAT.\n  " + "\n  ".join(problems) + "\n" + TROUBLESHOOTING)


class MaxM8Q:
    def __init__(self, port: str | None = None, baud: int = DEFAULT_BAUD) -> None:
        self.port = port
        self.baud = baud
        self._ser: serial.Serial | None = None
        self._parser = NMEAParser()

    def open(self) -> "MaxM8Q":
        if self.port is None:
            self.port = find_port(self.baud)
        self._ser = serial.Serial(self.port, self.baud, timeout=1.0)
        self._ser.reset_input_buffer()
        return self

    def close(self) -> None:
        if self._ser is not None:
            self._ser.close()
            self._ser = None

    def __enter__(self) -> "MaxM8Q":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    def read_line(self) -> str | None:
        """One raw line from the port (unvalidated), or None on a 1 s timeout."""
        raw = self._ser.readline()
        return raw.decode("ascii", errors="replace").strip() if raw else None

    def read_sentence(self) -> str | None:
        """One checksum-valid NMEA sentence, or None (timeout / junk)."""
        return _extract_sentence(self._ser.readline())

    def fixes(self, timeout: float | None = None) -> Iterator[Fix]:
        """Yield a Fix snapshot every epoch (valid or not) until `timeout` seconds pass."""
        deadline = None if timeout is None else time.monotonic() + timeout
        last_data = time.monotonic()
        while deadline is None or time.monotonic() < deadline:
            s = self.read_sentence()
            now = time.monotonic()
            if s is None:
                if now - last_data > SILENCE_LIMIT_S:
                    raise NoDataError(
                        f"No valid NMEA from {self.port} for {SILENCE_LIMIT_S:.0f} s "
                        f"(wrong port/baud, or another process is reading it).\n" + TROUBLESHOOTING)
                continue
            last_data = now
            snap = self._parser.feed(s)
            if snap is not None:
                yield snap

    def wait_for_fix(self, timeout: float | None = None) -> Fix | None:
        """Block until a valid position arrives. Returns None on timeout."""
        for fix in self.fixes(timeout=timeout):
            if fix.valid:
                return fix
        return None


# ------------------------------------------------------------------------- CLI

def format_fix(fix: Fix) -> str:
    stamp = f"[{fix.utc or '--:--:--'}]"
    if not fix.valid:
        return f"{stamp} waiting for fix... sats used {fix.sats_used}, in view {fix.sats_in_view}"
    parts = [stamp, f"{fix.lat:.6f}, {fix.lon:.6f}"]
    if fix.alt_m is not None:
        parts.append(f"alt {fix.alt_m:.1f} m")
    parts.append(f"sats {fix.sats_used}/{fix.sats_in_view}")
    if fix.hdop is not None:
        parts.append(f"HDOP {fix.hdop:.2f}")
    if fix.speed_kmh is not None:
        parts.append(f"{fix.speed_kmh:.1f} km/h")
    parts.append(fix.fix_type)
    return "  ".join(parts)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Read position from a u-blox MAX-M8Q GNSS HAT.")
    ap.add_argument("--port", help="serial device (default: auto-detect; Pi 5 GPIO UART = /dev/ttyAMA0)")
    ap.add_argument("--baud", type=int, default=DEFAULT_BAUD, help="baud rate (default 9600)")
    ap.add_argument("--once", action="store_true", help="wait for one valid fix, print it, exit")
    ap.add_argument("--timeout", type=float, default=180, help="seconds to wait with --once (default 180)")
    ap.add_argument("--json", action="store_true", help="output JSON")
    ap.add_argument("--raw", action="store_true", help="print raw serial lines (debugging)")
    args = ap.parse_args(argv)

    tty = sys.stderr.isatty()
    try:
        with MaxM8Q(args.port, args.baud) as gps:
            print(f"Reading {gps.port} @ {gps.baud} baud (Ctrl+C to stop)", file=sys.stderr)

            if args.raw:
                while True:
                    line = gps.read_line()
                    if line:
                        print(line, flush=True)

            if args.once:
                for fix in gps.fixes(timeout=args.timeout):
                    if fix.valid:
                        if tty:
                            print(file=sys.stderr)
                        if args.json:
                            print(json.dumps(fix.to_dict()))
                        else:
                            print(f"{fix.lat:.6f},{fix.lon:.6f}")
                            print(f"https://maps.google.com/?q={fix.lat:.6f},{fix.lon:.6f}", file=sys.stderr)
                        return 0
                    print(("\r" if tty else "") + format_fix(fix), end="" if tty else "\n",
                          file=sys.stderr, flush=True)
                if tty:
                    print(file=sys.stderr)
                print(f"No fix within {args.timeout:g} s. The antenna needs a clear view of the sky "
                      f"(outside or right against a window); a cold start takes ~30 s outdoors.",
                      file=sys.stderr)
                return 1

            for fix in gps.fixes():
                print(json.dumps(fix.to_dict()) if args.json else format_fix(fix), flush=True)

    except NoDataError as e:
        print(e, file=sys.stderr)
        return 2
    except serial.SerialException as e:
        print(f"Serial error: {e}\n{TROUBLESHOOTING}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
