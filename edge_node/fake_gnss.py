#!/usr/bin/env python3
"""fake_gnss - pretend to be the MAX-M8Q HAT, for testing indoors.

GPS needs a sky view; your desk does not have one. This opens a pseudo-
terminal and writes u-blox-style NMEA into it (RMC, VTG, GGA, GSA, GSV, GLL,
in the order a real M8 sends them), moving round a circle, so the whole
GNSS path - max_m8q_gps.py, gnss_source.py, the link, the base station, the
position columns in the logs - can be run end to end on the bench.

    python3 fake_gnss.py --link /tmp/GPS            # then, in another terminal:
    python3 sdr_tx.py --source sim --port /tmp/A --gnss-port /tmp/GPS
    python3 max_m8q_gps.py --port /tmp/GPS          # or just read it directly

Linux / Pi only (pseudo-terminals), like vlink.py.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time


def _nmea(body: str) -> bytes:
    c = 0
    for ch in body:
        c ^= ord(ch)
    return f"${body}*{c:02X}\r\n".encode("ascii")


def _dm(deg: float, lat: bool) -> tuple[str, str]:
    hemi = ("N" if deg >= 0 else "S") if lat else ("E" if deg >= 0 else "W")
    deg = abs(deg)
    d = int(deg)
    m = (deg - d) * 60.0
    return (f"{d:02d}{m:08.5f}" if lat else f"{d:03d}{m:08.5f}"), hemi


def epoch_sentences(t: float, lat: float | None, lon: float | None,
                    alt: float, speed_ms: float, course: float,
                    sats: int) -> list[bytes]:
    g = time.gmtime(t)
    hms = time.strftime("%H%M%S", g) + f".{int((t % 1) * 100):02d}"
    date = time.strftime("%d%m%y", g)
    fix = lat is not None
    if fix:
        la, ns = _dm(lat, True)
        lo, ew = _dm(lon, False)
    else:
        la = ns = lo = ew = ""
    kn = speed_ms * 3.6 / 1.852
    out = [
        _nmea(f"GNRMC,{hms},{'A' if fix else 'V'},{la},{ns},{lo},{ew},"
              f"{kn:.3f},{course:.2f},{date},,,{'A' if fix else 'N'}"),
        _nmea(f"GNVTG,{course:.2f},T,,M,{kn:.3f},N,{speed_ms*3.6:.3f},K,"
              f"{'A' if fix else 'N'}"),
        _nmea(f"GNGGA,{hms},{la},{ns},{lo},{ew},{1 if fix else 0},"
              f"{sats:02d},{1.1 if fix else 99.99},"
              f"{alt:.1f},M,28.1,M,,"),
        _nmea(f"GNGSA,A,{3 if fix else 1},,,,,,,,,,,,,2.0,1.1,1.7"),
    ]
    # GSV: sats in view, four per sentence
    view = sats + 3
    n_msgs = max(1, math.ceil(view / 4))
    for i in range(n_msgs):
        sv = ",".join(f"{prn:02d},45,{(prn * 37) % 360:03d},40"
                      for prn in range(1 + i * 4, min(view, (i + 1) * 4) + 1))
        out.append(_nmea(f"GPGSV,{n_msgs},{i + 1},{view:02d},{sv}"))
    out.append(_nmea(f"GNGLL,{la},{ns},{lo},{ew},{hms},{'A' if fix else 'V'},"
                     f"{'A' if fix else 'N'}"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--link", default="/tmp/GPS",
                    help="symlink to create for the fake port (default /tmp/GPS)")
    ap.add_argument("--lat", type=float, default=-36.8485)
    ap.add_argument("--lon", type=float, default=174.7633)
    ap.add_argument("--alt", type=float, default=40.0, help="metres, MSL")
    ap.add_argument("--radius", type=float, default=60.0,
                    help="circle radius in metres (default 60)")
    ap.add_argument("--speed", type=float, default=6.0,
                    help="ground speed in m/s (default 6)")
    ap.add_argument("--rate", type=float, default=1.0,
                    help="fixes per second (default 1, the M8Q default)")
    ap.add_argument("--no-fix-secs", type=float, default=3.0,
                    help="seconds of 'no fix' at start, like a cold start")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    import pty      # here, not at the top, so epoch_sentences() imports on Windows
    import tty
    master, slave = pty.openpty()
    tty.setraw(slave)
    # Nobody reading yet? Drop sentences rather than block when the pty
    # buffer fills - a real receiver doesn't wait for you either.
    os.set_blocking(master, False)
    path = os.ttyname(slave)
    if os.path.islink(args.link) or os.path.exists(args.link):
        os.remove(args.link)
    os.symlink(path, args.link)
    print(f"fake GNSS on {args.link} -> {path}  ({args.rate:g} Hz, circle of "
          f"{args.radius:g} m at {args.speed:g} m/s). Ctrl-C to stop.",
          file=sys.stderr)

    t0 = time.time()
    period = 1.0 / args.rate
    next_t = math.ceil(t0 / period) * period       # epochs on the tick, like a real receiver
    try:
        while True:
            time.sleep(max(0.0, next_t - time.time()) + 0.12)   # ~GGA latency
            el = next_t - t0
            fix = el >= args.no_fix_secs
            ang = args.speed * el / args.radius
            north = args.radius * math.sin(ang)
            east = args.radius * (1 - math.cos(ang))
            lat = args.lat + north / 111_320.0
            lon = args.lon + east / (111_320.0 * math.cos(math.radians(args.lat)))
            course = (math.degrees(math.atan2(math.sin(ang), math.cos(ang))) ) % 360
            sats = 9 if fix else 2
            for s in epoch_sentences(next_t, lat if fix else None,
                                     lon if fix else None, args.alt,
                                     args.speed if fix else 0.0, course, sats):
                try:
                    os.write(master, s)
                except (BlockingIOError, OSError):
                    pass
            if not args.quiet:
                state = f"{lat:.6f},{lon:.6f}" if fix else "no fix"
                print(f"\r  {time.strftime('%H:%M:%S', time.gmtime(next_t))}Z "
                      f"{state}   ", end="", file=sys.stderr, flush=True)
            next_t += period
    except KeyboardInterrupt:
        pass
    finally:
        try:
            os.remove(args.link)
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
