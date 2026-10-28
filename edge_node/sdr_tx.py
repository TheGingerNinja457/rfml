#!/usr/bin/env python3
"""sdr_tx - sweep several bands on the Pi and push the spectra over the SiK link.

    python3 sdr_tx.py --source sim                         # no hardware
    python3 sdr_tx.py --uri ip:192.168.2.1                 # real board, default bands
    python3 sdr_tx.py --uri ip:192.168.2.1 --band ism24    # just 2.4 GHz
    python3 sdr_tx.py --source sim --port /dev/pts/4       # against vlink.py
    python3 sdr_tx.py --list-bands                         # what presets exist

BANDS.  Defaults live in bands.py (DEFAULT_BANDS): 2.4 GHz, 5.8 GHz and
915-928 MHz. Change them there, or per run with --band. The bands are swept
round-robin on the one radio; each sweep goes out tagged with its band id, and
the Pi sends the band table on the status channel so the base station can name
them without sharing a config file.

THE BANDWIDTH PROBLEM.  17 kbit/s is 2125 B/s. A 256-bin sweep is ~210 B of
wire, so the link carries ~10 sweeps a second in total - shared across all the
bands, the status messages, and the GNSS data that is coming. So:

  * --reserve-kbps (default 3) is held back for aux channels (GNSS). The
    spectrum never spends it, even while nothing is using it.
  * the rest is paced: the loop waits for budget BEFORE capturing a sweep,
    then sends it straight away. Nothing queues, so what arrives is always the
    newest measurement (a queued sweep is where you WERE, which is useless
    when you are walking an emitter down), and no band gets starved.

On real hardware the SDR, not the link, is usually the limit - see bands.py
for sweep times. The 5-second report tells you which one you are hitting.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time

from aux_channels import PI_SOURCES, AuxRunner, TestPatternSource, load
from linkmux import BURST_SECS, ByteBudget, LinkMux, pick_port  # noqa: F401
from rssi_wire import (CH_LOG, CH_STATUS, CH_SWEEP, CSV_HEADER,
                       DelayedSweepLog, encode_status, encode_sweep,
                       encoded_size)
from sdr_scan import (add_common_args, build_engines, build_source,
                      list_band_presets)

DEF_KBPS = 17.0
DEF_RESERVE_KBPS = 3.0


def band_table(bands, engines, skipped) -> list:
    """[id, name, f_start MHz, f_stop MHz, bins, gain, state] per band.

    Compact on purpose - this rides the status channel every few seconds.
    """
    live = {e.band_id: e for e in engines}
    out = []
    for bid, b in enumerate(bands):
        e = live.get(bid)
        out.append([bid, b.name, round(b.f_start_hz / 1e6, 3),
                    round(b.f_stop_hz / 1e6, 3), b.bins,
                    None if e is None or e.gain_db is None else round(e.gain_db),
                    "ok" if e else "skip"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--port", default=None,
                    help="serial port for the radio (default: autodetect; "
                         "on the Pi prefer /dev/serial/by-id/...)")
    ap.add_argument("--baud", type=int, default=57600)
    ap.add_argument("--kbps", type=float, default=DEF_KBPS,
                    help="TOTAL link budget in kbit/s, framing included, all "
                         "channels. Raising this past what the radio carries "
                         f"LOWERS goodput (default: {DEF_KBPS:g})")
    ap.add_argument("--reserve-kbps", type=float, default=DEF_RESERVE_KBPS,
                    help="part of --kbps held back for GNSS / aux channels. "
                         "The spectrum never uses it "
                         f"(default: {DEF_RESERVE_KBPS:g})")
    ap.add_argument("--interval", type=float, default=0.0,
                    help="minimum seconds per cycle (one sweep of every "
                         "band), 0 = flat out")
    ap.add_argument("--status-every", type=float, default=10.0,
                    help="seconds between status messages (default: 10)")
    ap.add_argument("--auto-gain", action="store_true",
                    help="adjust each band's RX gain to keep its strongest "
                         "signal off the ADC rails. Steps only between "
                         "sweeps, never within one")
    ap.add_argument("--aux", action="append", default=[], metavar="MOD:CLASS",
                    help="extra aux source to run, e.g. gnss_source:GnssSource "
                         "(in addition to PI_SOURCES in aux_channels.py)")
    ap.add_argument("--gnss-port", default=None,
                    help="GNSS HAT serial port (default: /dev/ttyAMA0, the "
                         "Pi 5 GPIO UART). Point it at fake_gnss.py's port to "
                         "bench-test without a sky view")
    ap.add_argument("--gnss-baud", type=int, default=None,
                    help="GNSS HAT baud (default: 9600)")
    ap.add_argument("--no-gnss", action="store_true",
                    help="do not run the GNSS HAT source")
    ap.add_argument("--aux-test", default=None, metavar="HZ[:BYTES]",
                    help="send a test pattern on the reserved budget, e.g. "
                         "5:32 - stands in for GNSS until it exists")
    ap.add_argument("--csv", default=None, help="also log every sweep locally")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.list_bands:
        list_band_presets()
        return 0

    total_bps = args.kbps * 1000.0 / 8.0
    reserve_bps = args.reserve_kbps * 1000.0 / 8.0
    if not 0 <= reserve_bps < total_bps:
        ap.error("--reserve-kbps must be at least 0 and below --kbps")
    spectrum_bps = total_bps - reserve_bps

    # systemd and `kill` send SIGTERM; treat it like Ctrl-C so the radio port
    # is closed cleanly and the summary still prints.
    def _term(_sig, _frm):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _term)

    src = build_source(args)
    engines, skipped, bands = build_engines(args, src)

    print("bands:", file=sys.stderr)
    cycle_worst = 0
    for eng in engines:
        worst = encoded_size(eng.nbins)
        cycle_worst += worst if eng.every == 1 else worst / eng.every
        g = "agc" if eng.gain_db is None else f"{eng.gain_db:g} dB"
        print(f"  [{eng.band_id}] {eng.name:<10} "
              f"{eng.plan.f_start/1e6:9.3f}-{eng.plan.f_stop/1e6:9.3f} MHz  "
              f"{eng.nbins:4d} bins x {eng.df_out/1e3:5.0f} kHz  "
              f"{eng.plan.n_tiles:2d} tiles  gain {g:>6}  "
              f"<= {worst} B/sweep", file=sys.stderr)
    for bid, b, why in skipped:
        print(f"  [{bid}] {b.name:<10} SKIPPED - {why}", file=sys.stderr)

    print(f"\nlink budget {args.kbps:.1f} kbit/s = {total_bps:.0f} B/s:\n"
          f"  spectrum + status  {args.kbps - args.reserve_kbps:5.1f} kbit/s "
          f"({spectrum_bps:.0f} B/s)\n"
          f"  reserved for aux   {args.reserve_kbps:5.1f} kbit/s "
          f"({reserve_bps:.0f} B/s)", file=sys.stderr)
    print(f"worst-case cycle {cycle_worst:.0f} B -> at least "
          f"{spectrum_bps / max(1, cycle_worst):.1f} cycles/s guaranteed "
          f"(typical sweeps compress to ~60% of that)", file=sys.stderr)
    if args.kbps > 17.0:
        print("WARNING: --kbps is above the 17 kbit/s this link is rated for. "
              "Expect CRC errors, not more data.", file=sys.stderr)

    port = pick_port(args.port)
    from siklink import Link
    # siklink gets a slightly larger budget than we intend to use, so that OUR
    # budgets are the thing deciding the rate. If siklink's throttle bound
    # first it would block inside send() and rebuild a backlog.
    link = Link(port, args.baud, bps=total_bps * 1.15)
    mux = LinkMux(link, total_bps, reserve_bps)
    stop = threading.Event()

    # -- aux sources: the GNSS HAT, plus anything from --aux ------------------
    if args.gnss_port:
        os.environ["GNSS_PORT"] = args.gnss_port
    if args.gnss_baud:
        os.environ["GNSS_BAUD"] = str(args.gnss_baud)
    runners = []
    specs = [s for s in PI_SOURCES
             if not (args.no_gnss and s.startswith("gnss_source"))]
    specs += list(args.aux)
    sources = []
    for spec in specs:
        try:
            sources.append(load(spec))
        except Exception as e:
            print(f"aux source {spec} failed to load: {e}", file=sys.stderr)
    if args.aux_test:
        hz, _, size = args.aux_test.partition(":")
        sources.append(TestPatternSource(float(hz), int(size or 32)))
    for s in sources:
        r = AuxRunner(s, mux, stop)
        r.start()
        runners.append(r)
        print(f"aux: {s.name} on channel {s.chan}", file=sys.stderr)
    if not runners and reserve_bps:
        print(f"aux: none running - {args.reserve_kbps:g} kbit/s held idle "
              f"for GNSS", file=sys.stderr)

    for r in runners:
        if r.src.name == "gnss":
            print(f"      port {r.src.port} @ {r.src.baud} baud "
                  f"(--gnss-port to change, --no-gnss to skip)", file=sys.stderr)
    print(f"\ntransmitting on {link.ser.port} @ {args.baud} baud "
          f"(Ctrl-C to stop)\n", file=sys.stderr)

    csv = open(args.csv, "a", buffering=1) if args.csv else None
    if csv and csv.tell() == 0:
        csv.write(CSV_HEADER + "\n")
    # Rows get the position the GNSS source says we were at when each sweep
    # was taken. This log never loses a sweep to the radio link, so it is the
    # one to build heat maps from.
    pos_sources = [s for s in sources if hasattr(s, "position_at")]

    def position_at(t):
        for s in pos_sources:
            p = s.position_at(t)
            if p:
                return p
        return None
    log = DelayedSweepLog(csv, position_at if pos_sources else None,
                          precision=2) if csv else None

    live = {e.band_id: e for e in engines}
    sent = {e.band_id: 0 for e in engines}
    sdr_drops = 0
    last_wire = {e.band_id: encoded_size(e.nbins) for e in engines}
    waited = 0.0
    cycle = 0
    t_start = time.monotonic()
    next_status = 0.0
    next_report = time.monotonic() + 5.0
    rep_sent = dict(sent)
    rep_t = t_start
    temps = {"now": None, "at_drop": None}

    def fmt_temps(d):
        if not d:
            return ""
        return "/".join(f"{d[k]:.0f}" for k in ("rf", "fpga") if d.get(k)
                        is not None) + " C"

    def send_status(now):
        # Read the board temperatures here, every status (10 s): cheap, and
        # it shows on the laptop, so heat-related hangs are visible from the
        # ground.
        try:
            temps["now"] = src.temps()
        except OSError:
            pass
        st = {
            "node": args.node_id,
            "up": round(now - t_start, 1),
            "src": src.name,
            "kbps": args.kbps,
            "res": args.reserve_kbps,
            "bands": band_table(bands, engines, skipped),
            # per band id, in band-table order; None for skipped bands
            "sent": [sent.get(b) for b in range(len(bands))],
            "ms": [round(live[b].last_sweep_secs * 1000) if b in live else None
                   for b in range(len(bands))],
            "clips": sum(e.clip_count for e in engines),
            "sdr_drops": sdr_drops,
            "temp": ([temps["now"].get("rf"), temps["now"].get("fpga")]
                     if temps["now"] else None),
            "aux": [r.stats() for r in runners],
        }
        # Charged to the spectrum budget but not waited on: it is small and
        # periodic, and making the loop wait for it would stall a sweep.
        mux.send_spectrum(encode_status(st), CH_STATUS)

    try:
        while not stop.is_set():
            loop_t0 = time.monotonic()
            for eng in engines:
                if cycle % eng.every:
                    continue
                bid = eng.band_id

                # Wait for room BEFORE capturing, so the sweep is fresh when
                # it goes out and no band starves the others.
                waited += mux.wait_spectrum(last_wire[bid], stop)
                if stop.is_set():
                    break

                try:
                    sw = eng.sweep()
                except OSError as e:
                    # The radio went away mid-sweep: broken pipe, reset,
                    # timeout. Usually USB power or the SDR rebooting. Tell
                    # the base station, wait for it to come back, carry on -
                    # the GNSS thread and the link keep running meanwhile.
                    sdr_drops += 1
                    temps["at_drop"] = temps["now"]
                    print(f"\nSDR connection lost ({e}) - reconnecting."
                          + (f" Last SDR temps {fmt_temps(temps['now'])} "
                             f"(AD936x/Zynq)." if temps["now"] else ""),
                          file=sys.stderr)
                    mux.send_spectrum(encode_status(
                        {"ev": "sdr_lost", "err": str(e)[:48]}), CH_LOG)
                    took = src.reconnect(stop)
                    print(f"SDR back after {took:.1f} s "
                          f"(drop #{sdr_drops})\n", file=sys.stderr)
                    mux.send_spectrum(encode_status(
                        {"ev": "sdr_back", "s": round(took, 1),
                         "n": sdr_drops}), CH_LOG)
                    continue
                if log:
                    log.add(sw)
                last_wire[bid] = mux.send_spectrum(encode_sweep(sw), CH_SWEEP)
                sent[bid] += 1

                if args.auto_gain and eng.auto_gain and eng.gain_db is not None:
                    want = eng.suggest_gain(sw)
                    if want is not None:
                        eng.gain_db = float(want)
                        mux.send_spectrum(encode_status(
                            {"ev": "gain", "band": eng.name, "db": want}),
                            CH_LOG)

                now = time.monotonic()
                if now >= next_status:
                    next_status = now + args.status_every
                    send_status(now)

                if not args.quiet and now >= next_report:
                    el = now - t_start
                    dt = now - rep_t
                    rates = mux.rates_kbps()
                    per = "  ".join(
                        f"{e.name} {(sent[e.band_id]-rep_sent[e.band_id])/dt:4.1f}/s"
                        for e in engines)
                    bound = ("link-bound" if waited > 0.2 * dt else "SDR-bound")
                    print(f"{el:6.0f}s  {per}  |  spectrum "
                          f"{rates['spectrum']:4.1f} + aux {rates['aux']:4.1f} "
                          f"= {rates['total']:4.1f} kbit/s  ({bound})"
                          + (f"  SDR {fmt_temps(temps['now'])}"
                             if temps["now"] else "")
                          + ("  [CLIP]" if sw.clipped else ""),
                          file=sys.stderr)
                    rep_sent = dict(sent)
                    rep_t = now
                    waited = 0.0
                    next_report = now + 5.0
            cycle += 1
            if args.interval:
                stop.wait(max(0.0, args.interval - (time.monotonic() - loop_t0)))

    except KeyboardInterrupt:
        print("\nstopping", file=sys.stderr)
    finally:
        stop.set()
        for r in runners:
            r.join(timeout=2.0)
        el = max(1e-9, time.monotonic() - t_start)
        rates = mux.rates_kbps()
        print("\nsweeps sent: " + ", ".join(
            f"{e.name} {sent[e.band_id]} ({sent[e.band_id]/el:.1f}/s)"
            for e in engines), file=sys.stderr)
        print(f"mean rate {rates['total']:.2f} kbit/s of {args.kbps:.1f} "
              f"allowed (spectrum {rates['spectrum']:.2f}, "
              f"aux {rates['aux']:.2f} of {args.reserve_kbps:.1f} reserved)",
              file=sys.stderr)
        for r in runners:
            st = r.stats()
            print(f"aux {r.src.name}: {st['sent']} sent, {st['superseded']} "
                  f"superseded by newer, {st['err']} errors", file=sys.stderr)
        print(f"frames sent {link.frames_sent}, "
              f"throttle waits {link.throttle_waits}", file=sys.stderr)
        if sdr_drops:
            print(f"SDR dropped out {sdr_drops} time(s)"
                  + (f" (last temps before a drop {fmt_temps(temps['at_drop'])}"
                     f", AD936x/Zynq)" if temps["at_drop"] else "")
                  + ". If that repeats, see README troubleshooting: USB "
                  "networking, heat, power - or use the SDR's Ethernet.",
                  file=sys.stderr)
        link.close()
        src.close()
        if log:
            log.close()
        if csv:
            csv.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
