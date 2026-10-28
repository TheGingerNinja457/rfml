#!/usr/bin/env python3
"""SiK telemetry radio - receiver / link analyser (run this on the laptop).

Parses the framed stream from sik_tx.py and reports live throughput, packet
rate, loss and corruption. Prints a summary on Ctrl-C.

    python sik_rx.py                      # autodetect port, 57600 baud
    python sik_rx.py --port COM5          # Windows
    python sik_rx.py --port /dev/ttyUSB0  # Linux
    python sik_rx.py --csv run1.csv       # also log per-second stats

Live line explained:
    goodput   payload bytes/sec (the number you actually care about)
    wire      payload + framing bytes/sec off the radio
    pkt/s     frames received per second
    loss      gaps in the sequence number, cumulative
    crc       frames that arrived corrupted
    lat       t_recv - t_send; only meaningful if both clocks are NTP-synced
"""

import argparse
import struct
import sys
import time
from collections import deque

import serial

from siklink import HDR, HDR_LEN, MAX_PAYLOAD, OVERHEAD, SYNC, crc16


class FrameParser:
    """Incremental parser: feed it bytes, get back complete frames."""

    def __init__(self) -> None:
        self.buf = bytearray()
        self.crc_errors = 0
        self.resyncs = 0

    def feed(self, data: bytes):
        self.buf += data
        while True:
            i = self.buf.find(SYNC)
            if i < 0:
                # keep one byte in case the sync word straddles a read boundary
                if len(self.buf) > 1:
                    del self.buf[:-1]
                return
            if i:
                del self.buf[:i]
                self.resyncs += 1

            if len(self.buf) < len(SYNC) + HDR_LEN:
                return
            seq, t_send, plen = HDR.unpack_from(self.buf, len(SYNC))

            if plen > MAX_PAYLOAD:
                del self.buf[:1]           # bogus length, that wasn't a real sync
                self.resyncs += 1
                continue

            total = len(SYNC) + HDR_LEN + plen + 2
            if len(self.buf) < total:
                return

            body = bytes(self.buf[len(SYNC):total - 2])
            got = struct.unpack_from("<H", self.buf, total - 2)[0]
            if crc16(body) == got:
                yield seq, t_send, plen, total
                del self.buf[:total]
            else:
                self.crc_errors += 1
                del self.buf[:1]           # false sync or corruption; hunt for the next one


def find_port() -> str:
    from serial.tools import list_ports

    ports = list(list_ports.comports())
    pref = [p.device for p in ports
            if any(k in p.device for k in ("ttyUSB", "ttyACM", "usbserial", "usbmodem"))]
    if pref:
        return pref[0]
    if ports:
        return ports[0].device
    sys.exit("No serial ports found. Plug in the radio, or pass --port.")


def main() -> None:
    ap = argparse.ArgumentParser(description="SiK radio throughput receiver")
    ap.add_argument("--port", default=None, help="serial device / COM port (default: autodetect)")
    ap.add_argument("--baud", type=int, default=57600,
                    help="must match the radio's SERIAL_SPEED (default: 57600)")
    ap.add_argument("--window", type=float, default=1.0,
                    help="stats refresh interval in seconds (default: 1.0)")
    ap.add_argument("--csv", default=None, help="append per-window stats to this CSV file")
    args = ap.parse_args()

    port = args.port or find_port()
    ser = serial.Serial(port, args.baud, timeout=0.1)
    parser = FrameParser()

    csv = None
    if args.csv:
        csv = open(args.csv, "a", buffering=1)
        if csv.tell() == 0:
            csv.write("t,goodput_Bps,wire_Bps,pkts_per_s,lost_total,crc_errors,latency_ms\n")

    print(f"RX on {port} @ {args.baud} baud - waiting for packets... (Ctrl-C to stop)\n")

    t0 = time.monotonic()
    win_start = t0
    win_payload = win_wire = win_pkts = 0
    tot_payload = tot_wire = tot_pkts = 0
    raw_bytes = 0
    lost = 0
    expected = None
    lat_samples: deque = deque(maxlen=200)
    first_pkt_t = None

    try:
        while True:
            chunk = ser.read(4096)
            now = time.monotonic()
            if chunk:
                raw_bytes += len(chunk)
                for seq, t_send, plen, total in parser.feed(chunk):
                    if first_pkt_t is None:
                        first_pkt_t = now
                    if expected is not None and seq != expected:
                        if seq > expected:
                            lost += seq - expected      # gap
                        else:
                            expected = seq              # sender restarted
                    expected = seq + 1
                    win_payload += plen
                    win_wire += total
                    win_pkts += 1
                    lat_samples.append((time.time() - t_send) * 1000.0)

            if now - win_start >= args.window:
                dt = now - win_start
                tot_payload += win_payload
                tot_wire += win_wire
                tot_pkts += win_pkts
                lat = sum(lat_samples) / len(lat_samples) if lat_samples else float("nan")
                gp = win_payload / dt
                wire = win_wire / dt
                pps = win_pkts / dt
                seen = tot_pkts + lost
                loss_pct = 100.0 * lost / seen if seen else 0.0

                sys.stdout.write(
                    f"\rgoodput {gp:7.0f} B/s ({gp * 8 / 1000:6.2f} kbps) | "
                    f"wire {wire:7.0f} B/s | {pps:6.1f} pkt/s | "
                    f"loss {lost} ({loss_pct:4.1f}%) | crc {parser.crc_errors} | "
                    f"lat {lat:6.1f} ms   ")
                sys.stdout.flush()

                if csv:
                    csv.write(f"{now - t0:.2f},{gp:.1f},{wire:.1f},{pps:.2f},"
                              f"{lost},{parser.crc_errors},{lat:.1f}\n")

                win_payload = win_wire = win_pkts = 0
                win_start = now
    except KeyboardInterrupt:
        pass
    finally:
        ser.close()
        if csv:
            csv.close()
        tot_payload += win_payload
        tot_wire += win_wire
        tot_pkts += win_pkts
        span = (time.monotonic() - first_pkt_t) if first_pkt_t else 0.0
        seen = tot_pkts + lost

        print("\n\n" + "=" * 58)
        print("  LINK SUMMARY")
        print("=" * 58)
        print(f"  duration          : {span:.1f} s")
        print(f"  packets received  : {tot_pkts}")
        print(f"  packets lost      : {lost}"
              + (f"  ({100.0 * lost / seen:.2f}%)" if seen else ""))
        print(f"  crc errors        : {parser.crc_errors}")
        print(f"  resyncs           : {parser.resyncs}")
        print(f"  raw bytes off port: {raw_bytes}")
        if span > 0:
            print(f"  goodput (payload) : {tot_payload / span:.0f} B/s "
                  f"({tot_payload / span * 8 / 1000:.2f} kbps)")
            print(f"  wire throughput   : {tot_wire / span:.0f} B/s "
                  f"({tot_wire / span * 8 / 1000:.2f} kbps)")
            print(f"  packet rate       : {tot_pkts / span:.1f} pkt/s")
        if not tot_pkts and raw_bytes:
            print("\n  Saw raw bytes but no valid frames - baud rate mismatch is the "
                  "usual cause. Check both ends match the radio's SERIAL_SPEED.")
        elif not raw_bytes:
            print("\n  Nothing received at all. Check the port, the antenna, and that both "
                  "radios show a solid green link LED.")
        print("=" * 58)


if __name__ == "__main__":
    main()
