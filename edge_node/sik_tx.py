#!/usr/bin/env python3
"""SiK telemetry radio - transmitter (run this on the Raspberry Pi).

Blasts sequence-numbered, CRC-checked frames out the serial port at a
controlled rate so the receiving end can measure real throughput and loss.

    python3 sik_tx.py                          # 64B payload @ 20 pkt/s
    python3 sik_tx.py --size 128 --rate 40
    python3 sik_tx.py --ramp                   # step the rate up until the link chokes
    python3 sik_tx.py --rate 0                 # unthrottled (fills the radio buffer, see notes)

Ctrl-C to stop.
"""

import argparse
import os
import sys
import time

import serial

from siklink import MAX_PAYLOAD, OVERHEAD, build_frame


def find_port() -> str:
    """Guess the radio's port. USB-serial adapters show up as ttyUSB*."""
    from serial.tools import list_ports

    candidates = [p.device for p in list_ports.comports()
                  if "ttyUSB" in p.device or "ttyACM" in p.device]
    if not candidates:
        sys.exit("No USB serial port found. Is the radio plugged in? "
                 "Check `ls /dev/ttyUSB*` and pass --port explicitly.")
    return candidates[0]


def main() -> None:
    ap = argparse.ArgumentParser(description="SiK radio throughput transmitter")
    ap.add_argument("--port", default=None,
                    help="serial device (default: autodetect /dev/ttyUSB*)")
    ap.add_argument("--baud", type=int, default=57600,
                    help="serial baud, must match the radio's SERIAL_SPEED (default: 57600)")
    ap.add_argument("--size", type=int, default=64,
                    help="payload bytes per packet, 8-1024 (default: 64)")
    ap.add_argument("--rate", type=float, default=20.0,
                    help="packets per second; 0 = as fast as the port accepts (default: 20)")
    ap.add_argument("--duration", type=float, default=0,
                    help="seconds to run, 0 = until Ctrl-C (default: 0)")
    ap.add_argument("--ramp", action="store_true",
                    help="ignore --rate; step 5,10,20,40,80,160 pkt/s for --step-time each")
    ap.add_argument("--step-time", type=float, default=10.0,
                    help="seconds per step in ramp mode (default: 10)")
    ap.add_argument("--max-buffered", type=int, default=2048,
                    help="skip a packet if the OS TX buffer exceeds this many bytes "
                         "(prevents multi-second latency build-up; 0 disables)")
    args = ap.parse_args()

    if not 8 <= args.size <= MAX_PAYLOAD:
        sys.exit(f"--size must be between 8 and {MAX_PAYLOAD}")

    port = args.port or find_port()
    ser = serial.Serial(port, args.baud, timeout=0, write_timeout=5)
    time.sleep(0.2)
    ser.reset_output_buffer()

    frame_len = args.size + OVERHEAD
    filler = bytes((i * 7 + 13) & 0xFF for i in range(args.size))  # deterministic, non-trivial

    schedule = ([(r, args.step_time) for r in (5, 10, 20, 40, 80, 160)]
                if args.ramp else [(args.rate, args.duration or float("inf"))])

    print(f"TX on {port} @ {args.baud} baud")
    print(f"frame = {args.size}B payload + {OVERHEAD}B overhead = {frame_len}B")
    if not args.ramp and args.rate:
        print(f"target = {args.rate:g} pkt/s = {args.rate * frame_len:.0f} B/s "
              f"= {args.rate * frame_len * 8 / 1000:.1f} kbps")
    print("Ctrl-C to stop.\n")

    seq = 0
    skipped = 0
    started = time.monotonic()

    try:
        for rate, step_len in schedule:
            interval = 1.0 / rate if rate > 0 else 0.0
            step_start = time.monotonic()
            next_due = step_start
            step_sent = 0
            if args.ramp:
                print(f"--- step: {rate:g} pkt/s "
                      f"({rate * frame_len * 8 / 1000:.1f} kbps) for {step_len:g}s")

            while time.monotonic() - step_start < step_len:
                # Back off if the kernel/driver is already sitting on a backlog.
                # Writing faster than the air rate just buys you latency, not throughput.
                if args.max_buffered and (ser.out_waiting or 0) > args.max_buffered:
                    skipped += 1
                    time.sleep(0.01)
                    continue

                ser.write(build_frame(seq, time.time(), filler))
                seq += 1
                step_sent += 1

                if interval:
                    next_due += interval
                    delay = next_due - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
                    elif delay < -1.0:
                        next_due = time.monotonic()  # fell far behind, resync

                if step_sent % max(1, int(rate or 200)) == 0:
                    elapsed = time.monotonic() - started
                    sys.stdout.write(
                        f"\rsent {seq} pkts | {seq * frame_len / elapsed:8.0f} B/s offered"
                        f" | skipped(backpressure) {skipped}   ")
                    sys.stdout.flush()
            if args.ramp:
                print()
    except KeyboardInterrupt:
        pass
    finally:
        elapsed = time.monotonic() - started
        ser.flush()
        ser.close()
        print(f"\n\nstopped after {elapsed:.1f}s")
        print(f"  packets offered : {seq}")
        print(f"  bytes offered   : {seq * frame_len} ({seq * frame_len / elapsed:.0f} B/s avg)")
        print(f"  skipped         : {skipped} (radio buffer was full)")


if __name__ == "__main__":
    main()
