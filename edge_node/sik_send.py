#!/usr/bin/env python3
"""Send arbitrary data over the SiK link.

    python3 sik_send.py file report.csv           # reliable file transfer (default)
    python3 sik_send.py file big.bin --unreliable # fire and forget, no repair
    python3 sik_send.py text "pump 3 offline"     # one-off message
    echo hi | python3 sik_send.py stdin           # pipe anything in
    tail -f log.txt | python3 sik_send.py stdin --line-mode

Reliable mode sends the whole file, then asks the receiver what's missing and
resends only that. On a half-duplex link with ~150 ms turnaround, repairing in
batches at the end beats acknowledging every packet by a wide margin.
"""

import argparse
import hashlib
import json
import os
import struct
import sys
import time

from siklink import CHUNK, DONE, FILE_END, FILE_META, REPAIR, Link


def _progress(done: int, total: int, started: float, label: str = "") -> None:
    el = max(1e-9, time.monotonic() - started)
    pct = 100.0 * done / total if total else 100.0
    rate = done / el
    eta = (total - done) / rate if rate > 0 else 0
    sys.stdout.write(f"\r  {label}{done}/{total} ({pct:5.1f}%) "
                     f"{rate:6.0f} B/s  eta {eta:5.0f}s   ")
    sys.stdout.flush()


def send_file(link: Link, path: str, reliable: bool, rounds: int,
              wait: float, csize: int) -> int:
    data = open(path, "rb").read()
    digest = hashlib.sha256(data).hexdigest()
    chunks = [data[i:i + csize] for i in range(0, len(data), csize)] or [b""]

    meta = json.dumps({
        "name": os.path.basename(path),
        "size": len(data),
        "sha256": digest,
        "chunks": len(chunks),
        "csize": csize,
    }).encode()

    wire = len(data) + len(chunks) * (18 + 5)
    print(f"sending {path}")
    print(f"  {len(data)} bytes -> {len(chunks)} chunks of {csize}B "
          f"(~{wire} bytes on the wire)")
    print(f"  sha256 {digest[:16]}...")
    print(f"  est. {wire / link.bps:.0f}s at {link.bps:.0f} B/s\n")

    started = time.monotonic()

    # Metadata three times over - if it's lost the receiver can't do anything.
    for _ in range(3):
        link.send_msg(FILE_META, meta)

    for i, c in enumerate(chunks):
        link.send_msg(CHUNK, struct.pack("<I", i) + c)
        if i % 10 == 0 or i == len(chunks) - 1:
            _progress(i + 1, len(chunks), started, "chunk ")
    print()

    if not reliable:
        link.drain()
        print("\nsent (unreliable mode - no delivery confirmation)")
        return 0

    for rnd in range(1, rounds + 1):
        tag = rnd & 0xFF
        missing: list[int] = []
        got_reply = False

        # Ask "what did you miss?" and wait. Re-ask a few times: the question
        # or the answer can be lost, and either way we just hear nothing.
        for attempt in range(3):
            link.send_msg(FILE_END, bytes([tag]))
            link.drain()
            deadline = time.monotonic() + wait
            while time.monotonic() < deadline:
                for mtype, body in link.poll_raw(0.2):
                    if mtype == DONE and body and body[0] == tag:
                        el = time.monotonic() - started
                        print(f"\nreceiver confirmed complete in {el:.1f}s "
                              f"({len(data) / el:.0f} B/s effective)")
                        return 0
                    if mtype == REPAIR and len(body) >= 3 and body[0] == tag:
                        n = struct.unpack_from("<H", body, 1)[0]
                        if n:
                            missing.extend(struct.unpack_from(f"<{n}I", body, 3))
                        got_reply = True
                        # the list may span several messages; give them time
                        deadline = time.monotonic() + 2.0
                if got_reply and time.monotonic() >= deadline:
                    break
            if got_reply:
                break
            print(f"  round {rnd}: no reply (attempt {attempt + 1}/3)")

        if not got_reply:
            print(f"  round {rnd}: receiver silent, giving up on this round")
            continue

        missing = sorted(set(i for i in missing if i < len(chunks)))
        if not missing:
            continue
        print(f"  round {rnd}: resending {len(missing)} chunks "
              f"({100.0 * len(missing) / len(chunks):.1f}% of the file)")
        for j, i in enumerate(missing):
            link.send_msg(CHUNK, struct.pack("<I", i) + chunks[i])
            sys.stdout.write(f"\r  repair {j + 1}/{len(missing)}   ")
            sys.stdout.flush()
        link.drain()
        time.sleep(0.5)   # let the far end chew through them before we ask again
        print()

    print("\ngave up after %d repair rounds - receiver may have an incomplete file. "
          "Lower --bps or move the radios closer." % rounds)
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Send data over a SiK radio link")
    ap.add_argument("mode", choices=["file", "text", "stdin"])
    ap.add_argument("arg", nargs="?", help="file path, or the text to send")
    ap.add_argument("--port", default=None)
    ap.add_argument("--baud", type=int, default=57600)
    ap.add_argument("--bps", type=float, default=1800,
                    help="write budget in wire bytes/sec (default: 1800, "
                         "safe for a stock 64kbps link)")
    ap.add_argument("--chan", type=int, default=0, help="channel 0-255 for text/stdin")
    ap.add_argument("--csize", type=int, default=200, help="file chunk size (default: 200)")
    ap.add_argument("--unreliable", action="store_true",
                    help="skip the repair handshake (one-way links, live telemetry)")
    ap.add_argument("--rounds", type=int, default=12, help="max repair rounds (default: 12)")
    ap.add_argument("--wait", type=float, default=5.0,
                    help="seconds to wait for a repair reply (default: 5)")
    ap.add_argument("--line-mode", action="store_true",
                    help="stdin: send each line as its own message instead of buffering")
    args = ap.parse_args()

    if args.mode in ("file", "text") and not args.arg:
        ap.error(f"mode '{args.mode}' needs an argument")

    link = Link(args.port, args.baud, bps=args.bps)
    try:
        if args.mode == "file":
            return send_file(link, args.arg, not args.unreliable,
                             args.rounds, args.wait, args.csize)

        if args.mode == "text":
            link.send_text(args.arg, args.chan)
            link.drain()
            print(f"sent {len(args.arg)} chars on channel {args.chan}")
            return 0

        # stdin
        n = 0
        if args.line_mode:
            for line in sys.stdin:
                line = line.rstrip("\n")
                if line:
                    link.send_text(line, args.chan)
                    n += len(line)
                    sys.stderr.write(f"\r  sent {n} bytes")
        else:
            blob = sys.stdin.buffer.read()
            link.send(blob, args.chan)
            n = len(blob)
        link.drain()
        sys.stderr.write(f"\r  sent {n} bytes on channel {args.chan}\n")
        return 0
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130
    finally:
        link.close()


if __name__ == "__main__":
    sys.exit(main())
