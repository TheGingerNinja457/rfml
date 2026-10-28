#!/usr/bin/env python3
"""Receive whatever the other radio is sending.

    python sik_recv.py                      # print messages, save any files that arrive
    python sik_recv.py --outdir received    # where files land
    python sik_recv.py --stdout             # pipe message data to stdout
    python sik_recv.py --stdout > out.bin

Handles both message traffic and file transfers on the same link. When a file
arrives it replies with the list of missing chunks so the sender can repair,
then verifies the sha256 before writing anything out.
"""

import argparse
import hashlib
import json
import os
import struct
import sys
import time

from siklink import CHUNK, DATA, DONE, FILE_END, FILE_META, REPAIR, Link

IDX_PER_MSG = 48        # keeps a REPAIR message inside one small frame


class FileRx:
    """Collects chunks for one incoming file."""

    def __init__(self, meta: dict, outdir: str) -> None:
        self.name = os.path.basename(meta["name"]) or "received.bin"
        self.size = int(meta["size"])
        self.sha = meta["sha256"]
        self.total = int(meta["chunks"])
        self.outdir = outdir
        self.parts: dict[int, bytes] = {}
        self.started = time.monotonic()
        self.done = False
        print(f"\nincoming: {self.name}  {self.size} bytes in {self.total} chunks")

    def add(self, idx: int, data: bytes) -> None:
        if idx not in self.parts:
            self.parts[idx] = data
            n = len(self.parts)
            if n % 10 == 0 or n == self.total:
                el = max(1e-9, time.monotonic() - self.started)
                sys.stdout.write(f"\r  {n}/{self.total} chunks "
                                 f"({100.0 * n / self.total:5.1f}%)  "
                                 f"{sum(len(v) for v in self.parts.values()) / el:6.0f} B/s   ")
                sys.stdout.flush()

    def missing(self) -> list[int]:
        return [i for i in range(self.total) if i not in self.parts]

    def finish(self) -> bool:
        blob = b"".join(self.parts[i] for i in range(self.total))
        ok = hashlib.sha256(blob).hexdigest() == self.sha
        os.makedirs(self.outdir, exist_ok=True)
        path = os.path.join(self.outdir, self.name if ok else self.name + ".corrupt")
        with open(path, "wb") as f:
            f.write(blob)
        el = time.monotonic() - self.started
        print(f"\n  {'OK  ' if ok else 'BAD '} sha256 {'matches' if ok else 'MISMATCH'}")
        print(f"  wrote {path} ({len(blob)} bytes, {el:.1f}s, {len(blob) / el:.0f} B/s)")
        self.done = True
        return ok


def main() -> int:
    ap = argparse.ArgumentParser(description="Receive data over a SiK radio link")
    ap.add_argument("--port", default=None)
    ap.add_argument("--baud", type=int, default=57600)
    ap.add_argument("--bps", type=float, default=1800,
                    help="write budget for repair replies (default: 1800)")
    ap.add_argument("--outdir", default="received", help="where incoming files are written")
    ap.add_argument("--stdout", action="store_true",
                    help="write message payloads to stdout (logs go to stderr)")
    ap.add_argument("--chan", type=int, default=None, help="only show this channel")
    args = ap.parse_args()

    log = sys.stderr if args.stdout else sys.stdout
    link = Link(args.port, args.baud, bps=args.bps)
    cur: FileRx | None = None
    linger_until = 0.0

    print(f"listening on {link.ser.port} @ {args.baud} baud  (Ctrl-C to stop)", file=log)

    try:
        while True:
            for mtype, body in link.poll_raw(0.1):

                if mtype == DATA:
                    got = link.reassemble(body)
                    if got is None:
                        continue
                    chan, blob = got
                    if args.chan is not None and chan != args.chan:
                        continue
                    if args.stdout:
                        sys.stdout.buffer.write(blob)
                        sys.stdout.buffer.flush()
                    else:
                        try:
                            text = blob.decode()
                        except UnicodeDecodeError:
                            text = repr(blob)
                        print(f"[{time.strftime('%H:%M:%S')}] ch{chan}: {text}", file=log)

                elif mtype == FILE_META:
                    try:
                        meta = json.loads(body.decode())
                    except Exception:
                        continue
                    if cur is None or cur.name != os.path.basename(meta["name"]) or cur.done:
                        cur = FileRx(meta, args.outdir)

                elif mtype == CHUNK and cur and not cur.done and len(body) >= 4:
                    idx = struct.unpack_from("<I", body)[0]
                    cur.add(idx, body[4:])

                elif mtype == FILE_END and cur:
                    tag = body[0] if body else 0
                    if cur.done:
                        link.send_msg(DONE, bytes([tag]))   # sender missed our DONE
                        link.drain()
                        continue
                    miss = cur.missing()
                    if not miss:
                        cur.finish()
                        for _ in range(3):
                            link.send_msg(DONE, bytes([tag]))
                        link.drain()
                        linger_until = time.monotonic() + 10.0
                    else:
                        print(f"\n  {len(miss)} chunks missing, requesting resend", file=log)
                        for i in range(0, len(miss), IDX_PER_MSG):
                            part = miss[i:i + IDX_PER_MSG]
                            link.send_msg(REPAIR,
                                          bytes([tag]) + struct.pack("<H", len(part))
                                          + struct.pack(f"<{len(part)}I", *part))
                        link.drain()

            if linger_until and time.monotonic() > linger_until:
                linger_until = 0.0

    except KeyboardInterrupt:
        pass
    finally:
        if cur and not cur.done and cur.parts:
            print(f"\n\nincomplete: {len(cur.parts)}/{cur.total} chunks of {cur.name}", file=log)
        print(f"\nframes received: {link.frames_recv}  "
              f"crc errors: {link.parser.crc_errors}  resyncs: {link.parser.resyncs}", file=log)
        link.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
