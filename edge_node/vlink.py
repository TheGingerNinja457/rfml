#!/usr/bin/env python3
"""vlink - a fake pair of SiK radios made from two pseudo-terminals.

Lets you run sdr_tx.py and base_rx.py on one machine, against each other, with
no hardware at all. It bridges two PTYs and can throw away or corrupt bytes on
the way through, so you can prove the CRC layer and the loss statistics work
before you are standing in a field wondering why nothing arrives.

    python3 vlink.py                        # clean link
    python3 vlink.py --loss 5 --corrupt 0.5 # 5% of chunks dropped, 0.5% bytes flipped
    python3 vlink.py --rate 7000            # throttle to 7000 B/s each way

It prints the two device paths on startup:

    A (Pi side)   : /dev/pts/4
    B (base side) : /dev/pts/6

Then in two other terminals:

    python3 sdr_tx.py  --source sim --port /dev/pts/4
    python3 base_rx.py --port /dev/pts/6

The emulated radios add a fixed one-way latency because the real ones do, and
a chunk of the protocol design exists to cope with it.
"""

from __future__ import annotations

import argparse
import os
import pty
import random
import select
import sys
import threading
import time


class Pipe:
    """One direction of the link, with its own impairments and delay queue."""

    def __init__(self, name, src_fd, dst_fd, loss_pct, corrupt_pct,
                 rate_bps, latency_s, rng):
        self.name = name
        self.src_fd = src_fd
        self.dst_fd = dst_fd
        self.loss = loss_pct / 100.0
        self.corrupt = corrupt_pct / 100.0
        self.rate = rate_bps
        self.latency = latency_s
        self.rng = rng

        self.queue: list[tuple[float, bytes]] = []
        self.bytes_in = 0
        self.bytes_out = 0
        self.bytes_dropped = 0
        self.bits_flipped = 0
        self._tokens = 0.0
        self._last = time.monotonic()

    def ingest(self, data: bytes) -> None:
        self.bytes_in += len(data)

        # Drop in bursts rather than per byte. Real radio loss is bursty - a
        # whole frame goes, not a scattering of bytes - and per-byte dropping
        # would flatter the CRC layer by turning every loss into a
        # resynchronisation it handles trivially.
        out = bytearray()
        i = 0
        while i < len(data):
            n = min(len(data) - i, self.rng.randint(8, 64))
            if self.loss and self.rng.random() < self.loss:
                self.bytes_dropped += n
            else:
                out += data[i:i + n]
            i += n

        if self.corrupt and out:
            for j in range(len(out)):
                if self.rng.random() < self.corrupt:
                    out[j] ^= 1 << self.rng.randrange(8)
                    self.bits_flipped += 1

        if out:
            self.queue.append((time.monotonic() + self.latency, bytes(out)))

    def flush(self) -> None:
        """Emit whatever has served its latency, within the rate budget."""
        now = time.monotonic()
        if self.rate > 0:
            self._tokens = min(self.rate,
                               self._tokens + (now - self._last) * self.rate)
        self._last = now

        while self.queue and self.queue[0][0] <= now:
            due, blob = self.queue[0]
            if self.rate > 0:
                allow = int(min(len(blob), self._tokens))
                if allow <= 0:
                    return
                self._tokens -= allow
            else:
                allow = len(blob)

            try:
                os.write(self.dst_fd, blob[:allow])
                self.bytes_out += allow
            except OSError:
                self.queue.pop(0)
                continue

            if allow < len(blob):
                self.queue[0] = (due, blob[allow:])
                return
            self.queue.pop(0)

    def stats(self) -> str:
        drop = 100.0 * self.bytes_dropped / max(1, self.bytes_in)
        return (f"{self.name}: {self.bytes_in:>8} B in, {self.bytes_out:>8} B out, "
                f"{self.bytes_dropped:>6} B dropped ({drop:4.1f}%), "
                f"{self.bits_flipped:>5} bits flipped")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--loss", type=float, default=0.0,
                    help="percent of byte-runs discarded (default: 0)")
    ap.add_argument("--corrupt", type=float, default=0.0,
                    help="percent of surviving bytes with a bit flipped")
    ap.add_argument("--rate", type=float, default=0.0,
                    help="bytes/sec cap each way, 0 = unlimited")
    ap.add_argument("--latency", type=float, default=0.05,
                    help="one-way delay in seconds (default: 0.05)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--link-a", default=None,
                    help="symlink to create for the Pi side, e.g. /tmp/radioA")
    ap.add_argument("--link-b", default=None,
                    help="symlink to create for the base side")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    a_master, a_slave = pty.openpty()
    b_master, b_slave = pty.openpty()
    a_path, b_path = os.ttyname(a_slave), os.ttyname(b_slave)

    # Raw mode, or the tty line discipline will "helpfully" mangle binary -
    # \r becoming \n mid-frame is a memorable way to lose an afternoon.
    import termios
    import tty
    for fd in (a_master, a_slave, b_master, b_slave):
        try:
            tty.setraw(fd, termios.TCSANOW)
        except termios.error:
            pass

    for target, path in ((args.link_a, a_path), (args.link_b, b_path)):
        if target:
            try:
                if os.path.islink(target) or os.path.exists(target):
                    os.remove(target)
                os.symlink(path, target)
            except OSError as e:
                print(f"could not create symlink {target}: {e}", file=sys.stderr)

    a2b = Pipe("A->B", a_master, b_master, args.loss, args.corrupt,
               args.rate, args.latency, rng)
    b2a = Pipe("B->A", b_master, a_master, args.loss, args.corrupt,
               args.rate, args.latency, rng)

    print(f"A (Pi side)   : {args.link_a or a_path}")
    print(f"B (base side) : {args.link_b or b_path}")
    print(f"loss {args.loss}%  corrupt {args.corrupt}%  "
          f"rate {args.rate or 'unlimited'}  latency {args.latency*1000:.0f} ms")
    print("\n  python3 sdr_tx.py  --source sim --port "
          f"{args.link_a or a_path}")
    print(f"  python3 base_rx.py --port {args.link_b or b_path}\n")
    print("Ctrl-C to stop.")

    stop = threading.Event()

    def report():
        while not stop.wait(10.0):
            if not args.quiet:
                print(f"  {a2b.stats()}\n  {b2a.stats()}", file=sys.stderr)

    threading.Thread(target=report, daemon=True).start()

    try:
        while True:
            r, _, _ = select.select([a_master, b_master], [], [], 0.01)
            if a_master in r:
                try:
                    a2b.ingest(os.read(a_master, 4096))
                except OSError:
                    pass
            if b_master in r:
                try:
                    b2a.ingest(os.read(b_master, 4096))
                except OSError:
                    pass
            a2b.flush()
            b2a.flush()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        print(f"\n  {a2b.stats()}\n  {b2a.stats()}")
        for fd in (a_master, a_slave, b_master, b_slave):
            try:
                os.close(fd)
            except OSError:
                pass
        for target in (args.link_a, args.link_b):
            if target and os.path.islink(target):
                try:
                    os.remove(target)
                except OSError:
                    pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
