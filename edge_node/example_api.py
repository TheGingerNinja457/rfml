#!/usr/bin/env python3
"""Using siklink from your own code.

Run one of these on each end:

    python3 example_api.py sensor      # Pi: send a reading every 2s
    python  example_api.py listen      # laptop: print whatever arrives
    python  example_api.py echo        # laptop: reply to everything (link test)
"""

import json
import random
import sys
import time

from siklink import Link

# Channels are just a byte you pick. Use them to keep unrelated streams apart
# on the same radio - the receiver gets the channel back with every message.
CH_TELEMETRY = 1
CH_STATUS = 2


def sensor(link: Link) -> None:
    """Send a JSON reading every couple of seconds."""
    while True:
        reading = {
            "t": round(time.time(), 2),
            "rssi": random.randint(-95, -60),
            "temp_c": round(20 + random.random() * 5, 2),
        }
        link.send(json.dumps(reading).encode(), chan=CH_TELEMETRY)
        print("sent", reading)
        time.sleep(2)


def listen(link: Link) -> None:
    """Receive anything, decoding JSON on the telemetry channel."""
    print("listening...")
    while True:
        for chan, blob in link.poll(timeout=0.5):
            if chan == CH_TELEMETRY:
                try:
                    print("telemetry:", json.loads(blob))
                except json.JSONDecodeError:
                    print("telemetry (unparsed):", blob[:60])
            else:
                print(f"ch{chan}:", blob[:60])


def echo(link: Link) -> None:
    """Bounce every message back - handy for measuring round-trip time."""
    print("echoing...")
    while True:
        for chan, blob in link.poll(timeout=0.5):
            print(f"ch{chan}: {len(blob)}B -> echoing back")
            link.send(blob, chan=chan)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "listen"

    # bps is the write budget in wire bytes/sec. 1800 is safe on a stock
    # 64 kbps link; raise it once you've raised AIR_SPEED on the radios.
    with Link(bps=1800) as link:
        try:
            {"sensor": sensor, "listen": listen, "echo": echo}[mode](link)
        except KeyError:
            sys.exit("usage: example_api.py [sensor|listen|echo]")
        except KeyboardInterrupt:
            print("\nstopped")
