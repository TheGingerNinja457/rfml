# siklink — arbitrary data over a SiK 915 MHz telemetry link

Send files, messages or piped streams between two machines over a pair of SiK v3 radios. Handles framing, CRC, rate limiting and lost-packet repair, because the raw link does none of that for you.

## Files

| File | Purpose |
|---|---|
| `siklink.py` | Core library. **Required on both machines** — everything imports it. |
| `sik_send.py` | CLI: send a file, a message, or stdin |
| `sik_recv.py` | CLI: receive files and messages |
| `example_api.py` | How to use the library from your own code |
| `sik_tx.py` / `sik_rx.py` | The throughput testers from before, unchanged |

Copy the whole folder to both the Pi and the laptop. Only dependency is `pyserial`.

```bash
pip3 install pyserial --break-system-packages    # Pi
pip install pyserial                             # laptop
```

## Quick start

Laptop — start the receiver first and leave it running:

```bash
python sik_recv.py
```

Pi:

```bash
python3 sik_send.py file readings.csv           # send a file, verified
python3 sik_send.py text "pump 3 offline"       # one-off message
cat data.json | python3 sik_send.py stdin       # pipe anything
tail -f sensor.log | python3 sik_send.py stdin --line-mode
```

Files land in `received/` and are only written after the sha256 matches. A mismatch gets written as `<name>.corrupt` so you can see what happened rather than silently getting bad data.

## How file transfer survives a lossy link

Your measured link drops packets under load, so a naive send loses data. Rather than acknowledging every packet — which would be brutal on a half-duplex link with ~150 ms turnaround — the sender:

1. Sends the metadata (name, size, sha256, chunk count) three times over
2. Sends every chunk once
3. Asks "what did you miss?"
4. Resends only the missing chunk indices
5. Repeats from 3 until the receiver confirms a hash match

Batching repairs at the end costs one round trip per round instead of one per packet. In testing against a link dropping 5% of bytes, a 25 KB file completed in three repair rounds — 29 chunks missing, then 8, then 2, then a clean hash.

Each round is tagged with a round number. Without that, a reply from the previous round gets mistaken for the current one and the two ends drift out of phase — worth knowing if you modify the protocol.

## Rate limiting — read this before raising `--bps`

`--bps` is a write budget in **wire bytes per second**, framing included. It defaults to **1800**, which sits just under the ~2.2 KB/s clean capacity measured on a stock 64 kbps link.

Setting it too high does not get you more throughput. The radio's internal buffer overflows, frames get corrupted mid-flight, and goodput *drops* — that's the collapse you saw at 40 pkt/s during the ramp test. If you've raised `AIR_SPEED` on the radios, re-run `sik_tx.py --ramp`, find the rate that holds at 0% loss, and set `--bps` to roughly that.

| Radio config | Sensible `--bps` |
|---|---|
| Stock: 64 kbps air, ECC on | 1800 |
| `AIR_SPEED=128`, ECC on | ~4000 |
| `AIR_SPEED=250`, ECC off, 115200 serial | ~10000 (measure it) |

At 1800 B/s a 100 KB file takes roughly 65 seconds before repair overhead. Plan accordingly — this is a telemetry link, not a file server.

## Using it from Python

```python
from siklink import Link

with Link(port='/dev/ttyUSB0', bps=1800) as link:
    link.send(b'anything you like', chan=1)       # fragments automatically
    link.send_text("hello", chan=2)

    for chan, blob in link.poll(timeout=1.0):     # blocks up to 1s
        print(chan, blob)
```

`send()` takes arbitrary bytes of any length and preserves message boundaries — the receiver hands back exactly the blob you passed in, reassembled. Channels are just a byte you choose, useful for keeping unrelated streams apart on the one radio.

One gotcha: if you're draining `poll_raw()` yourself, don't also call `poll()`. Two readers compete for the same bytes and one silently loses messages. Use `link.reassemble(body)` on `DATA` messages inside your own `poll_raw` loop instead — `sik_recv.py` shows the pattern.

`example_api.py` has runnable sensor / listener / echo examples.

## Options worth knowing

**sik_send.py**

```
--bps 1800        write budget, wire bytes/sec
--csize 200       file chunk size; bigger amortises the 23B/chunk overhead
                  better but loses more per dropped packet
--unreliable      skip the repair handshake entirely (one-way links, live
                  telemetry where stale data is worthless)
--rounds 12       max repair rounds before giving up
--wait 5          seconds to wait for a repair reply
--chan 0          channel for text/stdin
```

**sik_recv.py**

```
--outdir received  where files are written
--stdout           message payloads to stdout, logs to stderr
--chan N           only show one channel
```

Pipe a stream straight through:

```bash
# Pi
tail -f /var/log/sensors.log | python3 sik_send.py stdin --line-mode --chan 3
# laptop
python sik_recv.py --stdout --chan 3 | while read l; do echo "got: $l"; done
```

## Troubleshooting

**Nothing arrives.** Both radios need a solid green LED. Flashing means unpaired — matching `NETID`, `AIR_SPEED` and frequency settings on both ends.

**Frames arrive but CRC errors climb.** Almost always `--bps` set above what the link can carry. Halve it.

**`ModuleNotFoundError: No module named 'siklink'`.** `siklink.py` isn't in the same folder as the script you ran.

**`AttributeError: module 'serial' has no attribute 'Serial'`.** The unrelated `serial` PyPI package is installed and clobbering pyserial. `pip uninstall -y serial pyserial` then `pip install pyserial`.

**Transfer stalls in repair rounds.** The reverse path is failing — the receiver's repair requests aren't reaching the sender. Check the link is genuinely bidirectional, and that nothing else has the port open.
