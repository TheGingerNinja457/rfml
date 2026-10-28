"""siklink - arbitrary data over a SiK telemetry radio link.

Two layers.

FRAME LAYER (unchanged from the throughput test, so sik_tx/sik_rx still work):

    AA 55 | seq:uint32 | t_send:float64 | len:uint16 | payload[len] | crc16:uint16

CRC16-CCITT over seq..payload. 18 bytes of overhead per frame.

MESSAGE LAYER lives inside the payload; first byte is the type:

    0x01 DATA      [chan:u8][flags:u8][bytes]   flags bit0 = more fragments follow
    0x02 FILE_META [json]
    0x03 CHUNK     [idx:u32][bytes]
    0x04 FILE_END  [round:u8]
    0x05 REPAIR    [round:u8][count:u16][idx:u32 * count]
    0x06 DONE      [round:u8]

The round byte on FILE_END/REPAIR/DONE matters: without it, a repair reply
from the previous round can be mistaken for the current one and the two ends
drift out of phase. The sender ignores any reply whose round doesn't match,
which also makes duplicate FILE_ENDs harmless.

The link is lossy, half-duplex and slow, so Link rate-limits everything it
writes. Overdriving a SiK radio does not increase throughput - it overflows
the radio's internal buffer and corrupts frames mid-flight.
"""

import struct
import sys
import time

import serial

# ---------------------------------------------------------------- frame layer

SYNC = b"\xAA\x55"
HDR = struct.Struct("<IdH")             # seq, t_send, payload length
HDR_LEN = HDR.size                      # 14
OVERHEAD = len(SYNC) + HDR_LEN + 2      # 18
MAX_PAYLOAD = 1024

# message types
DATA, FILE_META, CHUNK, FILE_END, REPAIR, DONE = 0x01, 0x02, 0x03, 0x04, 0x05, 0x06


def crc16(data: bytes, crc: int = 0xFFFF) -> int:
    """CRC16-CCITT (poly 0x1021, init 0xFFFF)."""
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def build_frame(seq: int, t_send: float, payload: bytes) -> bytes:
    body = HDR.pack(seq & 0xFFFFFFFF, t_send, len(payload)) + payload
    return SYNC + body + struct.pack("<H", crc16(body))


class FrameParser:
    """Incremental parser. Feed it bytes, iterate complete frames back out."""

    def __init__(self) -> None:
        self.buf = bytearray()
        self.crc_errors = 0
        self.resyncs = 0

    def feed(self, data: bytes):
        self.buf += data
        while True:
            i = self.buf.find(SYNC)
            if i < 0:
                if len(self.buf) > 1:
                    del self.buf[:-1]   # sync word may straddle the read boundary
                return
            if i:
                del self.buf[:i]
                self.resyncs += 1

            if len(self.buf) < len(SYNC) + HDR_LEN:
                return
            seq, t_send, plen = HDR.unpack_from(self.buf, len(SYNC))

            if plen > MAX_PAYLOAD:
                del self.buf[:1]
                self.resyncs += 1
                continue

            total = len(SYNC) + HDR_LEN + plen + 2
            if len(self.buf) < total:
                return

            body = bytes(self.buf[len(SYNC):total - 2])
            got = struct.unpack_from("<H", self.buf, total - 2)[0]
            if crc16(body) == got:
                yield seq, t_send, bytes(self.buf[len(SYNC) + HDR_LEN:total - 2])
                del self.buf[:total]
            else:
                self.crc_errors += 1
                del self.buf[:1]


# ------------------------------------------------------------------ port help

def find_port(explicit: str | None = None) -> str:
    """Return an explicit port, or guess at a USB serial adapter."""
    if explicit:
        return explicit
    from serial.tools import list_ports

    ports = list(list_ports.comports())
    pref = [p.device for p in ports
            if any(k in p.device for k in ("ttyUSB", "ttyACM", "usbserial", "usbmodem"))]
    if pref:
        return pref[0]
    if ports:
        return ports[0].device
    sys.exit("No serial port found. Plug the radio in, or pass --port.")


# ------------------------------------------------------------------ link layer

class Link:
    """A rate-limited, framed, bidirectional pipe over the radio.

        link = Link(port='/dev/ttyUSB0', bps=1800)
        link.send(b'hello world')
        for chan, blob in link.poll(timeout=1.0):
            print(chan, blob)

    `bps` is the write budget in *wire* bytes per second, framing included.
    Set it at or just under the link's measured clean capacity. Too high and
    you get corruption rather than throughput; see the README.
    """

    def __init__(self, port=None, baud=57600, bps=1800, max_buffered=2048,
                 max_payload=220):
        self.ser = serial.Serial(find_port(port), baud, timeout=0.05, write_timeout=10)
        self.bps = float(bps)
        self.max_buffered = max_buffered
        self.max_payload = min(max_payload, MAX_PAYLOAD)
        self.parser = FrameParser()

        self._seq = 0
        self._tokens = 0.0
        self._last = time.monotonic()
        self._partial: dict[int, bytearray] = {}   # per-channel reassembly

        self.bytes_sent = 0
        self.frames_sent = 0
        self.frames_recv = 0
        self.throttle_waits = 0

        time.sleep(0.2)
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()

    # -- writing ------------------------------------------------------------

    def _throttle(self, nbytes: int) -> None:
        """Token bucket over wire bytes, plus a check on the OS TX backlog."""
        while True:
            now = time.monotonic()
            self._tokens = min(self.bps, self._tokens + (now - self._last) * self.bps)
            self._last = now
            if self._tokens >= nbytes:
                self._tokens -= nbytes
                break
            self.throttle_waits += 1
            time.sleep(min(0.25, (nbytes - self._tokens) / self.bps))

        # Don't pile onto a backlog the driver hasn't drained yet.
        deadline = time.monotonic() + 5.0
        while (self.ser.out_waiting or 0) > self.max_buffered:
            if time.monotonic() > deadline:
                break
            time.sleep(0.02)

    def send_msg(self, mtype: int, body: bytes = b"") -> None:
        """Send one message. Must fit in a single frame."""
        payload = bytes([mtype]) + body
        if len(payload) > MAX_PAYLOAD:
            raise ValueError(f"message too big for one frame: {len(payload)} > {MAX_PAYLOAD}")
        frame = build_frame(self._seq, time.time(), payload)
        self._throttle(len(frame))
        self.ser.write(frame)
        self._seq += 1
        self.frames_sent += 1
        self.bytes_sent += len(frame)

    def send(self, data: bytes, chan: int = 0) -> None:
        """Send arbitrary bytes on a channel, fragmenting as needed.

        Message boundaries are preserved: the receiver reassembles and hands
        you back exactly the blob you passed in.
        """
        room = self.max_payload - 3        # type + chan + flags
        if not data:
            self.send_msg(DATA, bytes([chan, 0]))
            return
        for off in range(0, len(data), room):
            piece = data[off:off + room]
            more = 1 if off + room < len(data) else 0
            self.send_msg(DATA, bytes([chan, more]) + piece)

    def send_text(self, text: str, chan: int = 0) -> None:
        self.send(text.encode("utf-8"), chan)

    # -- reading ------------------------------------------------------------

    def poll_raw(self, timeout: float = 0.0):
        """Yield (mtype, body) for every message available within `timeout`."""
        deadline = time.monotonic() + timeout
        while True:
            chunk = self.ser.read(4096)
            if chunk:
                for _seq, _t, payload in self.parser.feed(chunk):
                    self.frames_recv += 1
                    if payload:
                        yield payload[0], payload[1:]
            if time.monotonic() >= deadline:
                return

    def reassemble(self, body: bytes):
        """Feed one DATA message body in; get (chan, blob) back once the last
        fragment lands, or None while more are still outstanding.

        Use this when you're already draining poll_raw() yourself - calling
        poll() as well would mean two readers competing for the same bytes,
        and whichever loses silently drops messages.
        """
        if len(body) < 2:
            return None
        chan, flags = body[0], body[1]
        buf = self._partial.setdefault(chan, bytearray())
        buf += body[2:]
        if flags & 1:
            return None
        out = bytes(buf)
        del self._partial[chan]
        return chan, out

    def poll(self, timeout: float = 0.0):
        """Yield (chan, data) for reassembled DATA messages only."""
        for mtype, body in self.poll_raw(timeout):
            if mtype == DATA:
                got = self.reassemble(body)
                if got is not None:
                    yield got

    # -- housekeeping -------------------------------------------------------

    def drain(self) -> None:
        """Block until everything queued has actually gone out the port."""
        try:
            self.ser.flush()
        except Exception:
            pass

    def close(self) -> None:
        self.drain()
        self.ser.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
