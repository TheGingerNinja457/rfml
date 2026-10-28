"""rssi_wire - compact wire format for binned spectrum sweeps over a slow link.

Needed on BOTH the Pi and the base station.

A sweep is a fixed grid of frequency bins covering one band (2.4 GHz, 5.8 GHz,
915 MHz - whatever bands.py says), each holding one power figure in dBm. Sent
whole, every sweep, with no dependency on any previous sweep - that is
deliberate. Delta-coding against the last sweep would compress far better, but
this link drops packets, and one lost sweep would then corrupt every sweep
after it until the next keyframe. Each sweep here stands alone: lose one, lose
exactly one.

WIRE LAYOUT v2  (little-endian, sent as a siklink DATA message on CH_SWEEP)

    magic      u8     0x53 'S'
    version    u8     0x02
    flags      u8     bit0 zlib, bit1 delta-along-frequency, bit2 ADC clipped
    node_id    u8     which scanner this came from
    band_id    u8     which band this sweep covers (index into the band table
                      the Pi sends on CH_STATUS)
    seq        u16    per-band sweep counter, wraps at 65536 - loss detection
    nbins      u16    number of frequency bins
    t_ms       u32    unix time in ms, truncated to 32 bits (wraps ~49 days)
    f0_khz     u32    centre frequency of bin 0, kHz
    df_hz      u32    bin spacing, Hz
    ref_dbm10  i16    dBm of quantisation level 0, in tenths of a dB
    step_db10  u8     dB per quantisation level, in tenths of a dB
    noise_dbm10 i16   estimated noise floor, tenths of a dBm
    gain_db    i8     SDR hardware gain in dB when this sweep was taken
                      -> 27 byte header
    bins       u8[nbins] (possibly compressed - see flags)

v1 is the same without band_id (26 bytes). decode_sweep() still reads v1 and
treats it as band 0, so old logs and an old Pi build keep working.

Quantisation is 0.5 dB by default over 256 levels, so 128 dB of range hung off
a per-sweep reference sitting just below the noise floor. Nothing below the
noise floor is worth bits, and 0.5 dB is well under the fading you will see on
a moving receiver, so this is not the lossy part of the chain.

BYTE BUDGET.  256 bins compresses to roughly 90-200 B depending on how busy the
band is, plus 27 B of header, plus siklink's 21 B per frame (18 framing + 3 of
DATA header). Call it 210-250 B of wire per sweep. encoded_size() tells you
exactly, and linkmux.py enforces it rather than trusting this paragraph.

CHANNELS.  siklink channels are just a byte. 1-3 are the spectrum system's;
4-15 are reserved for auxiliary streams that share the link (GNSS first) -
see aux_channels.py. The Pi holds back a slice of the link budget for them.
"""

from __future__ import annotations

import json
import struct
import time
import zlib
from collections import deque

# ---------------------------------------------------------------- constants

MAGIC = 0x53
VERSION = 0x02
VERSION_V1 = 0x01           # still decoded, as band 0

FLAG_ZLIB = 0x01
FLAG_DELTA = 0x02
# Set when the ADC saturated during this sweep. Every dBm figure in a clipped
# sweep reads LOW, because a clipped sine has less energy than the real one, so
# walking closer to an emitter can make it look like it is getting weaker. The
# base station greys these out rather than plotting them as truth.
FLAG_CLIP = 0x04

# siklink channels. Channels are just a byte; these keep the streams apart.
CH_SWEEP = 1        # binned spectrum, continuous
CH_STATUS = 2       # scanner status JSON + band table, every few seconds
CH_LOG = 3          # human-readable notices
CH_GNSS = 4         # reserved for the GNSS hat - see aux_channels.py
CH_AUX_FIRST = 4    # 4..15 are auxiliary streams sharing the reserved budget
CH_AUX_LAST = 15

HDR_V1 = struct.Struct("<BBBBHHIIIhBhb")
HDR = struct.Struct("<BBBBBHHIIIhBhb")
HDR_LEN_V1 = HDR_V1.size    # 26
HDR_LEN = HDR.size          # 27

# siklink framing: 18 B per frame (sync, seq, timestamp, length, crc) plus the
# 3 B DATA message header (type, chan, flags) inside every frame's payload.
SIK_FRAME_OVERHEAD = 18
SIK_DATA_HDR = 3
SIK_MAX_PAYLOAD = 220       # siklink.Link default max_payload

DEFAULT_STEP_DB = 0.5

# The 2.4 GHz ISM band as the ITU sees it.
BAND_START_HZ = 2_400_000_000
BAND_STOP_HZ = 2_483_500_000


# ------------------------------------------------------------- channel plans
# For overlaying meaning on the bins at the base station. Frequencies in Hz.

WIFI_24_CHANNELS = {n: (2_412_000_000 + (n - 1) * 5_000_000) for n in range(1, 14)}
WIFI_24_CHANNELS[14] = 2_484_000_000
WIFI_BW_HZ = 22_000_000

# BLE: 40 channels of 2 MHz. Advertising channels are 37, 38, 39.
BLE_ADVERTISING = {37: 2_402_000_000, 38: 2_426_000_000, 39: 2_480_000_000}


def ble_channel_freq(ch: int) -> int:
    """Centre frequency of a BLE channel index (0-39)."""
    if ch in BLE_ADVERTISING:
        return BLE_ADVERTISING[ch]
    if not 0 <= ch <= 36:
        raise ValueError(f"BLE channel out of range: {ch}")
    # Data channels 0-36 fill the gaps around the advertising channels.
    return 2_404_000_000 + ch * 2_000_000 + (2_000_000 if ch >= 11 else 0)


# ------------------------------------------------------------------ sweep

class Sweep:
    """One pass over one band: a power figure per frequency bin.

    `bins_dbm` is a list (or numpy array) of floats in dBm. `f0_hz` is the
    centre of bin 0 and `df_hz` the spacing, so bin i sits at f0 + i*df.
    `band_id` says which configured band this is; 0 for single-band setups.
    """

    __slots__ = ("seq", "t", "node_id", "f0_hz", "df_hz", "bins_dbm",
                 "noise_dbm", "gain_db", "clipped", "band_id",
                 "lat", "lon", "alt_m")

    def __init__(self, seq, t, node_id, f0_hz, df_hz, bins_dbm,
                 noise_dbm, gain_db, clipped=False, band_id=0,
                 lat=None, lon=None, alt_m=None):
        self.seq = int(seq)
        self.t = float(t)
        self.node_id = int(node_id)
        self.f0_hz = float(f0_hz)
        self.df_hz = float(df_hz)
        self.bins_dbm = bins_dbm
        self.noise_dbm = float(noise_dbm)
        self.gain_db = int(gain_db)
        self.clipped = bool(clipped)
        self.band_id = int(band_id)
        # Where the receiver was. Not sent with the sweep - the GNSS fixes
        # travel on their own channel and get matched up by timestamp - but
        # carried here so logs can record it.
        self.lat = lat
        self.lon = lon
        self.alt_m = alt_m

    @property
    def f_lo_hz(self) -> float:
        """Lower edge of the band this sweep covers."""
        return self.f0_hz - self.df_hz / 2.0

    @property
    def f_hi_hz(self) -> float:
        """Upper edge of the band this sweep covers."""
        return self.f0_hz + (self.nbins - 0.5) * self.df_hz

    @property
    def nbins(self) -> int:
        return len(self.bins_dbm)

    def freqs_hz(self):
        return [self.f0_hz + i * self.df_hz for i in range(self.nbins)]

    def bin_of(self, f_hz: float) -> int:
        """Index of the bin containing a frequency. Clamped to the sweep."""
        i = int(round((f_hz - self.f0_hz) / self.df_hz))
        return max(0, min(self.nbins - 1, i))

    def peak(self):
        """(index, dBm) of the strongest bin."""
        best = max(range(self.nbins), key=lambda i: self.bins_dbm[i])
        return best, float(self.bins_dbm[best])

    def __repr__(self):
        i, p = self.peak()
        return (f"<Sweep band={self.band_id} seq={self.seq} n={self.nbins} "
                f"{self.f0_hz / 1e6:.1f}MHz+{self.df_hz / 1e3:.0f}kHz "
                f"peak={p:.1f}dBm@{(self.f0_hz + i * self.df_hz) / 1e6:.2f}MHz "
                f"noise={self.noise_dbm:.1f}>")


# ------------------------------------------------------------------ encode

def _quantise(bins_dbm, ref_dbm: float, step_db: float) -> bytes:
    out = bytearray(len(bins_dbm))
    inv = 1.0 / step_db
    for i, v in enumerate(bins_dbm):
        # NaN/inf from an empty bin or a log of zero must not poison the frame.
        if v != v or v in (float("inf"), float("-inf")):
            out[i] = 0
            continue
        q = int((float(v) - ref_dbm) * inv + 0.5)
        out[i] = 0 if q < 0 else (255 if q > 255 else q)
    return bytes(out)


def _delta(raw: bytes) -> bytes:
    prev = 0
    out = bytearray(len(raw))
    for i, b in enumerate(raw):
        out[i] = (b - prev) & 0xFF
        prev = b
    return bytes(out)


def _undelta(d: bytes) -> bytes:
    prev = 0
    out = bytearray(len(d))
    for i, b in enumerate(d):
        prev = (prev + b) & 0xFF
        out[i] = prev
    return bytes(out)


def encode_sweep(sweep: Sweep, step_db: float = DEFAULT_STEP_DB,
                 headroom_db: float = 10.0, compress: bool = True,
                 version: int = VERSION) -> bytes:
    """Serialise a Sweep. Picks whichever of raw / zlib / delta+zlib is smallest.

    `headroom_db` is how far below the noise floor level 0 sits. Bins quieter
    than that clip to 0, which costs nothing because they carry no information
    and clipping them makes the compressor's job easier.
    """
    ref_dbm = round(sweep.noise_dbm - headroom_db, 1)
    raw = _quantise(sweep.bins_dbm, ref_dbm, step_db)

    body, flags = raw, 0
    if compress:
        cands = [(raw, 0), (zlib.compress(raw, 9), FLAG_ZLIB),
                 (zlib.compress(_delta(raw), 9), FLAG_ZLIB | FLAG_DELTA)]
        body, flags = min(cands, key=lambda c: len(c[0]))
    if sweep.clipped:
        flags |= FLAG_CLIP

    tail = (
        sweep.seq & 0xFFFF, sweep.nbins,
        int(sweep.t * 1000.0) & 0xFFFFFFFF,
        int(round(sweep.f0_hz / 1000.0)) & 0xFFFFFFFF,
        int(round(sweep.df_hz)) & 0xFFFFFFFF,
        max(-32768, min(32767, int(round(ref_dbm * 10)))),
        max(1, min(255, int(round(step_db * 10)))),
        max(-32768, min(32767, int(round(sweep.noise_dbm * 10)))),
        max(-128, min(127, int(sweep.gain_db))),
    )
    if version == VERSION_V1:           # only for testing old decoders/logs
        hdr = HDR_V1.pack(MAGIC, VERSION_V1, flags, sweep.node_id & 0xFF, *tail)
    else:
        hdr = HDR.pack(MAGIC, VERSION, flags, sweep.node_id & 0xFF,
                       sweep.band_id & 0xFF, *tail)
    return hdr + body


def wire_size(nbytes: int, max_payload: int = SIK_MAX_PAYLOAD) -> int:
    """Bytes a siklink DATA message of `nbytes` really costs on the air.

    Every frame carries 18 B of framing and 3 B of DATA header, and a message
    longer than one frame's room is split, so a 230 B blob is two frames.
    """
    room = max_payload - SIK_DATA_HDR
    frames = max(1, -(-int(nbytes) // room))
    return int(nbytes) + frames * (SIK_FRAME_OVERHEAD + SIK_DATA_HDR)


def encoded_size(nbins: int, sweep: Sweep | None = None, **kw) -> int:
    """Bytes on the wire for one sweep, framing included.

    Use this to size your budget rather than guessing. If you pass a real
    sweep you get the true compressed size; without one you get the
    incompressible worst case, which is what you should budget against.
    """
    if sweep is not None:
        n = len(encode_sweep(sweep, **kw))
    else:
        n = HDR_LEN + nbins + 11        # + zlib's worst-case expansion
    return wire_size(n)


# ------------------------------------------------------------------ decode

class DecodeError(ValueError):
    pass


def decode_sweep(blob: bytes, now: float | None = None) -> Sweep:
    """Parse bytes back into a Sweep. Raises DecodeError on anything malformed.

    Always call this inside a try/except. siklink's CRC catches corrupted
    frames, but a frame can still be *valid* and carry a truncated payload if
    the sender was interrupted mid-message.
    """
    if len(blob) < 2:
        raise DecodeError(f"short header: {len(blob)} B")
    if blob[0] != MAGIC:
        raise DecodeError(f"bad magic 0x{blob[0]:02X}")
    ver = blob[1]
    if ver == VERSION:
        if len(blob) < HDR_LEN:
            raise DecodeError(f"short header: {len(blob)} < {HDR_LEN}")
        (magic, ver, flags, node_id, band_id, seq, nbins, t_ms, f0_khz, df_hz,
         ref10, step10, noise10, gain) = HDR.unpack_from(blob, 0)
        body = blob[HDR_LEN:]
    elif ver == VERSION_V1:
        if len(blob) < HDR_LEN_V1:
            raise DecodeError(f"short header: {len(blob)} < {HDR_LEN_V1}")
        (magic, ver, flags, node_id, seq, nbins, t_ms, f0_khz, df_hz,
         ref10, step10, noise10, gain) = HDR_V1.unpack_from(blob, 0)
        band_id = 0
        body = blob[HDR_LEN_V1:]
    else:
        raise DecodeError(f"unsupported version {ver}")

    if not 1 <= nbins <= 8192:
        raise DecodeError(f"implausible nbins {nbins}")
    if step10 == 0 or df_hz == 0:
        raise DecodeError("zero step or bin width")
    if flags & FLAG_ZLIB:
        try:
            body = zlib.decompress(body)
        except zlib.error as e:
            raise DecodeError(f"zlib: {e}") from e
    if flags & FLAG_DELTA:
        body = _undelta(body)

    if len(body) != nbins:
        raise DecodeError(f"payload {len(body)} B, header claims {nbins} bins")

    ref_dbm = ref10 / 10.0
    step_db = step10 / 10.0
    bins = [ref_dbm + b * step_db for b in body]

    return Sweep(seq=seq, t=_restore_time(t_ms, now), node_id=node_id,
                 f0_hz=f0_khz * 1000.0, df_hz=float(df_hz), bins_dbm=bins,
                 noise_dbm=noise10 / 10.0, gain_db=gain,
                 clipped=bool(flags & FLAG_CLIP), band_id=band_id)


def _restore_time(t_ms: int, now: float | None) -> float:
    """Rebuild a full timestamp from the truncated 32-bit millisecond field.

    The high bits come from the receiver's own clock. Both ends only need to
    agree to within ~24 days for this to be unambiguous, which they will.
    """
    now_ms = int((time.time() if now is None else now) * 1000.0)
    high = now_ms & ~0xFFFFFFFF
    cand = high | t_ms
    # Pick whichever wrap epoch lands nearest to now.
    for adj in (0, 1 << 32, -(1 << 32)):
        if abs((cand + adj) - now_ms) <= (1 << 31):
            return (cand + adj) / 1000.0
    return cand / 1000.0


# ------------------------------------------------------------------ status

def encode_status(d: dict) -> bytes:
    return json.dumps(d, separators=(",", ":")).encode("utf-8")


def decode_status(blob: bytes) -> dict:
    try:
        v = json.loads(blob.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise DecodeError(f"status: {e}") from e
    if not isinstance(v, dict):
        raise DecodeError("status payload is not an object")
    return v


# ------------------------------------------------------------- loss tracking

class SeqTracker:
    """Turns the wrapping 16-bit sweep counter into a loss percentage.

    Sweeps arrive in order on this link - there is no retransmission and no
    reordering - so a gap in the sequence is a lost sweep, full stop.
    """

    def __init__(self, window: int = 200):
        self.window = window
        self.last: int | None = None
        self.received = 0
        self.lost = 0
        self._recent: list[int] = []

    def observe(self, seq: int) -> int:
        """Feed a sequence number in, get the number newly lost back."""
        self.received += 1
        missed = 0
        if self.last is not None:
            gap = (seq - self.last) & 0xFFFF
            if gap == 0:
                return 0                      # duplicate
            if gap > 1000:                    # sender restarted; don't count it
                self._recent.clear()
            else:
                missed = gap - 1
                self.lost += missed
        self.last = seq
        self._recent.append(missed)
        if len(self._recent) > self.window:
            del self._recent[:-self.window]
        return missed

    @property
    def loss_pct(self) -> float:
        """Loss over the recent window, which is what you want while walking
        around - a lifetime average hides the fact that you just lost the link."""
        if not self._recent:
            return 0.0
        got = len(self._recent)
        miss = sum(self._recent)
        return 100.0 * miss / max(1, got + miss)


# ------------------------------------------------------------------- CSV logs
# One format for every log the chain writes (sdr_scan, sdr_tx, base_rx), so
# base_rx --replay can play back any of them.

CSV_HEADER = ("t,seq,node_id,band_id,noise_dbm,gain_db,clipped,lat,lon,alt_m,"
              "f0_hz,df_hz,bins...")


def _opt(v, fmt):
    return "" if v is None else format(v, fmt)


def csv_row(sw: Sweep, precision: int = 1) -> str:
    """One log line. lat/lon/alt_m are empty when there was no GNSS fix."""
    return (f"{sw.t:.3f},{sw.seq},{sw.node_id},{sw.band_id},"
            f"{sw.noise_dbm:.2f},{sw.gain_db},{int(sw.clipped)},"
            f"{_opt(sw.lat, '.7f')},{_opt(sw.lon, '.7f')},"
            f"{_opt(sw.alt_m, '.1f')},"
            f"{sw.f0_hz:.0f},{sw.df_hz:.3f},"
            + ",".join(f"{float(v):.{precision}f}" for v in sw.bins_dbm))


class DelayedSweepLog:
    """Writes sweep rows a moment late, so each can be given a position.

    Fixes come once a second; a sweep taken just after a fix has no fix AFTER
    it yet, so its position could only be the stale one before it. Holding
    rows back `delay` seconds lets the next fix arrive, and the row gets an
    interpolated position instead of one up to a second (5-10 m of drone
    travel) out of date.

    `position_at(t)` returns (lat, lon, alt) or None; pass None for no GNSS.
    """

    def __init__(self, fh, position_at=None, delay: float = 1.5,
                 precision: int = 1):
        self.fh = fh
        self.position_at = position_at
        self.delay = delay if position_at else 0.0
        self.precision = precision
        self._q: deque = deque()

    def add(self, sw: Sweep) -> None:
        self._q.append((time.monotonic(), sw))
        self.flush()

    def flush(self, force: bool = False) -> None:
        cutoff = time.monotonic() - self.delay
        while self._q and (force or self._q[0][0] <= cutoff):
            _, sw = self._q.popleft()
            if self.position_at is not None and sw.lat is None:
                pos = self.position_at(sw.t)
                if pos:
                    sw.lat, sw.lon, sw.alt_m = pos
            self.fh.write(csv_row(sw, self.precision) + "\n")

    def close(self) -> None:
        self.flush(force=True)


def parse_csv_row(hdr: list[str], row: list[str]) -> Sweep:
    """Read one row of any log this chain has ever written.

    Five layouts exist in the wild: this one (with lat/lon/alt_m); the
    first multi-band one (no position); base_rx's old one (no band_id);
    sdr_tx's old one (no node_id either); sdr_scan's old one (no clipped
    column). Columns are found by name, and the bins are whatever follows
    df_hz, so all of them load. Raises ValueError/IndexError on junk.
    """
    col = {name.strip(): i for i, name in enumerate(hdr)}
    if "df_hz" not in col or "t" not in col:
        raise ValueError("not a sweep log header")

    def get(name, default=None, conv=float):
        i = col.get(name)
        return default if i is None else conv(row[i])

    def opt(name):
        i = col.get(name)
        return float(row[i]) if i is not None and row[i] != "" else None

    first_bin = col["df_hz"] + 1
    return Sweep(lat=opt("lat"), lon=opt("lon"), alt_m=opt("alt_m"),
                 seq=get("seq", 0, lambda v: int(float(v))),
                 t=get("t"),
                 node_id=get("node_id", 1, lambda v: int(float(v))),
                 f0_hz=get("f0_hz"), df_hz=get("df_hz"),
                 bins_dbm=[float(v) for v in row[first_bin:] if v != ""],
                 noise_dbm=get("noise_dbm"),
                 gain_db=get("gain_db", 0, lambda v: int(float(v))),
                 clipped=bool(get("clipped", 0, lambda v: int(float(v)))),
                 band_id=get("band_id", 0, lambda v: int(float(v))))
