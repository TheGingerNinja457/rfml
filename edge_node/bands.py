"""bands - WHICH frequency bands get scanned. Edit BAND_PRESETS / DEFAULT_BANDS.

Needed on the Pi. The base station imports it only for channel labels and
overlays - it learns the actual band list from the Pi over the link, so the
two ends do not have to be kept in sync by hand.

CHANGING THE BANDS
  1. In code: edit DEFAULT_BANDS below (names from BAND_PRESETS), or add your
     own entry to BAND_PRESETS and put its name in DEFAULT_BANDS.
  2. On the command line, without touching code. --band replaces the defaults
     and can be repeated or comma-separated:

        --band ism24                         one preset
        --band ism24,ism58,ism915            the defaults, spelled out
        --band ism24 --band gnss_l1          any mix of presets
        --band myband=433.05:434.79          custom, in MHz (start:stop)
        --band myband=433.05:434.79:64:30    ...with 64 bins and 30 dB gain
        --band 1240:1300:128                 custom, auto-named

     Fields after the name are  start:stop[:bins[:gain[:every]]]  in MHz / dB.
     `every` = sweep this band on every Nth cycle only (1 = every cycle).

WHAT EACH FIELD DOES
  bins      frequency bins across the band. More bins = finer resolution but
            more bytes per sweep. 256 across 83.5 MHz = 326 kHz per bin.
  gain_db   manual RX gain for this band. None = use --gain. Per band because
            one gain rarely suits every band: a strong local emitter in one
            (your own SiK radio in 915-928 MHz, for instance) clips the ADC at
            a gain that is fine everywhere else.
  cal_offset_db  dBm = dBFS - gain + this. None = use --cal-offset. The
            AD936x front end is not flat from 900 MHz to 5.8 GHz, so a proper
            calibration is per band. Relative RSSI does not need it.
  every     1 = sweep every cycle; 3 = every third cycle. Use it to spend less
            airtime on a band you care less about.
  lo_mode   None = use --lo-mode. See sdr_scan.TilePlan.
  auto_gain False = sdr_tx --auto-gain leaves this band's gain alone.

TUNING RANGE.  A stock AD9363 tunes 325-3800 MHz. 5.8 GHz needs the chip
running in AD9364 mode (70-6000 MHz) - on Pluto-style firmware that is
`fw_setenv attr_name compatible; fw_setenv attr_val ad9364; reboot` on the
SDR. sdr_tx.py reads the real range off the radio at startup and skips, with
a loud warning, any band it cannot reach rather than refusing to run.

SWEEP TIME.  Each tile is ~7 MHz in offset mode and costs about 10 ms on real
hardware (your survey.csv: 12 tiles in ~120 ms). So roughly:
    ism24   83.5 MHz  12 tiles  ~120 ms
    ism58    150 MHz  22 tiles  ~220 ms
    ism915    13 MHz   2 tiles   ~20 ms
plus a few ms per band for the big LO jump. ~3 full cycles a second.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

DEFAULT_BINS = 256
MAX_BANDS = 8                   # the wire format allows 256; the GUI does not


@dataclass
class Band:
    name: str
    f_start_hz: float
    f_stop_hz: float
    bins: int = DEFAULT_BINS
    gain_db: float | None = None
    cal_offset_db: float | None = None
    every: int = 1
    lo_mode: str | None = None
    auto_gain: bool = True          # let sdr_tx --auto-gain steer this band
    note: str = ""

    @property
    def span_hz(self) -> float:
        return self.f_stop_hz - self.f_start_hz

    @property
    def bin_hz(self) -> float:
        return self.span_hz / self.bins

    def __str__(self):
        g = "--gain" if self.gain_db is None else f"{self.gain_db:g} dB"
        ev = "" if self.every == 1 else f", every {self.every} cycles"
        return (f"{self.name}: {self.f_start_hz/1e6:.3f}-{self.f_stop_hz/1e6:.3f} "
                f"MHz, {self.bins} bins x {self.bin_hz/1e3:.0f} kHz, gain {g}{ev}")


# --------------------------------------------------------------- the table
#
# Add a band by adding a line here. Frequencies in Hz.

BAND_PRESETS: dict[str, Band] = {
    "ism24": Band("ism24", 2_400_000_000, 2_483_500_000, bins=256,
                  note="2.4 GHz ISM - WiFi, BLE, most drone control links"),

    "ism58": Band("ism58", 5_725_000_000, 5_875_000_000, bins=256,
                  note="5.8 GHz ISM - FPV video, WiFi ch149-165. Needs the SDR "
                       "in AD9364 mode. Max gain here is 62 dB, not 71"),

    "ism915": Band("ism915", 915_000_000, 928_000_000, bins=128, gain_db=20,
                   auto_gain=False,
                   note="NZ/AU 915-928 MHz - LoRa, ELRS 915, and YOUR OWN SiK "
                        "telemetry radio. Gain starts low because that radio "
                        "sits centimetres from the SDR and will clip it, and "
                        "auto-gain is off because it would chase your own "
                        "radio's bursts up and down"),

    # Not scanned by default - here so you can pick them with --band.
    "gnss_l1": Band("gnss_l1", 1_559_000_000, 1_610_000_000, bins=128,
                    note="GPS L1 / Galileo E1 / BeiDou B1 / GLONASS G1. Real "
                         "GNSS is below the noise floor; anything visible here "
                         "is interference or a jammer"),
    "fpv12": Band("fpv12", 1_240_000_000, 1_300_000_000, bins=128,
                  note="1.2-1.3 GHz analog FPV video, GNSS L2 nearby"),
    "ism433": Band("ism433", 433_050_000, 434_790_000, bins=64,
                   note="433 MHz LIPD - LoRa, key fobs, telemetry"),
    "ism868": Band("ism868", 863_000_000, 870_000_000, bins=64,
                   note="EU 868 MHz SRD - not used in NZ, here for completeness"),
    "fpv58_full": Band("fpv58_full", 5_645_000_000, 5_945_000_000, bins=384,
                       note="Every common 5.8 GHz FPV channel (A/B/E/F/R). "
                            "44 tiles - about 0.45 s per sweep"),
}

# What runs when you do not pass --band. Order = sweep order.
DEFAULT_BANDS = ["ism24", "ism58", "ism915"]


# ----------------------------------------------------- AD936x gain limits
# The manual gain range depends on which gain table the driver loads, which
# depends on the LO. Asking for more than the table holds makes the driver
# refuse or clamp, so clamp here first and say so.

def gain_limits_db(f_hz: float) -> tuple[float, float]:
    if f_hz < 1_300_000_000:
        return -1.0, 73.0
    if f_hz < 4_000_000_000:
        return -3.0, 71.0
    return -10.0, 62.0


# ------------------------------------------------------------------ parsing

def _parse_one(part: str) -> Band:
    part = part.strip()
    if part in BAND_PRESETS:
        return copy.deepcopy(BAND_PRESETS[part])

    name = None
    if "=" in part:
        name, part = part.split("=", 1)
        name = name.strip()
    fields = [f.strip() for f in part.split(":")]
    if len(fields) < 2:
        known = ", ".join(BAND_PRESETS)
        raise ValueError(f"'{part}' is neither a preset ({known}) nor "
                         f"start:stop in MHz")
    try:
        start = float(fields[0]) * 1e6
        stop = float(fields[1]) * 1e6
        bins = int(fields[2]) if len(fields) > 2 and fields[2] else DEFAULT_BINS
        gain = float(fields[3]) if len(fields) > 3 and fields[3] else None
        every = int(fields[4]) if len(fields) > 4 and fields[4] else 1
    except ValueError as e:
        raise ValueError(f"bad band spec '{part}': {e}") from e
    if not name:
        name = f"{start/1e6:.0f}-{stop/1e6:.0f}"
    return Band(name, start, stop, bins=bins, gain_db=gain, every=every)


def validate(bands: list[Band]) -> list[str]:
    """Raise on anything unusable; return warnings for anything merely odd."""
    warns = []
    if not bands:
        raise ValueError("no bands configured")
    if len(bands) > MAX_BANDS:
        raise ValueError(f"{len(bands)} bands - the limit is {MAX_BANDS}")
    names = [b.name for b in bands]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate band names: {names}")
    for b in bands:
        if b.f_stop_hz <= b.f_start_hz:
            raise ValueError(f"{b.name}: stop must be above start")
        if not 8 <= b.bins <= 4096:
            raise ValueError(f"{b.name}: bins must be 8..4096, got {b.bins}")
        if b.every < 1:
            raise ValueError(f"{b.name}: every must be >= 1")
        if len(b.name.encode()) > 16:
            raise ValueError(f"{b.name}: keep band names to 16 characters")
        if b.gain_db is not None:
            lo, hi = gain_limits_db(b.f_start_hz)
            if not lo <= b.gain_db <= hi:
                warns.append(f"{b.name}: gain {b.gain_db:g} dB is outside the "
                             f"AD936x range at this frequency ({lo:g}..{hi:g}); "
                             f"it will be clamped")
    s = sorted(bands, key=lambda b: b.f_start_hz)
    for a, b in zip(s, s[1:]):
        if b.f_start_hz < a.f_stop_hz:
            warns.append(f"{a.name} and {b.name} overlap - fine, but you are "
                         f"spending airtime on the overlap twice")
    if len(bands) > 4:
        warns.append(f"{len(bands)} bands - each gets a smaller share of the "
                     f"sweep rate and link budget, and the GUI gets crowded")
    return warns


def resolve_bands(specs: list[str] | None) -> list[Band]:
    """--band arguments (possibly None) -> validated list of Band."""
    if not specs:
        specs = [",".join(DEFAULT_BANDS)]
    bands: list[Band] = []
    for spec in specs:
        for part in spec.split(","):
            if part.strip():
                bands.append(_parse_one(part))
    validate(bands)
    return bands


# ------------------------------------------------------ labels and overlays
# Used by the base station to put names on detections and shade channels.
# (name, centre Hz, tolerance Hz). Nearest match within tolerance wins.

def _known_signals():
    out = []
    for ch, f in {37: 2_402_000_000, 38: 2_426_000_000,
                  39: 2_480_000_000}.items():
        out.append((f"BLE adv {ch}", f, 1.5e6))
    for n in range(1, 14):
        out.append((f"WiFi ch{n}", 2_412_000_000 + (n - 1) * 5_000_000, 6e6))
    for n in (149, 153, 157, 161, 165, 169, 173, 177):
        out.append((f"WiFi ch{n}", 5_000_000_000 + n * 5_000_000, 8e6))
    for i, f in enumerate((5658, 5695, 5732, 5769, 5806, 5843, 5880, 5917), 1):
        out.append((f"FPV R{i}", f * 1_000_000, 2.5e6))
    for name, f in (("GPS L1", 1_575_420_000), ("BeiDou B1", 1_561_098_000),
                    ("GLONASS G1", 1_602_000_000)):
        out.append((name, f, 3e6))
    return out


KNOWN_SIGNALS = _known_signals()

# (centre Hz, width Hz, label) - drawn shaded on the spectrum. Lines for
# anything with width 0.
OVERLAYS = (
    [(2_412_000_000, 22e6, "ch1"), (2_437_000_000, 22e6, "ch6"),
     (2_462_000_000, 22e6, "ch11")]
    + [(f, 0, "") for f in (2_402_000_000, 2_426_000_000, 2_480_000_000)]
    + [(5_745_000_000, 20e6, "ch149"), (5_785_000_000, 20e6, "ch157"),
       (5_825_000_000, 20e6, "ch165")]
    + [(1_575_420_000, 0, "L1")]
)
