# Multi-band RSSI survey — Pluto/AD9363 → SiK 915 MHz link → laptop

Sweeps several bands on the Zynq7020+AD9363 board (by default 2.4 GHz, 5.8 GHz
and 915–928 MHz), bins each one, and streams the binned spectra over your
915 MHz SiK link inside a 17 kbit/s budget, holding a slice of that budget back
for the GNSS HAT, which now sends its position on that slice. The laptop decodes
it all and shows a live spectrum and waterfall per band, plus RSSI-vs-time and a
relative-range display. Every logged sweep carries the latitude, longitude and
altitude it was taken at, ready for a heat map.

Built on your existing `siklink.py`: same framing, same CRC, same radios.
`siklink.py` itself is unchanged.

## Files

| File | Runs on | Purpose |
|---|---|---|
| `bands.py` | Pi | **Which bands get scanned.** Edit `DEFAULT_BANDS` / `BAND_PRESETS` |
| `rssi_wire.py` | **both** | Wire format for a binned sweep (v2 adds a band id). Required at both ends |
| `sdr_scan.py` | Pi | SDR capture + DSP. Standalone CLI for testing the radio |
| `sdr_tx.py` | Pi | Sweep every band → encode → send, inside the byte budget |
| `linkmux.py` | Pi | Shares the one serial link: paced spectrum budget + reserved aux budget |
| `aux_channels.py` | **both** | Slot for extra data on the link. Base classes, registry, test pattern |
| `max_m8q_gps.py` | Pi | Your MAX-M8Q HAT reader (one change: it no longer probes `/dev/ttyUSB0`) |
| `gnss_source.py` | **both** | Puts the HAT on the link: 31 B fixes on channel 4, positions onto sweeps |
| `base_rx.py` | laptop | Receive → decode → track → visualise, one column per band |
| `vlink.py` | Pi / Linux | Fake radio pair, so you can test with no hardware |
| `fake_gnss.py` | Pi / Linux | Fake GNSS HAT, so you can test indoors |
| `selftest.py` | either | 156 checks. Run this before you trust any of it |

Plus your originals: `siklink.py`, `sik_send.py`, `sik_recv.py`, `sik_tx.py`,
`sik_rx.py`, `example_api.py`.

## Install

The self-tests pass on Python 3.10, 3.11 and 3.13 with numpy 2.4, pyadi-iio
0.0.21 and pylibiio 0.25, installed as below against libiio 0.25 from apt.
Raspberry Pi OS Trixie ships Python 3.13 and libiio 0.26, and Bookworm ships
3.11 and 0.24. Both are libiio v0.x, which is what pyadi-iio's bindings expect.

### Raspberry Pi 5

Use 64-bit Raspberry Pi OS, set up with Raspberry Pi Imager with SSH turned on.

```bash
# 1. update everything
sudo apt update && sudo apt full-upgrade -y
sudo reboot

# 2. system packages: libiio + its tools (iio_info, iio_attr), venv support
sudo apt install -y libiio-utils python3-libiio python3-venv python3-pip

# 3. let your user open the SiK radio's serial port (log out and back in after)
sudo usermod -aG dialout $USER

# 4. a virtual environment for the Python packages
python3 -m venv --system-site-packages ~/sdrenv
source ~/sdrenv/bin/activate
pip install --upgrade pip
pip install -r requirements-pi.txt     # numpy pyserial pyadi-iio matplotlib
```

Run `source ~/sdrenv/bin/activate` in every new terminal before running the
scripts, or call `~/sdrenv/bin/python sdr_tx.py ...` directly.
(If you'd rather skip the venv:
`pip3 install -r requirements-pi.txt --break-system-packages`.)

Copy this folder to the Pi. From PowerShell on the laptop:

```powershell
scp -r edge_node <user>@<pi-name>.local:~/sdr
```

**Power.** A Pi 5 only gives USB devices 600 mA in total unless it's on the
official 27 W (5 A) supply, which raises it to 1.6 A. The SDR plus the SiK radio
can go over 600 mA, and the symptom is random USB disconnects. Use the 27 W
supply, or power the SDR from its own supply if it has one.

**Which USB port.** Plug the SDR and the SiK radio into the black USB 2 ports,
not the blue USB 3 ones. USB 3 radiates broadband noise around 2.4 GHz.

### The SDR board

It shows up to libiio as a PlutoSDR. Over USB it's `ip:192.168.2.1`. Over its
Ethernet port it's whatever you set (see "Using the SDR's Ethernet" under
Troubleshooting; this setup uses `ip:192.168.3.1`).

```bash
ip -br addr                              # a usb0 / enx... interface with 192.168.2.10
iio_info -s                              # scans for IIO devices
iio_info -u ip:192.168.2.1 | head -20    # talks to it
python3 sdr_scan.py --uri ip:192.168.2.1 --sweeps 3   # real sweeps, no radio link needed
```

If that last command prints `WARNING: skipping band ism58`, the radio is limited
to 325–3800 MHz. OpenSourceSDRLab lists this board as shipped at 70 MHz–6 GHz,
so it should be fine, but if not:

```bash
ssh root@192.168.2.1          # Pluto firmware default password: analog
fw_printenv attr_name attr_val   # what it's set to now
fw_setenv attr_name compatible
fw_setenv attr_val ad9364        # or ad9361 to keep both RX channels
reboot
```

If `sdr_scan.py` dies with `No device found with name cf-ad9361-lpc`, or the
SDR's `dmesg` shows `dig_tune ... FAILED`, the receive path didn't start in
two-channel mode. Put it in single-channel mode (all this code uses anyway):

```bash
ssh root@192.168.2.1 "fw_setenv mode 1r1t; reboot"
```

**Setting up a second board the same way.** Plug in one SDR at a time, since
both answer on `192.168.2.1`. Copy the settings off the working board, then
write the same values onto the new one:

```bash
ssh root@192.168.3.1 "fw_printenv mode attr_name attr_val usb_ethernet_mode ipaddr_eth netmask_eth"
# swap boards, then on the new one (settings it didn't print stay unset):
ssh-keygen -R 192.168.2.1; ssh-keygen -R 192.168.3.1   # new board, new SSH key
ssh root@192.168.2.1 "fw_setenv mode 1r1t; fw_setenv ipaddr_eth 192.168.3.1; fw_setenv netmask_eth 255.255.255.0; reboot"
sudo ip neigh flush dev eth0                           # forget the old board's MAC
```

Don't flash ADI's stock PlutoSDR firmware onto this board. It's a different Zynq
(7020 vs 7010) and needs its own image, which is written to an SD card.

### The SiK radio on the Pi

```bash
ls -l /dev/serial/by-id/      # use this path for --port; it never changes
```

### Laptop (Windows)

```powershell
winget install Python.Python.3.13      # or python.org - tick "Add python.exe to PATH"
py -m pip install --upgrade pip
py -m pip install -r requirements-laptop.txt   # numpy matplotlib pyserial
py -m serial.tools.list_ports -v       # find the SiK radio's COM port
py base_rx.py --port COM5              # use the COM number from the line above
```

If the radio doesn't show up as a COM port, Device Manager will show it as an
unknown device. Install the FTDI VCP driver or the Silicon Labs CP210x driver,
depending on which USB chip it uses.

`vlink.py` and `selftest.py --live` need a Linux pseudo-terminal, so run those
on the Pi (or in WSL). Plain `py selftest.py` works on Windows.

### Keeping it up to date

```bash
# Pi
sudo apt update && sudo apt full-upgrade -y
source ~/sdrenv/bin/activate && pip install --upgrade -r requirements-pi.txt
python3 selftest.py --live        # re-check after any libiio / pyadi-iio update
```

```powershell
# laptop
py -m pip install --upgrade -r requirements-laptop.txt
```

### The GNSS HAT

```bash
sudo raspi-config     # Interface Options > Serial Port: login shell NO, hardware YES
sudo reboot
python3 max_m8q_gps.py --port /dev/ttyAMA0 --once   # outside: prints lat,lon
```

That's it. `sdr_tx.py` starts it automatically. Don't run `max_m8q_gps.py` while
`sdr_tx.py` is running, because they'd fight over the port. Don't install gpsd
either, for the same reason (or `sudo systemctl disable --now gpsd.socket gpsd`
if it's already installed).

Every `.py` file here sits in the same folder at both ends. The laptop only
strictly needs `base_rx.py`, `rssi_wire.py`, `bands.py`, `aux_channels.py`,
`linkmux.py` and `siklink.py`.

## Quick start

Prove it works with no hardware first. Run this on the Pi (or any Linux box;
`vlink.py` doesn't run on Windows), in three terminals:

```bash
python3 vlink.py --link-a /tmp/A --link-b /tmp/B --loss 5        # fake radios
python3 fake_gnss.py --link /tmp/GPS                             # fake GNSS HAT
python3 sdr_tx.py  --source sim --port /tmp/A --gnss-port /tmp/GPS --csv pi.csv
python3 base_rx.py --port /tmp/B --csv base.csv                  # real GUI
```

Then the real thing:

```bash
# Pi
python3 sdr_tx.py --uri ip:192.168.2.1 --port /dev/serial/by-id/usb-FTDI_...

# laptop (Windows: give it the COM port)
python base_rx.py --port COM5 --csv survey.csv
python base_rx.py --no-gui          # over ssh
```

## Bands

The defaults, from `bands.py`:

| band | range | bins | bin width | tiles | ~sweep time on hardware |
|---|---|---|---|---|---|
| `ism24` | 2400–2483.5 MHz | 256 | 326 kHz | 12 | ~120 ms |
| `ism58` | 5725–5875 MHz | 256 | 586 kHz | 22 | ~220 ms |
| `ism915` | 915–928 MHz | 128 | 102 kHz | 2 | ~20 ms |

(Sweep times scaled from your `survey.csv`, which shows 12 tiles in ~120 ms.)
The bands share the one radio and are swept round-robin, so all three together
run at roughly 2.5–3 cycles a second.

**Changing them.** Either edit `DEFAULT_BANDS` in `bands.py` (and add your own
entries to `BAND_PRESETS`), or override per run without touching code:

```bash
python3 sdr_tx.py --band ism24                        # just 2.4 GHz
python3 sdr_tx.py --band ism24,gnss_l1,ism915         # presets, comma list
python3 sdr_tx.py --band ism24 --band my=433.05:434.79:64:30
python3 sdr_tx.py --list-bands                        # what presets exist
```

Custom bands are `name=START:STOP[:BINS[:GAIN[:EVERY]]]` in MHz and dB.
`EVERY=3` sweeps that band only every third cycle.

**Gain is per band.** One gain rarely suits every band — your own SiK radio sits
in 915–928 MHz centimetres from the SDR and clips it at a gain that is fine
for 2.4 GHz. So `ism915` starts at 20 dB and the others use `--gain` (40).
Each band's gain is set after tuning to it (the AD936x gain table changes at 1.3
and 4 GHz: max 73 / 71 / 62 dB) and every sweep carries the gain it used.

**5.8 GHz needs AD9364 mode.** A stock AD9363 tunes 325–3800 MHz. On
Pluto-style firmware:

```bash
ssh root@192.168.2.1        # Pluto default password: analog - check your board's docs
fw_setenv attr_name compatible
fw_setenv attr_val ad9364
reboot
```

`sdr_tx.py` reads the real tuning range off the radio at startup. If it can't
reach a band it prints why and skips that band — the other bands still run, and
the base station shows a note. Expect worse sensitivity up there: the AD9363 is
outside its datasheet range and your cal offset won't carry over from 2.4 GHz.

## The 17 kbit/s budget

17 kbit/s = 2125 B/s. It's split two ways:

```
--kbps 17            everything the Pi puts on air, framing included
--reserve-kbps 3     held back for GNSS / aux. The spectrum never touches it
= 14 kbit/s          spectrum sweeps + status messages
```

The reserve stays free **whether or not anything is using it**, so the GNSS
code can go in later without re-tuning anything.

Measured over the virtual link (`selftest.py --live`, 3 bands + a GNSS-sized
test stream at 5 Hz): spectrum 14.04 kbit/s, aux 2.12 of 3.0 reserved, total
16.16 kbit/s.

Bytes per sweep, typical (worst case in brackets), framing included:

| band | wire bytes | |
|---|---|---|
| `ism24` 256 bins | ~213 (336) | |
| `ism58` 256 bins | ~184 (336) | |
| `ism915` 128 bins | ~139 (187) | |
| one full cycle | ~536 (859) | → ~3.3 cycles/s at 14 kbit/s |

That's roughly balanced against the SDR's ~2.8 cycles/s, so on hardware you'll
be at the edge of both. The 5-second report on the Pi says which one is the
limit (`link-bound` / `SDR-bound`).

**Pacing, not dropping.** The single-band version captured a sweep and then
binned it if the budget said no. With three bands of different sizes that
starves the big one: the small 915 sweep always fits in the leftover budget and
the 5.8 GHz one never does. Now the loop waits for budget *before* capturing,
then sends straight away. Nothing is queued, so what arrives is still the newest
measurement, no SDR time is wasted on sweeps that get thrown away, and every
band gets its turn (`selftest.py` checks the sweep counts per band come out equal).

**Each sweep is independent.** Delta-coding against the previous sweep would
compress far better, but one lost packet would then corrupt everything after it
until the next keyframe. Lose a sweep here, lose exactly one sweep.

Set `--kbps` to what your link really carries. Overdriving a SiK radio *lowers*
goodput — it overflows the internal buffer and corrupts frames.

## GNSS

`gnss_source.py` wraps your `max_m8q_gps.py` and plugs it into the link's GNSS
slot. It's registered in `aux_channels.py`, so both ends pick it up with no flags.

```
Pi:     MAX-M8Q (/dev/ttyAMA0, 9600) -> MaxM8Q.fixes() -> 31 B fix, channel 4
laptop: GnssHandler -> position track -> lat/lon/alt on every logged sweep
```

- **Every epoch is sent**, fix or not: a "no fix, 3 sats" message is how you know
  the GPS is alive but still acquiring. The top line of the base station shows
  it.
- **On the air.** 31 B per fix, 52 B with framing: 0.42 kbit/s at the M8Q's
  default 1 Hz, so the 3 kbit/s reserve has lots of room. You could drop to
  `--reserve-kbps 1` and give the difference to the spectrum. I left it at 3 so
  a 5 Hz fix rate fits later.
- **Positions go on sweeps by timestamp.** Fixes arrive once a second and sweeps
  ~3 times a second, so each sweep gets a position interpolated between the two
  fixes either side of it. Logs are written ~1.5 s late so that the fix after
  each sweep has arrived by then. A sweep with no fix within 1.5 s gets empty
  lat/lon cells, not a stale position. Measured on the bench against the fake
  receiver doing 6 m/s: the tagged positions move at 5.99 m/s.
- **Two logs.** `sdr_tx.py --csv` on the Pi has every sweep, because nothing is
  lost to the radio. `base_rx.py --csv` on the laptop has what made it through
  the link. Use the Pi one for heat maps.
- **Clock.** Sweeps and fixes are both stamped with the Pi's clock, so they line
  up even when the Pi's clock is wrong, which it will be in the field with no
  NTP. The only error is how late the GGA sentence arrives, about 0.1–0.4 s at
  9600 baud, which is 1–2 m at drone speed. If the Pi's clock happens to agree
  with GPS, the exact GPS epoch is used instead.
- **Port.** Always explicit: `/dev/ttyAMA0` by default, or `--gnss-port`, or
  `--no-gnss` to leave it out. Autodetect isn't safe on the drone, because
  `/dev/ttyUSB0` is the SiK radio. That's why `max_m8q_gps.py` no longer probes
  it.
- **Indoors:** `python3 fake_gnss.py --link /tmp/GPS`, then `--gnss-port /tmp/GPS`.
  It sends the same sentence mix a real M8 does, driving round a circle.

### Adding something else to the link

The same slot takes anything. Write an `AuxSource` (Pi side) and an `AuxHandler`
(laptop side), and add them to `PI_SOURCES` / `BASE_HANDLERS` in
`aux_channels.py`. `gnss_source.py` is the worked example. Channels 5–14 are
free, and every message costs its payload + 21 B of framing.
`--aux-test HZ:BYTES` rehearses any rate and size against the reserve.

## Reading the display

One column per band:

**spectrum** — current sweep, max-hold, noise floor. Known channels shaded
(WiFi 1/6/11 at 2.4 GHz, WiFi 149/157/165 at 5.8 GHz), BLE advertising channels
marked green. The title shows sweep rate, loss, gain and floor for that band,
and turns red on ADC clipping.

**waterfall** — last 300 sweeps of that band, newest at top.

Across the bottom, for emitters in **all** bands (ranked by SNR over their own
band's floor, so a noisy band can't hog the list):

**RSSI vs time** — one line per tracked emitter. The thick line is median
filtered and the faint line is raw. The gap between them is your multipath fading.

**relative range** — how far you are from each emitter compared to the closest
you have been to it. `1.0x` means you are at your best position so far. Green
= closing, red = receding. This is the panel you navigate by.

The top line shows link stats and one line per aux channel (GNSS, test pattern).

## Localising without GPS — what this can and cannot tell you

It gives you **relative** range, not metres. Absolute distance from RSSI needs
a calibrated reference at a known range, and the first wall you walk past
invalidates that calibration. What you get instead:

```
d_now / d_closest = 10 ^ ((RSSI_best - RSSI_now) / (10n))
```

`n` is the path-loss exponent — 2.0 free space, 2.5 default here, 3–4 indoors.
Set it with `--pathloss-n`. It scales the range numbers; it does not change
which direction is warmer.

**Fading is the real enemy.** Move six centimetres at 2.4 GHz (half a
wavelength, and half that again at 5.8 GHz) and multipath can swing RSSI 20 dB.
So RSSI is **median** filtered over 5 sweeps, and the warmer/colder arrow is a
**linear fit over 4 seconds**. That lag is deliberate — an instant arrow points
at multipath, not at the transmitter.

Walk in a straight line for five seconds and watch the trend. Don't sweep the
receiver around. Cross-check by walking a second leg at right angles.

## The DSP, and the things that will bite you

**Sweeping.** The AD9363 gives ~20 MHz instantaneous. The bands are wider, so
each one tiles. Tiles abut exactly — no gaps, no double counting — and only the
middle 75% of each is kept.

**Power is density-correct.** `|X|²/(N·Σw²)`, so summing fine bins into coarse
ones recovers true power for CW *and* wideband. The tracker sums each emitter's
bins, so a tone split across two narrow 915 MHz bins still reads right.

### 1. The DC spike (`--lo-mode`)

`offset` (default) parks the LO to one side so DC lands outside the measured
region: no artefact, no blind spot, ~2× the tiles. `notch` is faster but blind
for ~98 kHz at each tile centre.

### 2. Stale buffers after a retune

libiio queues 4 kernel buffers by default, and the DMA keeps filling them in the
background. Right after a retune those can still hold samples from the *old*
LO, and they get stitched into the wrong tile. A real signal then shows up whole
tile-steps (7 MHz multiples) from where it is. `sdr_scan.py` now cuts the queue
to 1 buffer, as ADI's own multi-tune examples do. If the driver won't allow
that, it raises `--flush` to 4 instead. Your `survey.csv` has the BLE advertising
channels exactly where they belong, so this wasn't hitting you. This change just
makes sure a libiio or firmware update can't start it.

### 3. The AGC will ruin your RSSI

Manual gain is the default. With AGC on, the radio re-gains between tiles, and
"the signal got stronger" really means "the AGC backed off".

`--auto-gain` steps gain only *between* sweeps, per band. It now steers on each
band's **measured ADC loading** (loudest tile's total power, target −25 dBFS
RMS, and at least 6 dB between the largest sample and the rails), climbing at
most 10 dB a step. The old rule aimed the strongest *output bin* at −15 dBFS.
A 20 MHz carrier's strongest bin sits ~18 dB under what the ADC actually sees,
so the gain walked straight into clipping and back: on a −62 dBm WiFi carrier
it went 34→54→44→34 with 8 of 12 sweeps clipped. It's off for `ism915`, where it
would chase your own radio's bursts.

**ADC clipping** is detected per rail (I and Q separately) and flagged. A
clipped sweep reads **low**, so clipped sweeps are shown but excluded from
tracking.

## Calibration

Absolute dBm needs one measurement per band. Feed a signal generator at a known
level in, run a sweep, then:

```python
from sdr_scan import calibrate
calibrate(measured_dbm=-31.4, known_dbm=-40.0, cal_offset_db=-10.0)  # -> -18.6
```

Put the result in that band's `cal_offset_db` in `bands.py` (or `--cal-offset`
for all bands). The AD936x front end is nowhere near flat from 900 MHz to
5.8 GHz, so one number won't do for all three.

**You may not need to.** Localisation compares an emitter against itself over
time, and a constant offset cancels exactly.

## Verifying

```bash
python3 selftest.py           # 146 checks, ~20 s (the GNSS serial test is Linux-only)
python3 selftest.py --live    # 156: + real 3-band + GNSS tx/rx over a lossy virtual radio, ~45 s
```

New since the single-band version: band parsing, per-band tiles/gain/levels,
unreachable-band skipping, a mocked Pluto driver (stale buffers, scaling, gain
table order across band hops), wire v2 and v1 compatibility, the budget split
with a GNSS-sized stream offered at 2× its reserve, fragment interleaving through
the real siklink parser, per-band base-station routing, Pi clock skew, replay of
every log format, and auto-gain stability on wideband signals. GNSS: your
parser on a u-blox sentence mix, the fix message, clock handling, position
interpolation, `GnssSource` over a real serial port, and position-tagged logs at
both ends of the live test.

## Known limits

- **Your own SiK radio is in `ism915`.** When it transmits you'll see it, and
  at centimetres it can clip the SDR even at low gain. It can also desensitise
  the SDR when it's tuned to 2.4 or 5.8 GHz, because the AD9363 has no
  preselector. Separate the antennas as far as the airframe allows.
- **Each tile sees 0.4 ms of air.** A sweep is a string of short snapshots
  taken at different moments, so a bursty wideband signal (WiFi) looks blocky
  across tile boundaries, and a short burst is only caught if it happens during
  its tile's snapshot.
- **Seam accuracy.** A narrowband signal within ~20 kHz of a tile seam reads up
  to 2.5 dB off. Wideband signals are within 1 dB.
- **Noise floor** is the 20th percentile of bins. That survives up to ~80% of a
  band being occupied. A jammer covering a whole band raises the "floor" instead
  of showing as an emitter, so watch the floor figure in each band's title.
- **Multi-frame sweeps.** A busy 256-bin sweep can need two siklink frames.
  siklink reassembles per channel with no fragment index, so losing the last
  half of one sweep also spoils the next sweep on that channel. The decoder
  catches it (`bad`), it just costs two sweeps instead of one.
- **Position is only as good as the M8Q.** A few metres (HDOP on the base
  station's top line), and 1 Hz. A drone at 10 m/s moves 10 m between fixes,
  and interpolation assumes a straight line between them.

## Options worth knowing

**sdr_tx.py**

```
--source sim              run with no SDR attached
--uri ip:192.168.2.1      libiio URI of the board
--port /dev/serial/by-id/...   the radio. Autodetect avoids the SDR's own
                          console and u-blox GNSS, but be explicit on the Pi
--band SPEC               bands to scan, replacing bands.py defaults
--list-bands              show presets
--kbps 17                 total link budget, all channels
--reserve-kbps 3          held back for GNSS / aux
--gain 40                 manual RX gain for bands without their own
--auto-gain               steer each band's gain off the ADC rails
--lo-mode offset          'notch' is faster with blind spots
--interval 0              minimum seconds per cycle
--aux MOD:CLASS           run an extra aux source (the GNSS, later)
--gnss-port /dev/ttyAMA0  the GNSS HAT (default). --no-gnss to leave it out
--aux-test HZ[:BYTES]     test stream on the reserve
--csv file                log locally as well as transmitting
```

**base_rx.py**

```
--port COM5               the radio. On Windows, pass it - autodetect can
                          land on a Bluetooth COM port
--no-gui                  text dashboard, for ssh
--csv survey.csv          log every sweep as it arrives (now with band_id)
--aux-csv aux.csv         log every aux message (GNSS etc) raw, as hex
--aux-handler 4:MOD:CLASS decoder for an aux channel
--replay survey.csv       play a log back - any format this chain has written
--track 2412,5806         follow these MHz; each goes to the band containing it
--pathloss-n 2.5          2 free space, 3-4 indoors
--snr 8                   dB over noise to call something an emitter
--median 5                sweeps in the median filter
--merge-gap 3             bridge fading nulls up to N bins wide
```

## Troubleshooting

**`WARNING: skipping band ism58`.** The radio can't tune 5.8 GHz. See AD9364
mode above, or drop the band with `--band ism24,ism915`.

**Nothing arrives.** Both SiK radios need solid green. Flashing means unpaired:
check `NETID`, `AIR_SPEED` and frequency match at both ends. On Windows, check
`base_rx` opened the right COM port (it prints it).

**CRC errors climbing.** `--kbps` is above what the link carries. Lower it,
and `--reserve-kbps` will still hold back the same amount for GNSS.

**`ism915` always shows ADC CLIPPING.** That's your own telemetry radio. Lower
that band's gain in `bands.py`, or move the antennas apart.

**`BrokenPipeError`, `READ LINE: -32`, or "SDR connection lost".** The Pi's
network connection to the SDR over the USB cable closed. `sdr_tx.py` and
`sdr_scan.py` both reconnect by themselves and carry on, and the base station
shows "SDR DROPPED OUT". Work out which of these it is:

- **Power.** `dmesg -T | tail -30` on the Pi shows `USB disconnect`,
  `over-current` or `Undervoltage`, or `vcgencmd get_throttled` isn't `0x0`.
  The Pi 5 only gives USB 600 mA in total unless it's on the 27 W supply.
- **The SDR rebooted.** `ssh root@192.168.2.1 uptime` shows a short uptime.
- **Neither.** The SDR stayed up and on USB, and only the network session
  dropped. The SDR's USB networking defaults to RNDIS, a Windows protocol.
  Switch it to NCM, which Linux supports natively:
  `ssh root@192.168.2.1 "fw_setenv usb_ethernet_mode ncm; reboot"`. The address
  stays `192.168.2.1`. If it still drops, skip networking altogether with
  libiio's USB backend: `iio_info -s` shows the `usb:` URI to pass as `--uri`.

- **The SDR itself hung.** It stops answering entirely and only comes back
  after unplugging it (`iio_info` gives `Connection timed out (110)`). Next
  time, before unplugging, check whether it's alive on its serial console:
  `python3 -m serial.tools.miniterm /dev/serial/by-id/usb-Analog_Devices*-if03 115200`
  (log in `root` / `analog`, run `uptime; dmesg | tail -20`, exit with Ctrl-]).
  Console answers → only its networking died → use Ethernet (below). Console
  dead too → the board froze: heat or power.
- **It reconnects, then drops straight away, over and over** (libiio prints
  `Open unlocked: -32`). The SDR takes the connection and settings but hangs
  up as soon as streaming starts. The scripts now keep retrying instead of
  claiming it's back, and tell you to reboot it: `ssh root@192.168.3.1 reboot`
  (or `192.168.2.1`), or unplug it. Grab `ssh root@192.168.3.1 "uptime; free;
  dmesg | tail -40"` first: it shows what went wrong on the SDR's side.
- **Heat.** `sdr_scan.py` prints the AD936x and Zynq temperatures every 30 s
  and again when it drops; `sdr_tx.py` sends them to the laptop (`SDR 52/68 C`
  on the top line). If drops start once it's been running a while and the Zynq
  is in the 70s-80s °C, add a heatsink and a small fan. Commercial-grade Zynq
  parts are only rated to 85 °C.

A soak test counts the drops: `python3 sdr_scan.py --uri ip:192.168.2.1
--sweeps 3000` runs about 18 minutes and lists when each drop happened, how
long each took to come back, and the temperatures.

### Using the SDR's Ethernet instead of USB networking

If the board has an Ethernet port (Pluto+ and similar Zynq 7020 boards), cable
it straight to the Pi 5's Ethernet port. That takes the USB networking out of
the picture on both ends, and it's gigabit. Keep the USB cable in for power.
The code doesn't change; only `--uri` does.

On the SDR (over the USB link, while it still works). The default is DHCP,
which gets nothing on a direct cable, so give it a fixed address:

```bash
ssh root@192.168.2.1                  # password: analog
fw_printenv ipaddr_eth netmask_eth    # blank = DHCP
fw_setenv ipaddr_eth 192.168.3.1
fw_setenv netmask_eth 255.255.255.0
reboot
```

(Or from any PC, edit `config.txt` on the SDR's USB drive: set `ipaddr_eth`
and `netmask_eth` under `[USB_ETHERNET]`, which on these boards controls the
real Ethernet port, then eject the drive.)

On the Pi (Raspberry Pi OS Bookworm, NetworkManager):

```bash
sudo nmcli con add type ethernet ifname eth0 con-name sdr-eth \
  ipv4.method manual ipv4.addresses 192.168.3.10/24 ipv4.never-default yes \
  ipv6.method disabled connection.autoconnect-priority 10
sudo nmcli con up sdr-eth
ping -c 3 192.168.3.1
iio_info -u ip:192.168.3.1 | head
```

`never-default` keeps WiFi as the Pi's route to the internet. If you ever
plug the Pi's Ethernet into a router instead, `sudo nmcli con down sdr-eth`.
Then run everything with `--uri ip:192.168.3.1`.

**`No module named 'adi'`.** `pip3 install pyadi-iio --break-system-packages`.

**`Unable to create context`.** The board isn't reachable at `--uri`. Check
with `iio_info -u ip:192.168.2.1`.

**RSSI jumps 20 dB between sweeps.** That's normal multipath if you're moving.
If you're stationary, check AGC is off.

**Tracks keep appearing and vanishing.** Fading is fragmenting a wideband
carrier. Raise `--merge-gap` to 5 or 6.

## What changed from the single-band version

- Multiple bands (`bands.py`, `--band`), each with its own bins, gain, cal
  offset and cadence. Wire format v2 carries a band id; v1 still decodes.
- Link split into spectrum budget and a reserved aux budget. Status messages
  are now charged to the budget (they weren't before). Framing is counted
  exactly (21 B per frame, not 18).
- Pacing before capture instead of dropping after it, so no band starves.
- `aux_channels.py`: the GNSS slot, plus a test pattern.
- Fixes: stale kernel buffers after retune; ADC scale detection that could be
  off by 66 dB on a quiet first capture; auto-gain oscillation on wideband
  signals; clip detection per rail; RSSI plot empty on `--replay` (and in the
  field whenever the Pi's clock was off), because it was plotted against the
  laptop's clock; `--replay` couldn't read `sdr_scan.py` logs; serial
  autodetect could pick the SDR's console on the Pi or a Bluetooth port on
  Windows.
- `--bins`, `--f-start`, `--f-stop` are gone. Bands carry those now.
- `--decimate` is gone too. Use a band's `every` field.
- GNSS: `max_m8q_gps.py` wired in through `gnss_source.py`. Fixes go out on
  channel 4, positions are interpolated onto every logged sweep at both ends,
  aux sources retry until their hardware appears, and `fake_gnss.py` is there
  for bench tests.
