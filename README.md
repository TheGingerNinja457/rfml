# RF Jamming vs. Interference Classifier (2.4 GHz)

Machine learning classification of **intentional jamming** versus **unintentional interference** in the 2.4 GHz ISM band, developed for a planned UAV-mounted RF monitoring system (BE Hons industrial project).

Nine classes are generated synthetically as 1024-sample IQ windows at 20 MS/s (51.2 µs):

| Group | Classes |
|---|---|
| Clean | `noise` |
| Benign | `wifi_ofdm`, `wifi_dsss`, `bluetooth`, `wifi_congestion` |
| Jamming | `jam_tone`, `jam_barrage`, `jam_sweep`, `jam_pulsed` |

Real Wi-Fi and Bluetooth bursts from the Fraunhofer FKIE dataset are held out as a real-world test set. Three feature-based baselines (random forest, XGBoost, RBF-SVM on 11 hand-crafted features) are compared with a two-branch 1D CNN on raw IQ.

## Results (synthetic test set, 20,250 windows)

| Model | Accuracy | Macro-F1 | Jamming detection (Pd) | False alarms (Pfa) | Real-signal false alarms |
|---|---|---|---|---|---|
| Random forest | 82.5% | 0.824 | 75.0% | 4.5% | 25.5% |
| XGBoost | 83.1% | 0.830 | 76.5% | 4.7% | 30.7% |
| SVM (RBF) | 80.7% | 0.804 | 71.1% | 4.0% | 18.4% |
| **CNN** | **93.5%** | **0.935** | **88.9%** | **0.7%** | **8.8%** |

All results are from synthetic data except the last column (580 real FKIE windows, benign only). No real jamming was available for testing.

## Files

| File | Purpose |
|---|---|
| `build_dataset.py` | Generates the synthetic dataset, adds the FKIE real test bursts, writes `data/rf24.h5`, `data/rf24.features.csv` and a preview figure |
| `baseline.py` | Trains and evaluates random forest, XGBoost and SVM on the hand-crafted features → `results/baseline/` |
| `train_cnn.py` | Trains and evaluates the two-branch CNN → `results/cnn/` (includes the trained model `best.pt`) |
| `jam_metrics.py` | Shared helper: binary jamming-vs-not metrics (Pd, Pfa, ROC/AUC) and plots |
| `sdr_classify.py` | Optional: capture and classify live IQ from a Pluto-compatible (AD9363) SDR |
| `results/` | All metrics, confusion matrices and figures used in the report |
| `captures/` | Summaries and figures from three receive-only SDR captures (raw `.npz` files not included) |

## Setup

Python 3.12. From PowerShell in the repository folder:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
```

Training was run on an NVIDIA RTX 3060 Laptop GPU (6 GB); the CNN takes about 16 minutes. It runs on CPU but much more slowly.

## Data

The `data/` folder is not included in this repository. Download the Fraunhofer FKIE RF communication signal dataset (SigMF format):

- Part I: https://doi.org/10.24406/fordatis/416 (licence CC BY-NC 4.0)
- Part II: https://doi.org/10.24406/fordatis/417 (licence CC BY-SA 4.0)

Unzip both into `data/fkie/` so the folder looks like:

```
data/fkie/basak/...
data/fkie/vuorenmaa/...
data/fkie/iqengine/...
```

Only the Wi-Fi, Wi-Fi-DSSS, Bluetooth and Bluetooth LE bursts are used; other signal types are skipped automatically.

## Running

```powershell
python build_dataset.py build --fkie-dir data\fkie   # ~135,000 synthetic + 580 real windows (seed 0)
python baseline.py                                   # ~5-10 min
python train_cnn.py                                  # ~16 min on GPU
```

To re-evaluate the saved CNN without retraining (still requires `data/rf24.h5` from the first step):

```powershell
python train_cnn.py --eval-only
```

All settings default to the values used in the report (15,000 windows per class, 70/15/15 split, SNR/JNR −5 to 25 dB, seed 0).

### Live SDR (optional)

Requires libiio v0.25 and `pip install pylibiio pyadi-iio`:

```powershell
python sdr_classify.py check
python sdr_classify.py capture --channel 6 --seconds 5 --out captures\ch6_idle.npz
python sdr_classify.py classify captures\ch6_idle.npz --floor-ref captures\floor.npz
```

The project SDR appeared to have a receiver fault, so the captures in `captures/` are not representative of classifier performance.

## Edge node and base station (`edge_node/`)

The `edge_node/` folder contains the system the classifier is designed for: a Raspberry Pi 5 with an AD9363 (Pluto-compatible) SDR sweeps the 2.4 GHz, 5.8 GHz and 915 MHz bands, bins each spectrum, and streams it with u-blox MAX-M8Q GNSS fixes over a ~17 kbit/s SiK 915 MHz telemetry link to a laptop base station, which displays live spectra and logs position-tagged sweeps for an RSSI heat map.

| Runs on | Files |
|---|---|
| Pi | `sdr_tx.py`, `sdr_scan.py`, `bands.py`, `linkmux.py`, `max_m8q_gps.py` |
| Laptop | `base_rx.py` |
| Both | `rssi_wire.py`, `aux_channels.py`, `gnss_source.py`, `siklink.py` |
| Testing (no hardware needed) | `selftest.py`, `vlink.py`, `fake_gnss.py` |
| Link throughput tests | `sik_tx.py`, `sik_rx.py` |
| Link utilities | `sik_send.py`, `sik_recv.py`, `example_api.py` |

Full setup, wiring and usage are in `edge_node/README_RSSI.md` and `edge_node/README_siklink.md`. Quick test with no hardware:

```bash
cd edge_node
pip install -r requirements-laptop.txt
python selftest.py
```

The edge node was built and bench-tested but not mounted on the UAV or flown during the project.

## Regulatory note

No jamming signals were transmitted. All jamming classes are synthetic, and the SDR is used in receive-only mode. Operating radio jammer equipment in New Zealand is prohibited without a licence.
