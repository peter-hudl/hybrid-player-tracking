# Hybrid Player Tracking: Halmstadt Dataset

Time-aligned IMU and optical tracking dataset from a Halmstadt U19 training session, plus a reference implementation of an identity alignment method that matches anonymous WIMU accelerometers to optically-tracked players.

This is an input dataset and proof-of-concept for the [Future Hybrid Player Tracking System](https://hudl.enterprise.slack.com/docs/T025Q1R55/F097GRFTWKF) project, which explores combining Hudl's optical (Nexus/Focus) and electronic (WIMU/Titan) player tracking systems.

---

## Quick start

**1. Download data from S3**

```bash
aws s3 sync \
  s3://hudlrd-datasets/focus_nexus/halmstadt-u19-2025-08-16/parquet/ \
  data/
```

**2. Run identity alignment**

```bash
python alignment_method.py
```

Default paths are `data/wimu/` and `data/tracking_full.parquet`. Override with positional arguments:

```bash
python alignment_method.py data/wimu data/tracking_full.parquet -796.97
```

**3. Import as a library**

```python
from alignment_method import align_identities

result = align_identities(
    imu_dir="data/wimu",
    tracking_parquet="data/tracking_full.parquet",
    system_delta_s=-796.97,
)

print(result.assignments)       # {wimu_jersey: optical_jersey}
print(result.similarity_matrix) # (n_imu, n_optical) Pearson r matrix
```

---

## Dataset

### Session

Halmstadt U19 training session, 2025-08-16. 10 outfield players wearing WIMU
devices (jerseys 2, 3, 6, 7, 8, 9, 12, 14, 16, 17). Filmed with a 4-camera
Nexus rig. Session duration ~87 minutes (optical).

### Files

| Path | Rows | Description |
|---|---|---|
| `data/tracking_full.parquet` | 1.54 M | Full-session optical tracking, 10 FPS, 82 columns |
| `data/tracking.parquet` | 98 k | Short sub-session (~7.7 min), useful for quick tests |
| `data/wimu/jersey_N_imu.parquet` (×10) | varies | Per-player IMU: accel (sensor 300), gyro (sensor 302), attitude (sensor 301) merged on timecode |
| `data/wimu/jersey_N_gps.parquet` (×10) | varies | Per-player GPS (ground truth for time alignment validation, not used by the method) |
| `data/ground_truth_identity.parquet` | 10 | Verified identity mapping: jersey_number, track_id, raw_lag_s, delta_s, correlation_peak |

Total size: ~755 MB.

### Optical tracking columns (tracking_full.parquet)

| Column | Description |
|---|---|
| `session_time_s` | Seconds since optical session start |
| `track_id` | Optical track identifier (not stable across sessions) |
| `x`, `y` | World-space position (metres, pitch centre as origin) |
| `vx`, `vy` | World-space velocity (m/s) |
| `speed` | Scalar speed (m/s) |
| `team` | Team classifier (0 or 1) |
| `jersey_number` | Jersey classifier output (float, can be NaN) |
| `jersey_confidence` | Classifier confidence [0, 1] |
| 17× `kp_*_x`, `kp_*_y` | Pose keypoints in image pixels |

### WIMU IMU columns (jersey_N_imu.parquet)

| Column | Description |
|---|---|
| `session_time_s` | Seconds since earliest device timecode across all 10 devices |
| `timecode` | Raw device timecode (ms since epoch) |
| `accel_x/y/z` | Body-frame acceleration (g), ~300 Hz raw, ~88 Hz effective after merge |
| `gyro_x/y/z` | Body-frame angular velocity (deg/s) |
| `ac_earth_x/y/z` | Earth-frame acceleration (g, from attitude fusion) |
| `roll`, `pitch`, `yaw` | Euler angles (degrees) |

### Coordinate systems

- **Optical**: metres, pitch centre as origin, x along the long axis. World-space (camera calibration applied).
- **WIMU accel/gyro**: body-frame (device-fixed axes, orientation unknown).
- **WIMU attitude**: earth-frame (fusion output). `ac_earth_*` is gravity-compensated.
- **WIMU GPS**: WGS84 lat/lon. Coordinate transforms to pitch-space are in `config.toml` of the source repo (`aml-hp-data-visualiser`).

### Time alignment

The WIMU devices and the optical pipeline use independent clocks. The verified
offset is:

```
optical_session_time_s = wimu_session_time_s + (-796.97)
```

Determined by GPS-optical cross-correlation on 7 of 10 players; uncertainty
±0.2 s. The remaining 3 players (jerseys 2, 9, 16) are outliers — see Known
issues below.

### Known data issues

| Player | Issue |
|---|---|
| Jersey 2 | GPS cross-correlation is an outlier (delta = -1471 s vs. consensus -797 s). Optical/IMU overlap is limited; treat identity alignment result with caution. |
| Jersey 9 | WIMU device shutdown at ~53 min into the session. Data available only for the first half. |
| Jersey 14 | Optical jersey classifier produces confident misclassifications during some periods, contaminating the merged speed signal. |
| Jersey 16 | Optical detections only from ~31 min (late arrival or classifier failure). GPS cross-correlation is an outlier. |

---

## Method

The identity alignment method (E04b) matches each WIMU device to an
optically-tracked player using activity envelope correlation. The full
derivation, experiments, and results are in `docs/imu_optical_alignment_report.md`.

**Summary:**

1. Compute IMU activity envelope: `|sqrt(ax² + ay² + az²) - 1g|` binned to
   10 Hz, then 5-second rolling RMS.
2. Compute optical activity envelope: speed clipped to [0.5, 10.0] m/s,
   5-second rolling RMS.
3. Build a Pearson r similarity matrix over the time-aligned overlap window.
4. Solve the optimal assignment with the Hungarian algorithm.

Requires only a 3-axis accelerometer — no gyroscope, no sensor fusion, no
GPS. Achieves 7/10 correct assignments on this dataset (jerseys 9 and 16 fail
due to the data issues above; jersey 2 is marginal).

---

## Dependencies

```
pandas>=2.0
numpy
scipy
pyarrow
```

Optional (for rebuilding parquets from raw data):

```
zstandard          # raw tracking chunk loading (chunkstream_loader.py)
matplotlib         # diagnostic plots
pywimu             # raw .qul loading (Rust/PyO3, internal Hudl package)
```

Install:

```bash
pip install pandas numpy scipy pyarrow
```

---

## Repository structure

```
alignment_method.py     Reference implementation (importable + CLI)
data/                   Parquet data files (git-ignored; download from S3)
    tracking_full.parquet
    tracking.parquet
    ground_truth_identity.parquet
    wimu/
        jersey_N_imu.parquet
        jersey_N_gps.parquet
docs/
    imu_optical_alignment_report.md   Full findings report
    experiment_log.md                 Experiment tracker (E01–E09)
    gait_research_summary.md          Literature review: optical gait at 10 FPS
    tracker_notes.md                  BoT-SORT tracker internals
tools/
    upload_to_s3.sh       Upload data/ to S3 (see script for details)
    export_to_parquet.py  Rebuild parquets from raw source data
    chunkstream_loader.py Load raw zstd-compressed tracking chunks
    wimu_loader.py        Load raw WIMU .qul files
```

### Rebuilding the parquets

The parquets were built from:

- Raw optical tracking chunks (`.json.zst`) at `s3://hudl-experiments-v1/peter.swart/halmstadt-u19-2025-08-16/tracking/`
- Raw WIMU recordings (`.qul`) stored locally; originals on the WIMU device

`tools/export_to_parquet.py` contains the build script. It has hardcoded local
paths from the development environment — update `SCRATCH_DIR`, `REPO_DIR`, and
`WIMU_DIR` at the top of the file before running.
