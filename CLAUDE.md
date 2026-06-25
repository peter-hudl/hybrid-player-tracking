# hybrid-player-tracking

## Project purpose

Research repo for IMU-optical identity alignment in football. Matches anonymous
IMU wearables to optically-tracked players using activity-envelope correlation.
Applied to the Halmstadt U19 dataset (2025-08-16).

The broader goal is to determine whether IMU-only wearables plus optical tracking
can replace GPS/UWB for player tracking. This repo handles the identity alignment
step; downstream fusion (Kalman filter combining IMU + optical) is out of scope here.

## Design constraints

- **Device-agnostic IMU pipeline.** Use only raw 3-axis accelerometer
  (`accel_x/y/z`). Do not use `AC_Earth_X/Y/Z` (gravity-subtracted earth-frame
  accelerations from sensor 301) even though they are available in the WIMU
  parquets. Some wearables (e.g., Titan) may not provide attitude-derived channels.
  The pipeline must work with the lowest common denominator: raw accelerometer only.

- **Optical feature: speed only.** Optical acceleration (`ax/ay` from the tracker)
  is unusable (87% tracking jitter). Jersey confidence is uncalibrated (0.95 does
  not mean 95% correct); use `conf > 0.1` as a noise floor, not a quality filter.

- **10 FPS optical is below gait analysis threshold.** Literature requires 25-30 FPS
  for reliable stride/gait extraction. All gait-based approaches have been tried and
  failed (see `docs/experiment_log.md` E03, E06-E09). Do not retry without higher
  frame rate data.

- **Gyroscope is redundant at multi-second windows.** At 2-5s rolling aggregation,
  accelerometer captures rotational accelerations too. Gyro adds value only at
  sub-second timescales.

## Key technical facts

- **Time alignment (Halmstadt session):** `optical_time = wimu_session_time + (-796.97)`.
  Consensus from 7/10 players. Any new session needs its own time alignment.
- **IMU method (E04b):** `sqrt(ax^2 + ay^2 + az^2) - 1g`, rolling RMS, Pearson r
  similarity, Hungarian assignment. Achieves 8/9 on the full 70-min session.
- **Jersey 14 failure:** Persistent optical misclassification (ghost track). Not
  fixable at the alignment level.
- **10 players:** Jerseys 2, 3, 6, 7, 8, 9, 12, 14, 16, 17. All should be
  included in evaluation. Jerseys 2, 9, 16 have unreliable time alignment (outlier
  GPS cross-correlation); treat their assignment results with caution but still
  include them.

## Data

Parquet files live in `data/` (git-ignored). Download from S3:
`s3://hudlrd-datasets/focus_nexus/halmstadt-u19-2025-08-16/parquet/`

Do not commit data files to git. S3 is the canonical store.

## Conventions

- **Time-based parameters in time units.** Express window sizes, bin widths, and
  downsampling steps in seconds/minutes, not sample counts. Use polars duration
  string convention: `"2s"`, `"5s"`, `"1m"`.
- **Time-aware windowing.** Use polars `group_by_dynamic` (or equivalent) with the
  timestamp column as index for binning and rolling operations. Do not use
  fixed-sample-count windows on data with gaps.
- **NaN-aware throughout.** Expect sensor dropouts, track fragmentation, and
  misaligned joins. Use `np.nanmean`, `np.nanstd`, `skipna=True`, etc.
- **Timestamp as logical index.** Always join, resample, and compare across datasets
  using the `session_time_s` column, never default integer indices.

## Tooling

- Python >= 3.10, managed with `uv`
- Dependencies: see `pyproject.toml`
- Tests: `uv run pytest tests/ -v`
- Integration tests require data files; marked with `@pytest.mark.requires_data`

## Repo structure

```
alignment_method.py          Main pipeline (importable + CLI)
tools/
  plot_similarity.py         Similarity matrix heatmap
  export_to_parquet.py       Rebuild parquets from raw source data
  chunkstream_loader.py      Load raw zstd-compressed tracking chunks
  wimu_loader.py             Load raw WIMU .qul files
  upload_to_s3.sh            Upload data/ to S3
docs/
  followup_work.md           Planned upgrades and known limitations
  imu_optical_alignment_report.md   Full findings report
  experiment_log.md          Experiments E01-E09 with results
  gait_research_summary.md   Literature review on optical gait at 10 FPS
  tracker_notes.md           BoT-SORT tracker internals
tests/
  test_unit.py               Unit tests (synthetic data, no parquets needed)
  test_integration.py        Integration tests (requires data/)
  conftest.py                Shared fixtures and snapshot support
  snapshots/                 Baseline assignment + similarity snapshots
```

## Dead ends (do not retry)

See `docs/experiment_log.md` for full details:
- Heading change rate (Feature B): zero signal at 10 FPS
- Ankle separation stride frequency (E03): 10 FPS too coarse
- Speed-cadence R2 matching (E06): no short-window improvement
- A-S fingerprinting (E07): optical profiles indistinguishable
- Autocorrelation stride frequency (E08): worse than FFT
- Hip world-space oscillation (E09): position noise drowns stride signal
- Confidence threshold filtering (E05): ghost-track detections are high-confidence
