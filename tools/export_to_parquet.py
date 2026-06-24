"""
export_to_parquet.py

Phase 0.3: Export all tracking and WIMU data to parquet files.

Outputs:
    parquet/tracking.parquet
    parquet/wimu/jersey_N_gps.parquet   (N in [2, 3, 6, 7, 8, 9, 12, 14, 16, 17])
    parquet/wimu/jersey_N_imu.parquet
"""

import sys
from pathlib import Path

# ── Path setup ─────────────────────────────────────────────────────────────────
SCRATCH_DIR = Path('/Users/peter.swart/scratch/aml-hp-data-visualiser/IMU-Opt')
REPO_DIR    = Path('/Users/peter.swart/Documents/GitHub/aml-hp-data-visualiser')
WIMU_DIR    = Path('/Users/peter.swart/Documents/data_visualiser_repo_data_backup/Halmstadt_Dataset/wimu_data')
TRACKING_DIR = SCRATCH_DIR / 'tracking'
PARQUET_DIR  = SCRATCH_DIR / 'parquet'

sys.path.insert(0, str(SCRATCH_DIR))
sys.path.insert(0, str(REPO_DIR))

import pandas as pd
from chunkstream_loader import load_chunkstream
from src.wimu_loader import load_wimu_gps, load_wimu_imu

JERSEYS = [2, 3, 6, 7, 8, 9, 12, 14, 16, 17]
TIMEZONE = 'UTC+2'


def export_tracking(parquet_dir: Path) -> pd.DataFrame:
    print("Loading optical tracking chunkstream...")
    df = load_chunkstream(TRACKING_DIR)
    out_path = parquet_dir / 'tracking.parquet'
    df.to_parquet(out_path, index=False)
    print(f"  Wrote {out_path.name}: {df.shape[0]:,} rows, {df.shape[1]} cols, "
          f"time {df.session_time_s.min():.1f}–{df.session_time_s.max():.1f}s")
    return df


def find_global_t0() -> int:
    """
    Load sensor 301 timecodes for all 10 jerseys and return the minimum
    first timecode (ms) across all devices.
    """
    print("\nDetermining global t0 across all WIMU devices (sensor 301)...")
    first_timecodes = {}

    for jersey in JERSEYS:
        fp = WIMU_DIR / f'jersey_{jersey}.qul'
        imu = load_wimu_imu(fp, timezone=TIMEZONE, sensors=[301])
        if 301 not in imu or imu[301].empty:
            raise RuntimeError(f"Could not load sensor 301 for jersey {jersey}")
        tc0 = int(imu[301]['timecode'].iloc[0])
        first_timecodes[jersey] = tc0

    t0 = min(first_timecodes.values())
    spread_ms = max(first_timecodes.values()) - t0

    print(f"  Per-device first timecodes (ms since epoch):")
    for jersey in JERSEYS:
        offset_ms = first_timecodes[jersey] - t0
        print(f"    jersey_{jersey}: {first_timecodes[jersey]}  (+{offset_ms} ms from t0)")
    print(f"  t0 = {t0} ms  |  spread = {spread_ms} ms ({spread_ms/1000:.3f} s)")

    if spread_ms > 5000:
        print(f"  WARNING: spread of {spread_ms} ms exceeds 5 s — devices may not be synchronised")

    return t0


def merge_imu(imu_dict: dict) -> pd.DataFrame:
    """
    Merge sensors 300 (accel), 302 (gyro), 301 (attitude) on timecode.

    All three sensors are expected to have identical timecodes. We do an outer
    join so any small discrepancies are preserved rather than silently dropped.
    After the merge, only one 'timestamp' column is kept.
    """
    df_accel = imu_dict[300].rename(columns={'timestamp': 'timestamp_300'})
    df_gyro  = imu_dict[302].rename(columns={'timestamp': 'timestamp_302'})
    df_att   = imu_dict[301]  # keep timestamp from attitude sensor

    # Merge accel + gyro on timecode
    df = df_accel.merge(df_gyro, on='timecode', how='outer')
    # Merge in attitude; its timestamp column becomes the single 'timestamp'
    df = df.merge(df_att, on='timecode', how='outer')

    # Drop the extra timestamp columns from accel/gyro sensors
    df = df.drop(columns=['timestamp_300', 'timestamp_302'])

    # Sort by timecode to restore chronological order after outer join
    df = df.sort_values('timecode').reset_index(drop=True)

    return df


def export_wimu(parquet_dir: Path, t0: int) -> None:
    wimu_out = parquet_dir / 'wimu'
    wimu_out.mkdir(exist_ok=True)

    for jersey in JERSEYS:
        fp = WIMU_DIR / f'jersey_{jersey}.qul'
        print(f"\n  jersey_{jersey}:")

        # ── GPS ────────────────────────────────────────────────────────────────
        df_gps = load_wimu_gps(fp, timezone=TIMEZONE)
        if df_gps.empty:
            print(f"    WARNING: GPS empty for jersey {jersey}")
        else:
            df_gps['session_time_s'] = (df_gps['timecode'] - t0) / 1000.0
            out = wimu_out / f'jersey_{jersey}_gps.parquet'
            df_gps.to_parquet(out, index=False)
            print(f"    GPS: {df_gps.shape[0]:,} rows  "
                  f"time {df_gps.session_time_s.min():.1f}–{df_gps.session_time_s.max():.1f}s  "
                  f"-> {out.name}")

        # ── IMU ────────────────────────────────────────────────────────────────
        imu_dict = load_wimu_imu(fp, timezone=TIMEZONE)
        missing = [s for s in [300, 302, 301] if s not in imu_dict]
        if missing:
            print(f"    WARNING: Missing IMU sensors {missing} for jersey {jersey}")

        df_imu = merge_imu(imu_dict)
        df_imu['session_time_s'] = (df_imu['timecode'] - t0) / 1000.0
        out = wimu_out / f'jersey_{jersey}_imu.parquet'
        df_imu.to_parquet(out, index=False)
        print(f"    IMU: {df_imu.shape[0]:,} rows  "
              f"time {df_imu.session_time_s.min():.1f}–{df_imu.session_time_s.max():.1f}s  "
              f"-> {out.name}")


def main() -> None:
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Optical tracking
    export_tracking(PARQUET_DIR)

    # 2. Global t0 for WIMU session_time_s
    t0 = find_global_t0()

    # 3. WIMU GPS + IMU for each player
    print("\nExporting WIMU data...")
    export_wimu(PARQUET_DIR, t0)

    # 4. Verification summary
    print("\n" + "=" * 60)
    print("Verification summary")
    print("=" * 60)

    t = pd.read_parquet(PARQUET_DIR / 'tracking.parquet')
    print(f"tracking.parquet: {t.shape}, "
          f"time {t.session_time_s.min():.1f}–{t.session_time_s.max():.1f}s")

    for jersey in JERSEYS:
        gps = pd.read_parquet(PARQUET_DIR / 'wimu' / f'jersey_{jersey}_gps.parquet')
        imu = pd.read_parquet(PARQUET_DIR / 'wimu' / f'jersey_{jersey}_imu.parquet')
        ok_cols = all(c in imu.columns for c in ['accel_x', 'gyro_x', 'ac_earth_x', 'session_time_s'])
        print(f"jersey_{jersey}: GPS={gps.shape}, IMU={imu.shape}, "
              f"IMU time {imu.session_time_s.min():.1f}–{imu.session_time_s.max():.1f}s, "
              f"cols_ok={ok_cols}")


if __name__ == '__main__':
    main()
