"""
IMU-Optical Identity Alignment: Reference Implementation

Matches anonymous IMU wearable devices to optically-tracked football players
using activity-envelope correlation. Requires only a 3-axis accelerometer
(no gyroscope or sensor fusion) and optical speed at 10 FPS.

Method (E04b):
  IMU:     |sqrt(ax^2 + ay^2 + az^2)| - 1g  ->  5s rolling RMS
  Optical: speed clipped to [0.5, 10.0] m/s  ->  5s rolling RMS
  Match:   Pearson r on aligned envelopes, Hungarian assignment

Usage:
    from alignment_method import align_identities

    result = align_identities(
        imu_dir="path/to/wimu/parquet/",
        tracking_parquet="path/to/tracking_full.parquet",
        system_delta_s=-796.97,
    )
    # result.assignments: dict[int, int]  (wimu_jersey -> optical_jersey)
    # result.similarity_matrix: np.ndarray
    # result.jerseys_imu / result.jerseys_optical: label lists
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


# ── Defaults ─────────────────────────────────────────────────────────────────

OPTICAL_FPS = 10.0
OPTICAL_TEAM = 0
JERSEY_CONF_MIN = 0.1
SPEED_CLIP = (0.5, 10.0)
ACCEL_CLIP_G = 5.0
RMS_WINDOW_S = 5.0
MIN_VALID_FRAC = 0.10
MIN_CORR_SAMPLES = 100


# ── Result container ─────────────────────────────────────────────────────────

@dataclass
class AlignmentResult:
    assignments: dict[int, int]
    similarity_matrix: np.ndarray
    jerseys_imu: list[int]
    jerseys_optical: list[int]


# ── Optical track merging ────────────────────────────────────────────────────

def load_and_merge_optical(
    tracking_parquet: str | Path,
    team: int = OPTICAL_TEAM,
    conf_min: float = JERSEY_CONF_MIN,
) -> dict[int, pd.DataFrame]:
    """
    Load optical tracking and merge fragmented tracks per player using
    the jersey number classifier.

    Returns dict: jersey_number -> DataFrame[session_time_s, speed]
    """
    df = pd.read_parquet(
        tracking_parquet,
        columns=["session_time_s", "speed", "jersey_number",
                 "jersey_confidence", "team"],
    )
    sub = df[(df["team"] == float(team)) & (df["jersey_confidence"] > conf_min)]

    result: dict[int, pd.DataFrame] = {}
    for j in sub["jersey_number"].dropna().unique():
        j_int = int(j)
        jdf = sub[sub["jersey_number"] == j][
            ["session_time_s", "speed", "jersey_confidence"]
        ].copy()
        if len(jdf) == 0:
            continue
        jdf = (
            jdf.sort_values("jersey_confidence", ascending=False)
               .drop_duplicates(subset="session_time_s")
               .sort_values("session_time_s")
               .reset_index(drop=True)
        )
        result[j_int] = jdf[["session_time_s", "speed"]]
    return result


# ── IMU envelope extraction ──────────────────────────────────────────────────

def extract_imu_envelope(
    imu_parquet: str | Path,
    t_start_wimu: float,
    t_end_wimu: float,
    t_grid: np.ndarray,
) -> np.ndarray:
    """
    Extract activity envelope from raw accelerometer data (E04b method).

    Computes |accel_magnitude| - 1g, bins onto the 10 Hz optical time grid,
    and applies a 5-second rolling RMS.

    Parameters
    ----------
    imu_parquet : path to jersey_N_imu.parquet
    t_start_wimu, t_end_wimu : WIMU session time bounds
    t_grid : regular 10 Hz time grid in optical session time

    Returns
    -------
    np.ndarray of shape (len(t_grid),), NaN where no data.
    """
    imu = pd.read_parquet(imu_parquet)
    win = imu[
        (imu["session_time_s"] >= t_start_wimu)
        & (imu["session_time_s"] < t_end_wimu)
    ].copy()

    if len(win) < 50:
        return np.full(len(t_grid), np.nan)

    win = win.sort_values("session_time_s")

    ax = win["accel_x"].clip(-ACCEL_CLIP_G, ACCEL_CLIP_G).values
    ay = win["accel_y"].clip(-ACCEL_CLIP_G, ACCEL_CLIP_G).values
    az = win["accel_z"].clip(-ACCEL_CLIP_G, ACCEL_CLIP_G).values
    mag = np.sqrt(ax**2 + ay**2 + az**2) - 1.0

    opt_t = win["session_time_s"].values + (t_grid[0] - t_start_wimu)
    return _bin_and_rms(mag, opt_t, t_grid)


# ── Optical envelope extraction ──────────────────────────────────────────────

def extract_optical_envelope(
    player_df: pd.DataFrame,
    t_start_opt: float,
    t_end_opt: float,
    t_grid: np.ndarray,
) -> np.ndarray:
    """
    Extract activity envelope from merged optical player speed.

    Clips speed to [0.5, 10.0] m/s and applies a 5-second rolling RMS.

    Parameters
    ----------
    player_df : DataFrame with session_time_s and speed columns
    t_start_opt, t_end_opt : optical session time bounds
    t_grid : regular 10 Hz time grid

    Returns
    -------
    np.ndarray of shape (len(t_grid),), NaN where no data.
    """
    win = player_df[
        (player_df["session_time_s"] >= t_start_opt)
        & (player_df["session_time_s"] < t_end_opt)
    ].copy()

    if len(win) < 50:
        return np.full(len(t_grid), np.nan)

    win = win.sort_values("session_time_s")
    spd = win["speed"].clip(SPEED_CLIP[0], SPEED_CLIP[1]).values
    t = win["session_time_s"].values
    return _bin_and_rms(spd, t, t_grid)


# ── Similarity and assignment ────────────────────────────────────────────────

def build_similarity_matrix(
    imu_envelopes: dict[int, np.ndarray],
    optical_envelopes: dict[int, np.ndarray],
) -> tuple[np.ndarray, list[int], list[int]]:
    """
    Compute pairwise Pearson r between IMU and optical envelopes.

    Returns (matrix, imu_jersey_list, optical_jersey_list).
    """
    imu_jerseys = sorted(imu_envelopes.keys())
    opt_jerseys = sorted(optical_envelopes.keys())

    sim = np.zeros((len(imu_jerseys), len(opt_jerseys)))
    for i, ij in enumerate(imu_jerseys):
        for j, oj in enumerate(opt_jerseys):
            sim[i, j] = _nan_pearson_r(
                imu_envelopes[ij], optical_envelopes[oj]
            )
    return sim, imu_jerseys, opt_jerseys


def solve_assignment(
    similarity: np.ndarray,
    imu_jerseys: list[int],
    optical_jerseys: list[int],
) -> dict[int, int]:
    """
    Hungarian algorithm on the similarity matrix.

    Returns dict: wimu_jersey -> optical_jersey.
    """
    cost = 1.0 - similarity
    row_ind, col_ind = linear_sum_assignment(cost)
    return {imu_jerseys[r]: optical_jerseys[c] for r, c in zip(row_ind, col_ind)}


# ── Top-level function ───────────────────────────────────────────────────────

def align_identities(
    imu_dir: str | Path,
    tracking_parquet: str | Path,
    system_delta_s: float,
    t_start_opt: float | None = None,
    t_end_opt: float | None = None,
    jerseys: list[int] | None = None,
) -> AlignmentResult:
    """
    Run the full identity alignment pipeline.

    Parameters
    ----------
    imu_dir : directory containing jersey_N_imu.parquet files
    tracking_parquet : path to tracking_full.parquet
    system_delta_s : time offset (optical_time = wimu_time + system_delta_s)
    t_start_opt, t_end_opt : analysis window in optical session time.
        If None, uses the full overlapping range.
    jerseys : specific jersey numbers to include. If None, auto-detects
        from available parquet files.

    Returns
    -------
    AlignmentResult with assignments, similarity matrix, and jersey lists.
    """
    imu_dir = Path(imu_dir)
    tracking_parquet = Path(tracking_parquet)

    # Discover available jerseys
    if jerseys is None:
        jerseys = sorted(
            int(p.stem.split("_")[1].split("_")[0])
            for p in imu_dir.glob("jersey_*_imu.parquet")
        )

    # Load and merge optical tracks
    optical_players = load_and_merge_optical(tracking_parquet)
    available_optical = sorted(set(jerseys) & set(optical_players.keys()))

    # Determine analysis window
    if t_start_opt is None or t_end_opt is None:
        opt_times = [
            optical_players[j]["session_time_s"]
            for j in available_optical if j in optical_players
        ]
        t_start_opt = min(s.min() for s in opt_times)
        t_end_opt = max(s.max() for s in opt_times)

    # Build 10 Hz time grid
    t_grid = np.arange(t_start_opt, t_end_opt, 1.0 / OPTICAL_FPS)

    # Convert window to WIMU time
    t_start_wimu = t_start_opt - system_delta_s
    t_end_wimu = t_end_opt - system_delta_s

    # Extract envelopes
    imu_envelopes: dict[int, np.ndarray] = {}
    for j in jerseys:
        imu_path = imu_dir / f"jersey_{j}_imu.parquet"
        if imu_path.exists():
            env = extract_imu_envelope(imu_path, t_start_wimu, t_end_wimu, t_grid)
            if np.nansum(np.isfinite(env)) > MIN_CORR_SAMPLES:
                imu_envelopes[j] = env

    optical_envelopes: dict[int, np.ndarray] = {}
    for j in available_optical:
        env = extract_optical_envelope(
            optical_players[j], t_start_opt, t_end_opt, t_grid
        )
        if np.nansum(np.isfinite(env)) > MIN_CORR_SAMPLES:
            optical_envelopes[j] = env

    # Build similarity matrix and solve
    sim, imu_j, opt_j = build_similarity_matrix(imu_envelopes, optical_envelopes)
    assignments = solve_assignment(sim, imu_j, opt_j)

    return AlignmentResult(
        assignments=assignments,
        similarity_matrix=sim,
        jerseys_imu=imu_j,
        jerseys_optical=opt_j,
    )


# ── Internal helpers ─────────────────────────────────────────────────────────

def _bin_and_rms(
    values: np.ndarray,
    times: np.ndarray,
    t_grid: np.ndarray,
) -> np.ndarray:
    """Bin values onto 10 Hz grid, interpolate small gaps, apply rolling RMS."""
    bin_edges = np.append(t_grid, t_grid[-1] + 1.0 / OPTICAL_FPS)
    bin_idx = np.searchsorted(bin_edges, times, side="right") - 1
    valid = (bin_idx >= 0) & (bin_idx < len(t_grid))

    sq_sum = np.zeros(len(t_grid))
    counts = np.zeros(len(t_grid), dtype=int)
    for ib, v in zip(bin_idx[valid], values[valid]):
        sq_sum[ib] += v**2
        counts[ib] += 1

    binned = np.full(len(t_grid), np.nan)
    has = counts > 0
    binned[has] = np.sqrt(sq_sum[has] / counts[has])

    s = pd.Series(binned, index=t_grid).interpolate(method="linear", limit=20)

    rms_w = max(1, int(RMS_WINDOW_S * OPTICAL_FPS))
    s_rms = s.rolling(rms_w, center=True, min_periods=rms_w // 2).apply(
        lambda x: np.sqrt(np.nanmean(x**2)), raw=True
    )
    return s_rms.values


def _nan_pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r over jointly non-NaN samples. Returns 0.0 if insufficient data."""
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < MIN_CORR_SAMPLES:
        return 0.0
    x, y = a[valid], b[valid]
    mx, my = np.nanmean(x), np.nanmean(y)
    sx, sy = np.nanstd(x), np.nanstd(y)
    if sx < 1e-9 or sy < 1e-9:
        return 0.0
    return float(np.nanmean((x - mx) * (y - my)) / (sx * sy))


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    IMU_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/wimu")
    TRACKING = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/tracking_full.parquet")
    DELTA = float(sys.argv[3]) if len(sys.argv) > 3 else -796.97

    print(f"IMU dir:  {IMU_DIR}")
    print(f"Tracking: {TRACKING}")
    print(f"Delta:    {DELTA}s")
    print()

    result = align_identities(
        imu_dir=IMU_DIR,
        tracking_parquet=TRACKING,
        system_delta_s=DELTA,
    )

    print("=== Identity Alignment ===")
    print(f"{'WIMU jersey':<14} {'Assigned to':<14} {'Correct?'}")
    for ij in result.jerseys_imu:
        oj = result.assignments.get(ij, None)
        correct = "YES" if ij == oj else "no"
        print(f"  {ij:<12} opt_{oj:<10} {correct}")

    n_correct = sum(1 for ij, oj in result.assignments.items() if ij == oj)
    print(f"\nAccuracy: {n_correct}/{len(result.assignments)}")
