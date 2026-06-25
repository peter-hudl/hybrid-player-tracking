"""
Unit tests for alignment_method.py — no real data files required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alignment_method import (
    OPTICAL_FPS,
    RMS_WINDOW_S,
    _bin_and_rms,
    _nan_pearson_r,
    build_similarity_matrix,
    extract_imu_envelope,
    extract_optical_envelope,
    load_and_merge_optical,
    solve_assignment,
)


# ── _nan_pearson_r ───────────────────────────────────────────────────────────

def test_pearson_identical():
    x = np.linspace(0, 1, 200)
    assert _nan_pearson_r(x, x) == pytest.approx(1.0, abs=1e-9)


def test_pearson_anticorrelated():
    x = np.linspace(0, 1, 200)
    assert _nan_pearson_r(x, -x) == pytest.approx(-1.0, abs=1e-9)


def test_pearson_all_nan():
    x = np.full(200, np.nan)
    assert _nan_pearson_r(x, x) == 0.0


def test_pearson_too_few_valid():
    x = np.full(200, np.nan)
    x[:50] = np.linspace(0, 1, 50)   # < MIN_CORR_SAMPLES=100
    assert _nan_pearson_r(x, x) == 0.0


def test_pearson_constant_series():
    # zero std -> returns 0.0
    x = np.ones(200)
    y = np.linspace(0, 1, 200)
    assert _nan_pearson_r(x, y) == 0.0


# ── _bin_and_rms ─────────────────────────────────────────────────────────────

def _make_t_grid(duration_s: float = 60.0) -> np.ndarray:
    return np.arange(0.0, duration_s, 1.0 / OPTICAL_FPS)


def test_bin_and_rms_output_shape():
    t_grid = _make_t_grid(60.0)
    rng = np.random.default_rng(0)
    times = np.sort(rng.uniform(0.0, 60.0, 6000))
    values = rng.normal(0.5, 0.1, len(times))
    out = _bin_and_rms(values, times, t_grid)
    assert out.shape == t_grid.shape


def test_bin_and_rms_positive_interior():
    t_grid = _make_t_grid(60.0)
    rng = np.random.default_rng(1)
    times = np.sort(rng.uniform(0.0, 60.0, 6000))
    values = rng.uniform(0.1, 1.0, len(times))
    out = _bin_and_rms(values, times, t_grid)
    rms_w = int(RMS_WINDOW_S * OPTICAL_FPS)
    # Interior (past the rolling window edges) should be positive and finite
    interior = out[rms_w:-rms_w]
    assert np.all(np.isfinite(interior))
    assert np.all(interior > 0)


def test_bin_and_rms_no_nan_interior():
    t_grid = _make_t_grid(60.0)
    rng = np.random.default_rng(2)
    times = np.sort(rng.uniform(5.0, 55.0, 6000))
    values = np.abs(rng.normal(0.3, 0.05, len(times)))
    out = _bin_and_rms(values, times, t_grid)
    rms_w = int(RMS_WINDOW_S * OPTICAL_FPS)
    interior = out[rms_w:-rms_w]
    assert not np.any(np.isnan(interior))


# ── extract_optical_envelope ─────────────────────────────────────────────────

def _make_speed_df(n: int = 3600, rng_seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(rng_seed)
    t = np.sort(rng.uniform(0.0, 360.0, n))
    speed = rng.uniform(1.0, 8.0, n)
    return pd.DataFrame({"session_time_s": t, "speed": speed})


def test_extract_optical_envelope_shape():
    df = _make_speed_df()
    t_grid = _make_t_grid(360.0)
    out = extract_optical_envelope(df, 0.0, 360.0, t_grid)
    assert out.shape == t_grid.shape


def test_extract_optical_envelope_clips_high_speed():
    rng = np.random.default_rng(3)
    n = 3600
    t = np.sort(rng.uniform(0.0, 360.0, n))
    speed = rng.uniform(12.0, 20.0, n)  # all above clip max of 10.0
    df = pd.DataFrame({"session_time_s": t, "speed": speed})
    t_grid = _make_t_grid(360.0)
    out = extract_optical_envelope(df, 0.0, 360.0, t_grid)
    finite = out[np.isfinite(out)]
    assert len(finite) > 0
    # RMS of clipped-to-10 values should never exceed 10
    assert np.all(finite <= 10.0 + 1e-9)


def test_extract_optical_envelope_too_few_rows():
    df = pd.DataFrame({"session_time_s": [1.0, 2.0], "speed": [3.0, 4.0]})
    t_grid = _make_t_grid(60.0)
    out = extract_optical_envelope(df, 0.0, 60.0, t_grid)
    assert np.all(np.isnan(out))


# ── extract_imu_envelope ─────────────────────────────────────────────────────

def _write_stationary_imu(path, n: int = 5000, fs: float = 100.0):
    """Write a parquet file simulating a stationary sensor (accel ≈ [0,0,1g])."""
    t = np.arange(n) / fs
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.02, n)
    df = pd.DataFrame({
        "session_time_s": t,
        "accel_x": noise,
        "accel_y": noise * 0.5,
        "accel_z": 1.0 + noise,
    })
    df.to_parquet(path, index=False)


def test_extract_imu_envelope_stationary_near_zero(tmp_path):
    p = tmp_path / "jersey_1_imu.parquet"
    _write_stationary_imu(p, n=5000, fs=100.0)
    duration = 5000 / 100.0
    t_grid = np.arange(0.0, duration, 1.0 / OPTICAL_FPS)
    out = extract_imu_envelope(p, 0.0, duration, t_grid)
    assert out.shape == t_grid.shape
    finite = out[np.isfinite(out)]
    assert len(finite) > 0
    # stationary sensor: |accel| - 1g ≈ 0; RMS should be small
    assert np.nanmean(finite) < 0.15


def test_extract_imu_envelope_shape(tmp_path):
    p = tmp_path / "jersey_2_imu.parquet"
    _write_stationary_imu(p)
    duration = 5000 / 100.0
    t_grid = np.arange(0.0, duration, 1.0 / OPTICAL_FPS)
    out = extract_imu_envelope(p, 0.0, duration, t_grid)
    assert out.shape == t_grid.shape


# ── build_similarity_matrix ──────────────────────────────────────────────────

def test_build_similarity_matrix_diagonal_dominates():
    rng = np.random.default_rng(7)
    n = 500
    # Three distinct signals
    signals = {
        1: rng.normal(0, 1, n),
        2: rng.normal(5, 1, n),
        3: np.sin(np.linspace(0, 10, n)),
    }
    imu_env = {j: s for j, s in signals.items()}
    opt_env = {j: s + rng.normal(0, 0.01, n) for j, s in signals.items()}

    sim, imu_j, opt_j = build_similarity_matrix(imu_env, opt_env)

    assert sim.shape == (3, 3)
    for i, ij in enumerate(imu_j):
        opt_idx = opt_j.index(ij)
        row = sim[i].copy()
        row[opt_idx] = -np.inf
        assert sim[i, opt_idx] > row.max(), (
            f"Jersey {ij}: diagonal {sim[i, opt_idx]:.3f} should dominate off-diagonal {row.max():.3f}"
        )


# ── solve_assignment ─────────────────────────────────────────────────────────

def test_solve_assignment_identity():
    sim = np.eye(4) * 0.9 + 0.05
    jerseys = [1, 2, 3, 4]
    assignments = solve_assignment(sim, jerseys, jerseys)
    assert assignments == {1: 1, 2: 2, 3: 3, 4: 4}


def test_solve_assignment_non_square():
    # More optical players than IMU devices
    imu_j = [1, 2]
    opt_j = [1, 2, 3]
    sim = np.array([[0.9, 0.2, 0.1],
                    [0.1, 0.8, 0.2]])
    assignments = solve_assignment(sim, imu_j, opt_j)
    assert set(assignments.keys()) == {1, 2}
    assert assignments[1] == 1
    assert assignments[2] == 2


# ── load_and_merge_optical ───────────────────────────────────────────────────

def _write_tracking_parquet(path, n_high_conf: int = 200, n_low_conf: int = 100):
    """Synthetic tracking parquet with two jersey numbers and mixed confidence."""
    rng = np.random.default_rng(99)
    rows = []
    t_base = 0.0
    for jersey in [7, 11]:
        # High-confidence rows
        for i in range(n_high_conf):
            rows.append({
                "session_time_s": t_base + i * 0.1,
                "speed": rng.uniform(1.0, 5.0),
                "jersey_number": float(jersey),
                "jersey_confidence": rng.uniform(0.96, 1.0),
                "team": 0.0,
            })
        # Low-confidence rows (extra unique times to avoid dedup with high-conf)
        for i in range(n_low_conf):
            rows.append({
                "session_time_s": t_base + n_high_conf * 0.1 + i * 0.1,
                "speed": rng.uniform(0.5, 3.0),
                "jersey_number": float(jersey),
                "jersey_confidence": rng.uniform(0.05, 0.15),
                "team": 0.0,
            })
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_load_and_merge_high_conf_fewer_rows(tmp_path):
    p = tmp_path / "tracking.parquet"
    _write_tracking_parquet(p)

    players_low = load_and_merge_optical(p, conf_min=0.1)
    players_high = load_and_merge_optical(p, conf_min=0.95)

    for jersey in [7, 11]:
        assert jersey in players_low
        assert jersey in players_high
        assert len(players_high[jersey]) < len(players_low[jersey]), (
            f"Jersey {jersey}: high-conf should return fewer rows"
        )


def test_load_and_merge_deduplication(tmp_path):
    """Duplicate session_time_s rows are removed (highest conf kept)."""
    rows = [
        {"session_time_s": 1.0, "speed": 3.0, "jersey_number": 5.0, "jersey_confidence": 0.9, "team": 0.0},
        {"session_time_s": 1.0, "speed": 2.0, "jersey_number": 5.0, "jersey_confidence": 0.5, "team": 0.0},
        {"session_time_s": 2.0, "speed": 4.0, "jersey_number": 5.0, "jersey_confidence": 0.8, "team": 0.0},
    ]
    p = tmp_path / "tracking.parquet"
    pd.DataFrame(rows).to_parquet(p, index=False)

    players = load_and_merge_optical(p, conf_min=0.1)
    assert 5 in players
    df = players[5]
    # session_time_s=1.0 should appear exactly once
    assert (df["session_time_s"] == 1.0).sum() == 1
    assert len(df) == 2
