"""
Snapshot tests for align_identities — requires data/tracking_full.parquet.

Run with --snapshot-update to generate/refresh snapshots:
    pytest tests/test_integration.py --snapshot-update -v

Normal run asserts results match the stored snapshots:
    pytest tests/test_integration.py -v

The comparison test (test_conf_comparison) only uses snapshots, so it runs
even without data/ present once snapshots have been generated.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from alignment_method import align_identities

DATA_DIR = Path(__file__).parent.parent / "data"
IMU_DIR = DATA_DIR / "wimu"
TRACKING_PARQUET = DATA_DIR / "tracking_full.parquet"
SYSTEM_DELTA_S = -796.97

SNAPSHOTS_DIR = Path(__file__).parent / "snapshots"


# ── helpers ──────────────────────────────────────────────────────────────────

def _snapshots_exist(*names: str) -> bool:
    return all(
        (SNAPSHOTS_DIR / f"{n}.json").exists()
        and (SNAPSHOTS_DIR / f"{n}_sim.npy").exists()
        for n in names
    )


# ── baseline snapshot test (conf_min=0.1) ────────────────────────────────────

@pytest.mark.requires_data
def test_baseline_full_session(snapshot):
    result = align_identities(
        imu_dir=IMU_DIR,
        tracking_parquet=TRACKING_PARQUET,
        system_delta_s=SYSTEM_DELTA_S,
        conf_min=0.1,
    )
    snapshot.assert_assignments("baseline", result.assignments)
    snapshot.assert_similarity("baseline", result.similarity_matrix)


# ── high-confidence snapshot test (conf_min=0.95) ────────────────────────────

@pytest.mark.requires_data
def test_high_conf_full_session(snapshot):
    result = align_identities(
        imu_dir=IMU_DIR,
        tracking_parquet=TRACKING_PARQUET,
        system_delta_s=SYSTEM_DELTA_S,
        conf_min=0.95,
    )
    snapshot.assert_assignments("high_conf", result.assignments)
    snapshot.assert_similarity("high_conf", result.similarity_matrix)


# ── comparison diagnostic (uses snapshots only) ───────────────────────────────

@pytest.mark.skipif(
    not _snapshots_exist("baseline", "high_conf"),
    reason="Run --snapshot-update first to generate baseline and high_conf snapshots",
)
def test_conf_comparison(snapshot):
    """
    Diagnostic: print where conf_min=0.1 and conf_min=0.95 disagree.
    No assertion — always passes; inspect stdout with -s.
    """
    baseline_asgn = snapshot.load_assignments("baseline")
    high_conf_asgn = snapshot.load_assignments("high_conf")
    baseline_sim = snapshot.load_similarity("baseline")
    high_conf_sim = snapshot.load_similarity("high_conf")

    # Divergent assignments
    all_imu = sorted(set(baseline_asgn) | set(high_conf_asgn))
    divergent = [
        j for j in all_imu
        if baseline_asgn.get(j) != high_conf_asgn.get(j)
    ]

    print("\n=== conf_min comparison ===")
    if divergent:
        print(f"\nDivergent assignments ({len(divergent)}/{len(all_imu)} jerseys):")
        print(f"  {'IMU jersey':<12} {'baseline(0.1)':<16} {'high_conf(0.95)'}")
        for j in divergent:
            b = baseline_asgn.get(j, "–")
            h = high_conf_asgn.get(j, "–")
            print(f"  {j:<12} {str(b):<16} {h}")
    else:
        print("\nAssignments identical across both thresholds.")

    # Per-player similarity delta for correctly-matched pairs
    # (use baseline jersey lists as reference)
    import json
    baseline_j_path = SNAPSHOTS_DIR / "baseline.json"
    baseline_data = json.loads(baseline_j_path.read_text())
    imu_jerseys_b = sorted(int(k) for k in baseline_data)

    high_conf_j_path = SNAPSHOTS_DIR / "high_conf.json"
    high_conf_data = json.loads(high_conf_j_path.read_text())
    imu_jerseys_h = sorted(int(k) for k in high_conf_data)

    common = sorted(set(imu_jerseys_b) & set(imu_jerseys_h))

    if len(common) > 0 and baseline_sim.shape == high_conf_sim.shape:
        print(f"\nSimilarity delta (high_conf - baseline) for {len(common)} shared players:")
        print(f"  {'IMU jersey':<12} {'baseline r':<14} {'high_conf r':<14} {'delta'}")
        for j in common:
            bi = imu_jerseys_b.index(j)
            hi = imu_jerseys_h.index(j)
            b_opt = baseline_asgn.get(j)
            if b_opt is None:
                continue
            # Find column index for the assigned optical jersey in each matrix
            # We only compare diagonal (correctly-assigned) entries
            if j == b_opt and j == high_conf_asgn.get(j):
                b_r = baseline_sim[bi, bi] if bi < baseline_sim.shape[1] else float("nan")
                h_r = high_conf_sim[hi, hi] if hi < high_conf_sim.shape[1] else float("nan")
                delta = h_r - b_r
                print(f"  {j:<12} {b_r:<14.4f} {h_r:<14.4f} {delta:+.4f}")
    elif baseline_sim.shape != high_conf_sim.shape:
        print(
            f"\nNote: similarity matrices have different shapes "
            f"({baseline_sim.shape} vs {high_conf_sim.shape}) — "
            "conf_min affected which optical players were available."
        )
