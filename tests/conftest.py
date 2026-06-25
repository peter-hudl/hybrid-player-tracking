"""
Shared fixtures and helpers for the hybrid-player-tracking test suite.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

SNAPSHOTS_DIR = Path(__file__).parent / "snapshots"
DATA_DIR = Path(__file__).parent.parent / "data"
TRACKING_PARQUET = DATA_DIR / "tracking_full.parquet"


# ── CLI option ───────────────────────────────────────────────────────────────

def pytest_addoption(parser):
    parser.addoption(
        "--snapshot-update",
        action="store_true",
        default=False,
        help="Regenerate snapshot files instead of asserting against them.",
    )


# ── Markers ──────────────────────────────────────────────────────────────────

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_data: skip if data/tracking_full.parquet is absent",
    )


def pytest_collection_modifyitems(config, items):
    skip_no_data = pytest.mark.skip(reason="data/tracking_full.parquet not present")
    for item in items:
        if "requires_data" in item.keywords and not TRACKING_PARQUET.exists():
            item.add_marker(skip_no_data)


# ── Snapshot fixture ─────────────────────────────────────────────────────────

class Snapshot:
    def __init__(self, update: bool):
        self._update = update

    def assert_assignments(self, name: str, assignments: dict[int, int]):
        path = SNAPSHOTS_DIR / f"{name}.json"
        if self._update:
            SNAPSHOTS_DIR.mkdir(exist_ok=True)
            path.write_text(
                json.dumps({str(k): v for k, v in assignments.items()}, indent=2)
            )
            return
        saved = {int(k): v for k, v in json.loads(path.read_text()).items()}
        assert assignments == saved, (
            f"Assignments diverged from snapshot {path.name}.\n"
            f"  Got:      {assignments}\n"
            f"  Expected: {saved}"
        )

    def assert_similarity(self, name: str, sim: np.ndarray, atol: float = 1e-4):
        path = SNAPSHOTS_DIR / f"{name}_sim.npy"
        if self._update:
            SNAPSHOTS_DIR.mkdir(exist_ok=True)
            np.save(path, sim)
            return
        saved = np.load(path)
        np.testing.assert_allclose(sim, saved, atol=atol, err_msg=f"Similarity matrix diverged from {path.name}")

    def load_assignments(self, name: str) -> dict[int, int]:
        path = SNAPSHOTS_DIR / f"{name}.json"
        return {int(k): v for k, v in json.loads(path.read_text()).items()}

    def load_similarity(self, name: str) -> np.ndarray:
        return np.load(SNAPSHOTS_DIR / f"{name}_sim.npy")


@pytest.fixture
def snapshot(request):
    return Snapshot(update=request.config.getoption("--snapshot-update"))
