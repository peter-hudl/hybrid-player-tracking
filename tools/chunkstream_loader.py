"""
chunkstream_loader.py

Load zstd-compressed JSON chunk files from an optical tracking chunkstream
into a single pandas DataFrame.

Phase 0.1 of the IMU-Optical Identity Alignment pipeline.
"""
import json
import math
from pathlib import Path

import pandas as pd
import zstandard as zstd

# ── Column index constants (from the variables dict, fixed across chunks) ──────
_X        = 8   # _WORLD_POSITION_METERS_X
_Y        = 9   # _WORLD_POSITION_METERS_Y
_VX       = 10  # _WORLD_VELOCITY_METERS_SECOND_X
_VY       = 11  # _WORLD_VELOCITY_METERS_SECOND_Y
_AX       = 12  # _WORLD_ACCEL_METERS_SECOND_SQ_X
_AY       = 13  # _WORLD_ACCEL_METERS_SECOND_SQ_Y
_JERSEY   = 14  # _JERSEY_NUMBER
_JERSEY_C = 15  # _JERSEY_CONFIDENCE
_TEAM     = 6   # _OBJECT_DETECTION_TEAM
_CLASS    = 4   # _OBJECT_DETECTION_CLASS

# Pose keypoints: 17 keypoints, each with x/y/z at indices 17-67
# and confidence at indices 68-84.
_POSE_NAMES = [
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]
_POSE_XYZ_START = 17   # indices 17..66 inclusive (17 * 3 = 51 values)
_POSE_CONF_START = 68  # indices 68..84 inclusive (17 values)

# Build the output column names for pose keypoints
_POSE_COLS = []
for _name in _POSE_NAMES:
    _POSE_COLS += [
        f"pose_{_name}_x",
        f"pose_{_name}_y",
        f"pose_{_name}_z",
        f"pose_{_name}_conf",
    ]

FRAMES_PER_CHUNK = 40


def load_chunk(path: str | Path) -> dict:
    """Load and decompress a single chunk file, returning the raw JSON dict."""
    path = Path(path)
    with open(path, "rb") as fh:
        raw = zstd.ZstdDecompressor().decompress(fh.read(), max_output_size=50 * 1024 * 1024)
    return json.loads(raw)


def _extract_chunk_rows(chunk: dict, chunk_index: int, records: list) -> None:
    """
    Parse a single chunk and append row dicts to `records`.

    Parameters
    ----------
    chunk : dict
        Parsed JSON from a .json.zst file.
    chunk_index : int
        Zero-based index of this chunk in sorted filename order.
    records : list
        Mutable list to which row dicts are appended.
    """
    # Build reverse map: row_index -> track_id (str)
    trackables: dict[str, int] = chunk["trackables"]
    idx_to_track: dict[int, str] = {v: k for k, v in trackables.items()}

    tracking: list = chunk["tracking"]

    for local_frame_idx, frame in enumerate(tracking):
        if frame is None:
            continue

        global_frame_idx = chunk_index * FRAMES_PER_CHUNK + local_frame_idx
        session_time_s = global_frame_idx / 10.0

        for row_idx, row in enumerate(frame):
            # Skip null rows (trackable not visible)
            if row is None:
                continue

            # Skip rows with no position data
            if row[_X] is None:
                continue

            track_id = idx_to_track.get(row_idx)
            if track_id is None:
                continue

            vx = row[_VX]
            vy = row[_VY]
            speed = math.sqrt((vx or 0.0) ** 2 + (vy or 0.0) ** 2)

            record: dict = {
                "session_time_s": session_time_s,
                "frame_idx": global_frame_idx,
                "track_id": track_id,
                "x": row[_X],
                "y": row[_Y],
                "vx": vx,
                "vy": vy,
                "ax": row[_AX],
                "ay": row[_AY],
                "speed": speed,
                "jersey_number": row[_JERSEY],
                "jersey_confidence": row[_JERSEY_C],
                "team": row[_TEAM],
                "object_class": row[_CLASS],
            }

            # Pose keypoints: 17 keypoints, x/y/z at 17-67, conf at 68-84
            for kp_idx, kp_name in enumerate(_POSE_NAMES):
                xyz_base = _POSE_XYZ_START + kp_idx * 3
                conf_idx = _POSE_CONF_START + kp_idx
                record[f"pose_{kp_name}_x"] = row[xyz_base]
                record[f"pose_{kp_name}_y"] = row[xyz_base + 1]
                record[f"pose_{kp_name}_z"] = row[xyz_base + 2]
                record[f"pose_{kp_name}_conf"] = row[conf_idx]

            records.append(record)


def load_chunkstream(tracking_dir: str | Path) -> pd.DataFrame:
    """Load all chunk files from tracking_dir into a single DataFrame.

    Parameters
    ----------
    tracking_dir : str or Path
        Directory containing .json.zst chunk files named like 00000000.json.zst.

    Returns
    -------
    pd.DataFrame
        One row per (frame, trackable) with position, velocity, acceleration,
        jersey, team, object class, and pose keypoint columns.
    """
    tracking_dir = Path(tracking_dir)
    chunk_files = sorted(tracking_dir.glob("*.json.zst"))
    if not chunk_files:
        raise FileNotFoundError(f"No .json.zst files found in {tracking_dir}")

    records: list[dict] = []

    for chunk_index, chunk_path in enumerate(chunk_files):
        chunk = load_chunk(chunk_path)
        _extract_chunk_rows(chunk, chunk_index, records)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Enforce dtypes
    df["session_time_s"] = df["session_time_s"].astype(float)
    df["frame_idx"] = df["frame_idx"].astype(int)
    df["track_id"] = df["track_id"].astype(str)
    df["speed"] = df["speed"].astype(float)

    return df
