"""
WIMU Data Loader Module

This module provides functions to load GPS, UWB/LPS, and IMU sensor data from WIMU .qul files.

WIMU files contain positional and inertial data:
- GPS data (sensor 81): Latitude/Longitude coordinates that need transformation to pitch coordinates
- UWB/LPS data (sensor 82): X/Y coordinates already in pitch coordinate system
- Accelerometer (sensor 300): Raw acceleration in X/Y/Z axes
- Attitude (sensor 301): Orientation angles (phi, theta, psi) plus body-frame and earth-frame acceleration
- Gyroscope (sensor 302): Angular velocity in X/Y/Z axes

Timezone Handling:
- WIMU timecodes are stored as Unix epoch milliseconds
- If the WIMU device clock was set to local time, specify the timezone parameter
- The loader will interpret the milliseconds as local time, then convert to UTC
- All output timestamps are UTC-aware for consistent internal processing

Functions:
    load_wimu_data: Load both GPS and UWB data from a WIMU file (optionally also IMU)
    load_wimu_gps: Load GPS data (lat/lon) from a WIMU file
    load_wimu_uwb: Load UWB/LPS data (x/y pitch coordinates) from a WIMU file
    load_wimu_imu: Load raw IMU sensor data (accelerometer, attitude, gyroscope) from a WIMU file
    load_wimu_attitude: Load attitude sensor data (sensor 301) from a WIMU file
    load_all_wimu_files: Load data from multiple WIMU files in a directory
"""

import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Callable, Dict, List
from pywimu import PyWimu


# Sensor ID -> column names mapping for IMU sensors
_IMU_SENSOR_COLUMNS = {
    300: ['timecode', 'accel_x', 'accel_y', 'accel_z'],
    302: ['timecode', 'gyro_x', 'gyro_y', 'gyro_z'],
    301: [
        'timecode', 'phi', 'theta', 'psi',
        'ac_body_x', 'ac_body_y', 'ac_body_z',
        'ac_earth_x', 'ac_earth_y', 'ac_earth_z',
        'euler_x', 'euler_y', 'euler_z',
    ],
}

_DEFAULT_IMU_SENSORS = [300, 302, 301]


def _convert_timecodes(df: pd.DataFrame, timezone: str, sensor_label: str) -> pd.DataFrame:
    """
    Add a UTC-aware 'timestamp' column to a DataFrame whose 'timecode' column
    contains milliseconds since epoch (possibly in local time).

    Args:
        df: DataFrame with a 'timecode' column (ms since epoch)
        timezone: Timezone string (e.g. "UTC", "UTC+2", "Europe/Oslo")
        sensor_label: Human-readable label used in warning messages

    Returns:
        The same DataFrame with a 'timestamp' column added (in-place on copy).
    """
    df['local_datetime'] = pd.to_datetime(df['timecode'], unit='ms')

    try:
        if timezone.upper() == "UTC":
            df['timestamp'] = df['local_datetime'].dt.tz_localize('UTC')
        elif timezone.upper().startswith("UTC"):
            # Parse offset like "+2" or "-5"
            offset_str = timezone[3:]
            if ":" in offset_str:
                hours, minutes = offset_str.split(":")
                offset_hours = int(hours)
                offset_minutes = int(minutes) if int(hours) >= 0 else -int(minutes)
            else:
                offset_hours = int(offset_str)
                offset_minutes = 0

            # Etc/GMT offsets are sign-inverted relative to UTC offsets
            tz_str = f"Etc/GMT{-offset_hours:+d}"
            df['timestamp'] = df['local_datetime'].dt.tz_localize(tz_str).dt.tz_convert('UTC')
        else:
            # IANA timezone name
            df['timestamp'] = df['local_datetime'].dt.tz_localize(timezone).dt.tz_convert('UTC')
    except Exception as e:
        print(f"Warning: Failed to apply timezone {timezone} to WIMU {sensor_label}, treating as UTC: {e}")
        df['timestamp'] = df['local_datetime'].dt.tz_localize('UTC')

    df = df.drop(columns=['local_datetime'])
    return df


def load_wimu_gps(file_path: str | Path, timezone: str = "UTC") -> pd.DataFrame:
    """
    Load WIMU GPS data from a WIMU .qul file.

    GPS data is from sensor ID 81 and contains latitude/longitude coordinates
    that need to be transformed to pitch coordinates.

    WIMU devices store timecodes as milliseconds since epoch. If the device clock
    was set to local time, the timezone parameter adjusts timestamps to UTC.

    Args:
        file_path: Path to the WIMU .qul file
        timezone: Timezone string for WIMU timestamps (e.g., "UTC", "UTC+2", "Europe/Oslo")
                 If WIMU clock was set to local time, specify the timezone to convert to UTC

    Returns:
        DataFrame with columns: timecode, lat, lon, speed, datetime, distance, sats, timestamp
        Speed is converted from km/h to m/s.
        All timestamps are converted to UTC.
        Returns empty DataFrame if loading fails.
    """
    file_path = str(file_path).replace("\\", "/")

    try:
        wimu = PyWimu(file_path)
        df_gps = wimu.get_sensor(81).to_pandas()

        # Select and rename columns for consistency
        df_gps = df_gps[["timecode", "LAT", "LON", "SPEED", "DATETIME", "DIST", "SATCOUNT"]]
        df_gps.columns = ['timecode', 'lat', 'lon', 'speed', 'datetime', 'distance', 'sats']

        # Convert speed from km/h to m/s
        df_gps['speed'] = df_gps['speed'] / 3.6

        df_gps = _convert_timecodes(df_gps, timezone, "GPS")

        return df_gps

    except Exception as e:
        print(f"Error loading WIMU GPS data from {file_path}: {e}")
        return pd.DataFrame()


def load_wimu_uwb(file_path: str | Path, uwb_transformer: Optional[Callable] = None, timezone: str = "UTC") -> pd.DataFrame:
    """
    Load WIMU UWB/LPS data from a WIMU .qul file.

    UWB data is from sensor ID 82 and contains X/Y coordinates in the UWB coordinate system.
    If a transformer is provided, it applies a rigid transformation (rotation + translation)
    to align with the pitch coordinate system.

    WIMU devices store timecodes as milliseconds since epoch. If the device clock
    was set to local time, the timezone parameter adjusts timestamps to UTC.

    Args:
        file_path: Path to the WIMU .qul file
        uwb_transformer: Optional function to transform UWB (x, y) to pitch coordinates (x, y).
                        If provided, adds transformed 'x' and 'y' columns.
        timezone: Timezone string for WIMU timestamps (e.g., "UTC", "UTC+2", "Europe/Oslo")
                 If WIMU clock was set to local time, specify the timezone to convert to UTC

    Returns:
        DataFrame with columns: timecode, x, y, speed, distance, timestamp
        Speed is converted from km/h to m/s.
        Coordinates are transformed if transformer provided.
        All timestamps are converted to UTC.
        Returns empty DataFrame if loading fails.
    """
    file_path = str(file_path).replace("\\", "/")

    try:
        wimu = PyWimu(file_path)
        df_uwb = wimu.get_sensor(82).to_pandas()

        # Select and rename columns for consistency
        df_uwb = df_uwb[["timecode", "X", "Y", "SPEED", "DISTANCE"]]
        df_uwb.columns = ['timecode', 'x', 'y', 'speed', 'distance']

        # Apply transformation if provided
        if uwb_transformer is not None:
            try:
                transformed = df_uwb.apply(
                    lambda row: uwb_transformer(row['x'], row['y']),
                    axis=1,
                    result_type='expand'
                )
                df_uwb[['x', 'y']] = transformed
            except Exception as e:
                print(f"Warning: Failed to transform UWB coordinates: {e}")

        # Convert speed from km/h to m/s
        df_uwb['speed'] = df_uwb['speed'] / 3.6

        df_uwb = _convert_timecodes(df_uwb, timezone, "UWB")

        return df_uwb

    except Exception as e:
        print(f"Error loading WIMU UWB data from {file_path}: {e}")
        return pd.DataFrame()


def load_wimu_imu(
    file_path: str | Path,
    timezone: str = "UTC",
    sensors: Optional[List[int]] = None
) -> Dict[int, pd.DataFrame]:
    """
    Load raw IMU sensor data from a WIMU .qul file.

    Loads one or more of the following sensors:
        300 — Accelerometer: accel_x, accel_y, accel_z
        302 — Gyroscope:     gyro_x, gyro_y, gyro_z
        301 — Attitude:      phi, theta, psi, ac_body_x/y/z, ac_earth_x/y/z, euler_x/y/z

    WIMU devices store timecodes as milliseconds since epoch. If the device clock
    was set to local time, the timezone parameter adjusts timestamps to UTC.

    Args:
        file_path: Path to the WIMU .qul file
        timezone: Timezone string for WIMU timestamps (e.g., "UTC", "UTC+2", "Europe/Oslo")
                 If WIMU clock was set to local time, specify the timezone to convert to UTC
        sensors: List of sensor IDs to load. Defaults to [300, 302, 301].

    Returns:
        Dict mapping sensor_id -> DataFrame. Each DataFrame has a 'timecode' column,
        sensor-specific columns, and a UTC-aware 'timestamp' column.
        Sensors that fail to load are omitted from the dict.
    """
    file_path = str(file_path).replace("\\", "/")

    if sensors is None:
        sensors = _DEFAULT_IMU_SENSORS

    result: Dict[int, pd.DataFrame] = {}

    try:
        wimu = PyWimu(file_path)
    except Exception as e:
        print(f"Error opening WIMU file {file_path}: {e}")
        return result

    for sensor_id in sensors:
        try:
            df = wimu.get_sensor(sensor_id).to_pandas()

            # Rename to lowercase snake_case
            col_names = _IMU_SENSOR_COLUMNS[sensor_id]
            df.columns = col_names

            df = _convert_timecodes(df, timezone, f"IMU sensor {sensor_id}")
            result[sensor_id] = df

        except Exception as e:
            print(f"Warning: Failed to load IMU sensor {sensor_id} from {file_path}: {e}")

    return result


def load_wimu_attitude(file_path: str | Path, timezone: str = "UTC") -> pd.DataFrame:
    """
    Load WIMU attitude sensor data (sensor 301) from a WIMU .qul file.

    Convenience wrapper around load_wimu_imu for the attitude sensor, which is
    the primary sensor for IMU-optical alignment work.

    Attitude data contains orientation angles (phi, theta, psi) and acceleration
    in both body frame and earth frame, plus Euler angles.

    Args:
        file_path: Path to the WIMU .qul file
        timezone: Timezone string for WIMU timestamps (e.g., "UTC", "UTC+2", "Europe/Oslo")
                 If WIMU clock was set to local time, specify the timezone to convert to UTC

    Returns:
        DataFrame with columns:
            timecode, phi, theta, psi,
            ac_body_x, ac_body_y, ac_body_z,
            ac_earth_x, ac_earth_y, ac_earth_z,
            euler_x, euler_y, euler_z,
            timestamp
        Returns empty DataFrame if loading fails.
    """
    imu = load_wimu_imu(file_path, timezone=timezone, sensors=[301])
    if 301 not in imu:
        print(f"Warning: Attitude sensor (301) not found in {file_path}")
        return pd.DataFrame()
    return imu[301]


def load_wimu_data(
    file_path: str | Path,
    gps_transformer: Optional[Callable] = None,
    uwb_transformer: Optional[Callable] = None,
    start_time: Optional[pd.Timestamp] = None,
    end_time: Optional[pd.Timestamp] = None,
    timezone: str = "UTC",
    include_imu: bool = False
) -> Tuple:
    """
    Load both GPS and UWB data from a WIMU .qul file.

    Args:
        file_path: Path to the WIMU .qul file
        gps_transformer: Optional function to transform GPS (lat/lon) to pitch coordinates (x, y).
                        If provided, adds 'x' and 'y' columns to GPS DataFrame.
        uwb_transformer: Optional function to transform UWB (x, y) to pitch coordinates (x, y).
                        If provided, transforms UWB coordinates.
        start_time: Optional start time for filtering data before transformation
        end_time: Optional end time for filtering data before transformation
        timezone: Timezone string for WIMU timestamps (e.g., "UTC", "UTC+2", "Europe/Oslo")
                 If WIMU clock was set to local time, specify the timezone to convert to UTC
        include_imu: If True, also load IMU sensors and return a 3-tuple.
                    Default is False for backward compatibility.

    Returns:
        When include_imu=False (default): Tuple of (df_gps, df_uwb)
        When include_imu=True: Tuple of (df_gps, df_uwb, imu_dict)
            - df_gps: GPS data with lat/lon (and optionally x/y if transformer provided)
            - df_uwb: UWB data with x/y pitch coordinates (transformed if transformer provided)
            - imu_dict: Dict[int, pd.DataFrame] mapping sensor_id to IMU DataFrames
        All timestamps are in UTC.
    """
    df_gps = load_wimu_gps(file_path, timezone=timezone)
    df_uwb = load_wimu_uwb(file_path, uwb_transformer=None, timezone=timezone)  # Don't transform yet

    # Filter by time range BEFORE transformation for performance
    if start_time is not None and end_time is not None:
        # Ensure timezone-aware UTC timestamps
        if start_time.tz is None:
            start_time = start_time.tz_localize('UTC')
        else:
            start_time = start_time.tz_convert('UTC')

        if end_time.tz is None:
            end_time = end_time.tz_localize('UTC')
        else:
            end_time = end_time.tz_convert('UTC')

        # Filter GPS data
        if not df_gps.empty and 'timestamp' in df_gps.columns:
            df_gps = df_gps[
                (df_gps['timestamp'] >= start_time) &
                (df_gps['timestamp'] <= end_time)
            ].copy()

        # Filter UWB data
        if not df_uwb.empty and 'timestamp' in df_uwb.columns:
            df_uwb = df_uwb[
                (df_uwb['timestamp'] >= start_time) &
                (df_uwb['timestamp'] <= end_time)
            ].copy()

    # Apply GPS transformation if provided (only to filtered data)
    if gps_transformer is not None and not df_gps.empty:
        try:
            transformed = df_gps.apply(
                lambda row: gps_transformer(row['lon'], row['lat']),
                axis=1,
                result_type='expand'
            )
            df_gps[['x', 'y']] = transformed
        except Exception as e:
            print(f"Warning: Failed to transform GPS coordinates: {e}")

    # Apply UWB transformation if provided (only to filtered data)
    if uwb_transformer is not None and not df_uwb.empty:
        try:
            transformed = df_uwb.apply(
                lambda row: uwb_transformer(row['x'], row['y']),
                axis=1,
                result_type='expand'
            )
            df_uwb[['x', 'y']] = transformed
        except Exception as e:
            print(f"Warning: Failed to transform UWB coordinates: {e}")

    if include_imu:
        imu_dict = load_wimu_imu(file_path, timezone=timezone)
        return df_gps, df_uwb, imu_dict

    return df_gps, df_uwb


def load_all_wimu_files(
    directory: str | Path,
    gps_transformer: Optional[Callable] = None,
    uwb_transformer: Optional[Callable] = None,
    file_pattern: str = "*.qul",
    jerseys: Optional[List[int]] = None,
    start_time: Optional[pd.Timestamp] = None,
    end_time: Optional[pd.Timestamp] = None,
    timezone: str = "UTC",
    include_imu: bool = False
) -> Dict[str, Tuple]:
    """
    Load GPS and UWB data from WIMU files in a directory.

    Args:
        directory: Path to directory containing WIMU .qul files
        gps_transformer: Optional function to transform GPS to pitch coordinates
        uwb_transformer: Optional function to transform UWB to pitch coordinates
        file_pattern: Glob pattern for matching files (default: "*.qul") - ignored if jerseys specified
        jerseys: Optional list of jersey numbers to load (e.g., [2, 3, 8] -> jersey_2.qul, jersey_3.qul, jersey_8.qul)
        start_time: Optional start time for filtering data before transformation
        end_time: Optional end time for filtering data before transformation
        timezone: Timezone string for WIMU timestamps (e.g., "UTC", "UTC+2", "Europe/Oslo")
                 If WIMU clock was set to local time, specify the timezone to convert to UTC
        include_imu: If True, dict values become 3-tuples (df_gps, df_uwb, imu_dict).
                    Default is False for backward compatibility.

    Returns:
        Dictionary mapping device IDs to tuples.
        When include_imu=False (default): device_id -> (df_gps, df_uwb)
        When include_imu=True: device_id -> (df_gps, df_uwb, imu_dict)
        Device ID is jersey number (e.g., "jersey_2" for jersey_2.qul)
        All timestamps are in UTC.
    """
    directory = Path(directory)
    wimu_data = {}

    if jerseys is not None and len(jerseys) > 0:
        # Load specific jersey files
        for jersey in jerseys:
            jersey_file = directory / f"jersey_{jersey}.qul"
            if jersey_file.exists():
                device_id = f"jersey_{jersey}"
                print(f"Loading WIMU data from {jersey_file.name}...")
                result = load_wimu_data(
                    jersey_file,
                    gps_transformer,
                    uwb_transformer,
                    start_time,
                    end_time,
                    timezone=timezone,
                    include_imu=include_imu
                )
                df_gps, df_uwb = result[0], result[1]

                if not df_gps.empty or not df_uwb.empty:
                    wimu_data[device_id] = result
                    print(f"Loaded {len(df_gps)} GPS records, {len(df_uwb)} UWB records for {device_id}")
                    if not df_gps.empty and 'timestamp' in df_gps.columns:
                        first_ts = df_gps['timestamp'].iloc[0]
                        print(f"  First GPS timestamp: {first_ts}")
                else:
                    print(f"No data loaded from {jersey_file.name}")
            else:
                print(f"Warning: Jersey file jersey_{jersey}.qul not found, skipping")
    else:
        # Load all files matching pattern (default behavior)
        file_list = list(directory.glob(file_pattern))

        # Prefer jersey_*.qul files if they exist
        jersey_files = list(directory.glob("jersey_*.qul"))
        if jersey_files:
            file_list = jersey_files

        for file_path in sorted(file_list):
            # Extract device ID from filename
            filename = file_path.stem
            if filename.startswith("jersey_"):
                device_id = filename  # Use full jersey_X as device ID
            else:
                # Legacy: extract device ID (e.g., "WIMU_1" from "WIMU_1-log_...")
                device_id = filename.split('-')[0] if '-' in filename else filename

            print(f"Loading WIMU data from {file_path.name}...")
            result = load_wimu_data(
                file_path,
                gps_transformer,
                uwb_transformer,
                start_time,
                end_time,
                timezone=timezone,
                include_imu=include_imu
            )
            df_gps, df_uwb = result[0], result[1]

            if not df_gps.empty or not df_uwb.empty:
                wimu_data[device_id] = result
                print(f"Loaded {len(df_gps)} GPS records, {len(df_uwb)} UWB records for {device_id}")
                if not df_gps.empty and 'timestamp' in df_gps.columns:
                    first_ts = df_gps['timestamp'].iloc[0]
                    print(f"  First GPS timestamp: {first_ts}")
            else:
                print(f"No data loaded from {file_path.name}")

    return wimu_data


def filter_wimu_by_timerange(
    wimu_data: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
    start_time: pd.Timestamp,
    end_time: pd.Timestamp
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Filter WIMU data to a specific time range.

    Args:
        wimu_data: Dictionary of device_id -> (df_gps, df_uwb) tuples
        start_time: Start timestamp for filtering
        end_time: End timestamp for filtering

    Returns:
        Filtered dictionary with same structure
    """
    filtered_data = {}

    # Ensure start_time and end_time are timezone-aware UTC
    if start_time.tz is None:
        start_time = start_time.tz_localize('UTC')
    else:
        start_time = start_time.tz_convert('UTC')

    if end_time.tz is None:
        end_time = end_time.tz_localize('UTC')
    else:
        end_time = end_time.tz_convert('UTC')

    for device_id, (df_gps, df_uwb) in wimu_data.items():
        # Filter GPS data
        if not df_gps.empty and 'timestamp' in df_gps.columns:
            # Ensure GPS timestamps are timezone-aware
            gps_timestamps = pd.to_datetime(df_gps['timestamp'])
            if gps_timestamps.dt.tz is None:
                gps_timestamps = gps_timestamps.dt.tz_localize('UTC')
            else:
                gps_timestamps = gps_timestamps.dt.tz_convert('UTC')

            df_gps_filtered = df_gps[
                (gps_timestamps >= start_time) &
                (gps_timestamps <= end_time)
            ].copy()
        else:
            df_gps_filtered = df_gps.copy()

        # Filter UWB data
        if not df_uwb.empty and 'timestamp' in df_uwb.columns:
            # Ensure UWB timestamps are timezone-aware
            uwb_timestamps = pd.to_datetime(df_uwb['timestamp'])
            if uwb_timestamps.dt.tz is None:
                uwb_timestamps = uwb_timestamps.dt.tz_localize('UTC')
            else:
                uwb_timestamps = uwb_timestamps.dt.tz_convert('UTC')

            df_uwb_filtered = df_uwb[
                (uwb_timestamps >= start_time) &
                (uwb_timestamps <= end_time)
            ].copy()
        else:
            df_uwb_filtered = df_uwb.copy()

        filtered_data[device_id] = (df_gps_filtered, df_uwb_filtered)

    return filtered_data


def get_wimu_summary(
    wimu_data: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]
) -> pd.DataFrame:
    """
    Get a summary of loaded WIMU data.

    Args:
        wimu_data: Dictionary of device_id -> (df_gps, df_uwb) tuples

    Returns:
        DataFrame with summary statistics for each device
    """
    summary_rows = []

    for device_id, (df_gps, df_uwb) in wimu_data.items():
        row = {
            'device_id': device_id,
            'gps_records': len(df_gps),
            'uwb_records': len(df_uwb),
        }

        if not df_gps.empty and 'timestamp' in df_gps.columns:
            row['gps_start'] = df_gps['timestamp'].min()
            row['gps_end'] = df_gps['timestamp'].max()
            row['gps_duration_sec'] = (row['gps_end'] - row['gps_start']).total_seconds()

        if not df_uwb.empty and 'timestamp' in df_uwb.columns:
            row['uwb_start'] = df_uwb['timestamp'].min()
            row['uwb_end'] = df_uwb['timestamp'].max()
            row['uwb_duration_sec'] = (row['uwb_end'] - row['uwb_start']).total_seconds()

        summary_rows.append(row)

    return pd.DataFrame(summary_rows)
