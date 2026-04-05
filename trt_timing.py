"""TRT operation timing telemetry — SQLite-backed event log.

Records start/end timestamps for engine builds, loads, and refits.
Provides ETA estimates based on trimmed mean of recent measurements.
"""

import logging
import os
import sqlite3
import threading
import time

import folder_paths

log = logging.getLogger("comfyui_tensorrt")

_local = threading.local()

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS trt_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    start_ts REAL NOT NULL,
    end_ts REAL,
    event_type TEXT NOT NULL,
    model_name TEXT,
    model_type TEXT,
    resolution TEXT,
    batch_size INTEGER,
    lora_hash TEXT,
    result TEXT NOT NULL DEFAULT 'unknown',
    message TEXT
);
"""


def db_path():
    """Return path to the timings database."""
    return os.path.join(folder_paths.models_dir, "tensorrt", "timings.db")


def _get_conn():
    """Return a thread-local SQLite connection (created on first use)."""
    conn = getattr(_local, "conn", None)
    if conn is not None:
        return conn
    path = db_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(_SCHEMA)
    conn.commit()
    _local.conn = conn
    return conn


def begin_event(
    event_type,
    model_name=None,
    model_type=None,
    resolution=None,
    batch_size=None,
    lora_hash=None,
):
    """Insert an event row at operation start. Returns the row id."""
    conn = _get_conn()
    cur = conn.execute(
        "INSERT INTO trt_events "
        "(start_ts, event_type, model_name, model_type, resolution, batch_size, lora_hash, result) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, 'unknown')",
        (
            time.time(),
            event_type,
            model_name,
            model_type,
            resolution,
            batch_size,
            lora_hash,
        ),
    )
    conn.commit()
    return cur.lastrowid


def end_event(row_id, result, message=""):
    """Update an event row at operation end."""
    if row_id is None:
        return
    conn = _get_conn()
    conn.execute(
        "UPDATE trt_events SET end_ts = ?, result = ?, message = ? WHERE id = ?",
        (time.time(), result, message or "", row_id),
    )
    conn.commit()


def estimate_eta(event_type, model_type=None, resolution=None):
    """Estimate duration from the last 5 successful events.

    Drops the 2 extremes (min + max), averages the remaining 3.
    Returns seconds as float, or None if fewer than 3 data points.
    """
    conn = _get_conn()
    rows = conn.execute(
        "SELECT (end_ts - start_ts) AS duration "
        "FROM trt_events "
        "WHERE event_type = ? "
        "  AND (model_type = ? OR ? IS NULL) "
        "  AND (resolution = ? OR ? IS NULL) "
        "  AND result = 'success' "
        "  AND end_ts IS NOT NULL "
        "ORDER BY start_ts DESC LIMIT 5",
        (event_type, model_type, model_type, resolution, resolution),
    ).fetchall()

    durations = [r[0] for r in rows if r[0] is not None and r[0] > 0]
    if len(durations) < 3:
        return None

    durations.sort()
    # Drop min and max (the 2 extremes)
    trimmed = durations[1:-1] if len(durations) >= 4 else durations
    # If exactly 3, keep all 3
    if len(durations) == 3:
        trimmed = durations
    return sum(trimmed) / len(trimmed)
