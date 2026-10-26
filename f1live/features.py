"""Feature vectors. Same function serves training (from lap-level tables) and live (from SessionState)."""
from __future__ import annotations

import numpy as np

from .state import DRS_OPEN, PIT_LANE_SPEED_MAX, DriverState, SessionState

COMPOUNDS = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]

LAP_FEATURES = [
    "lap_number", "laps_remaining_frac", "stint_lap", "tyre_age", "pit_count",
    "last_lap_delta_best", "deg_slope", "interval", "gap_behind", "position",
    "track_status_sc", "rain",
] + [f"cmp_{c}" for c in COMPOUNDS]

TELEMETRY_FEATURES = ["drs_frac", "lap_max_speed", "speed_now", "throttle_mean", "brake_frac", "pit_speed_flag"]


def lap_features(row: dict) -> list[float]:
    """row keys mirror LAP_FEATURES semantics; used by train.py and live."""
    return [float(row.get(f, 0.0) or 0.0) for f in LAP_FEATURES]


def from_state(st: SessionState, s: DriverState) -> dict:
    behind = st.behind(s.number)
    total = st.total_laps or 60
    row = {
        "lap_number": s.lap_number,
        "laps_remaining_frac": max(0.0, (total - s.lap_number) / total),
        "stint_lap": s.stint_lap,
        "tyre_age": s.tyre_age,
        "pit_count": s.pit_count,
        "last_lap_delta_best": (s.last_lap - s.best_lap) if (s.last_lap and s.best_lap) else 0.0,
        "deg_slope": s.deg_slope(),
        "interval": s.interval if s.interval is not None else 30.0,
        "gap_behind": behind.interval if (behind and behind.interval is not None) else 30.0,
        "position": s.position or 20,
        "track_status_sc": 1.0 if st.track_status in ("SC", "VSC") else 0.0,
        "rain": 1.0 if st.rain else 0.0,
    }
    for c in COMPOUNDS:
        row[f"cmp_{c}"] = 1.0 if s.compound == c else 0.0
    row.update(telemetry_features(s))
    return row


def telemetry_features(s: DriverState) -> dict:
    sp = np.array(s.speed or [0])
    th = np.array(s.throttle or [0])
    br = np.array(s.brake or [0])
    drs = list(s.drs)
    recent = sp[-12:]  # last ~3-4 s
    return {
        "drs_frac": (s.lap_drs_samples / s.lap_samples) if s.lap_samples else 0.0,
        "lap_max_speed": float(s.lap_max_speed),
        "speed_now": float(sp[-1]),
        "throttle_mean": float(th.mean()),
        "brake_frac": float((br > 0).mean()),
        # pit-lane signature: sustained speed in the limiter band with throttle applied
        "pit_speed_flag": 1.0 if (len(recent) >= 8 and 40 < recent.mean() < PIT_LANE_SPEED_MAX
                                  and recent.std() < 8 and th[-12:].mean() > 5) else 0.0,
        "drs_open_now": 1.0 if (drs and drs[-1] in DRS_OPEN) else 0.0,
    }
