"""Predictors. Two trained classifiers (pit-next-lap, overtake-next-lap) plus telemetry heuristics."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .features import LAP_FEATURES, from_state
from .state import DriverState, SessionState

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


@dataclass
class Prediction:
    driver: str
    position: int | None
    pit_next_lap: float
    overtake_next_lap: float
    projected_lap: float | None
    tyre_age: int
    compound: str
    flags: list[str]


class Predictor:
    def __init__(self, model_dir: Path = MODEL_DIR):
        self.pit = _load(model_dir / "pit_next_lap.joblib")
        self.ovt = _load(model_dir / "overtake_next_lap.joblib")

    def predict(self, st: SessionState) -> list[Prediction]:
        out = []
        for s in sorted(st.drivers.values(), key=lambda d: d.position or 99):
            if s.lap_number == 0 and not s.speed:
                continue
            row = from_state(st, s)
            x = np.array([[row[f] for f in LAP_FEATURES]])
            p_pit = _proba(self.pit, x)
            p_ovt = _proba(self.ovt, x)
            flags = []
            # --- telemetry overrides / live signals ---
            if row["pit_speed_flag"]:
                p_pit = 0.95 if p_pit != p_pit else max(p_pit, 0.95)
                flags.append("PIT LANE")
            if row["drs_open_now"] and s.interval is not None and s.interval < 1.0:
                p_ovt = min(1.0, p_ovt + 0.15) if p_ovt == p_ovt else p_ovt
                flags.append("DRS")
            if st.track_status in ("SC", "VSC"):
                p_pit = min(1.0, p_pit * 1.8 + 0.1) if p_pit == p_pit else p_pit  # cheap stop window
                flags.append(st.track_status)
            if _stopped(s):
                flags.append("STOPPED?")
            if s.interval is not None and s.interval < 0.5 and s.position and s.position > 1:
                flags.append("ATTACK")
            out.append(Prediction(
                driver=s.name, position=s.position, pit_next_lap=p_pit, overtake_next_lap=p_ovt,
                projected_lap=project_lap(s), tyre_age=s.tyre_age, compound=s.compound, flags=flags,
            ))
        return out


def project_lap(s: DriverState) -> float | None:
    """Expected next lap time: last lap adjusted for the degradation trend."""
    if not s.lap_times:
        return None
    return s.lap_times[-1] + s.deg_slope()


def _stopped(s: DriverState) -> bool:
    """Car has been near-stationary on track (not pit lane) for the last ~5 s."""
    sp = list(s.speed)[-15:]
    return len(sp) >= 15 and max(sp) < 30 and s.lap_number > 0 and not s.in_pit


def _load(p: Path):
    return joblib.load(p) if p.exists() else None


def _proba(model, x: np.ndarray) -> float:
    if model is None:
        return float("nan")
    return float(model.predict_proba(x)[0, 1])


def new_classifier() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                          l2_regularization=1.0, random_state=0)
