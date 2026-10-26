"""Rolling per-driver session state updated from feed events."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

from .feed import Event

DRS_OPEN = {10, 12, 14}
PIT_LANE_SPEED_MAX = 90  # km/h; pit limiter is 80, allow jitter


@dataclass
class DriverState:
    number: int
    name: str = ""
    team: str = ""
    position: int | None = None
    gap_to_leader: float | None = None
    interval: float | None = None  # to car ahead
    lap_number: int = 0
    last_lap: float | None = None
    best_lap: float | None = None
    lap_times: list[float] = field(default_factory=list)
    stint_start_lap: int = 1
    compound: str = "UNKNOWN"
    tyre_age_at_start: int = 0
    pit_count: int = 0
    last_pit_lap: int = 0
    in_pit: bool = False
    # telemetry buffers (last ~60 samples ≈ 15-20 s)
    speed: deque = field(default_factory=lambda: deque(maxlen=60))
    throttle: deque = field(default_factory=lambda: deque(maxlen=60))
    brake: deque = field(default_factory=lambda: deque(maxlen=60))
    drs: deque = field(default_factory=lambda: deque(maxlen=60))
    gear: deque = field(default_factory=lambda: deque(maxlen=60))
    last_telemetry: datetime | None = None
    lap_max_speed: int = 0
    lap_drs_samples: int = 0
    lap_samples: int = 0

    @property
    def tyre_age(self) -> int:
        return self.tyre_age_at_start + max(0, self.lap_number - self.stint_start_lap + 1)

    @property
    def stint_lap(self) -> int:
        return max(1, self.lap_number - self.stint_start_lap + 1)

    def deg_slope(self, n: int = 5) -> float:
        """Lap-time trend over last n laps (s/lap). Positive = getting slower."""
        xs = self.lap_times[-n:]
        if len(xs) < 3:
            return 0.0
        k = len(xs)
        mx = (k - 1) / 2
        my = sum(xs) / k
        num = sum((i - mx) * (y - my) for i, y in enumerate(xs))
        den = sum((i - mx) ** 2 for i in range(k))
        return num / den if den else 0.0


class SessionState:
    def __init__(self, driver_list: list[dict], total_laps: int | None = None):
        self.drivers: dict[int, DriverState] = {}
        for d in driver_list:
            n = d["driver_number"]
            self.drivers[n] = DriverState(n, d.get("name_acronym", str(n)), d.get("team_name", ""))
        self.total_laps = total_laps
        self.track_status = "GREEN"
        self.flags: deque = deque(maxlen=20)
        self.rain = False
        self.track_temp: float | None = None
        self.t: datetime | None = None
        self.leader_lap = 0

    def d(self, n: int) -> DriverState:
        if n not in self.drivers:
            self.drivers[n] = DriverState(n, str(n))
        return self.drivers[n]

    def apply(self, e: Event) -> None:
        self.t = e.t
        r = e.data
        k = e.kind
        if k == "telemetry":
            s = self.d(r["driver_number"])
            s.speed.append(r.get("speed", 0) or 0)
            s.throttle.append(r.get("throttle", 0) or 0)
            s.brake.append(r.get("brake", 0) or 0)
            s.drs.append(r.get("drs", 0) or 0)
            s.gear.append(r.get("n_gear", 0) or 0)
            s.last_telemetry = e.t
            s.lap_samples += 1
            s.lap_max_speed = max(s.lap_max_speed, r.get("speed", 0) or 0)
            if (r.get("drs") or 0) in DRS_OPEN:
                s.lap_drs_samples += 1
        elif k == "interval":
            s = self.d(r["driver_number"])
            s.gap_to_leader = _num(r.get("gap_to_leader"))
            s.interval = _num(r.get("interval"))
        elif k == "position":
            self.d(r["driver_number"]).position = r.get("position")
        elif k == "lap":
            s = self.d(r["driver_number"])
            s.lap_number = max(s.lap_number, r["lap_number"])
            self.leader_lap = max(self.leader_lap, s.lap_number)
            lt = r.get("lap_duration")
            if lt and not r.get("is_pit_out_lap") and lt < 200:
                s.last_lap = lt
                s.lap_times.append(lt)
                s.best_lap = lt if s.best_lap is None else min(s.best_lap, lt)
            s.lap_max_speed, s.lap_drs_samples, s.lap_samples = 0, 0, 0
            s.in_pit = False
        elif k == "pit":
            s = self.d(r["driver_number"])
            s.pit_count += 1
            s.last_pit_lap = r.get("lap_number", s.lap_number)
            s.in_pit = True
        elif k == "stint":
            s = self.d(r["driver_number"])
            lap_start = r.get("lap_start") or 1
            # accept only the stint we are in (replay sees future stints in the historical table)
            if s.stint_start_lap <= lap_start <= max(1, s.lap_number + 1):
                s.stint_start_lap = lap_start
                s.compound = r.get("compound") or "UNKNOWN"
                s.tyre_age_at_start = r.get("tyre_age_at_start") or 0
        elif k == "race_control":
            msg = (r.get("message") or "").upper()
            flag = r.get("flag")
            self.flags.append((e.t, r.get("category"), flag, msg))
            if "SAFETY CAR" in msg and "ENDING" not in msg and "IN THIS LAP" not in msg:
                self.track_status = "VSC" if "VIRTUAL" in msg else "SC"
            elif flag == "RED":
                self.track_status = "RED"
            elif "CLEAR" in msg or flag == "GREEN" or "SAFETY CAR IN THIS LAP" in msg or "ENDING" in msg:
                self.track_status = "GREEN"
        elif k == "weather":
            self.rain = bool(r.get("rainfall"))
            self.track_temp = r.get("track_temperature")

    def ahead_of(self, n: int) -> DriverState | None:
        s = self.d(n)
        if s.position is None or s.position <= 1:
            return None
        for o in self.drivers.values():
            if o.position == s.position - 1:
                return o
        return None

    def behind(self, n: int) -> DriverState | None:
        s = self.d(n)
        if s.position is None:
            return None
        for o in self.drivers.values():
            if o.position == s.position + 1:
                return o
        return None


def _num(v):
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None  # e.g. "+1 LAP"
