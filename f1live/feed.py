"""Event feed: polls OpenF1 and yields time-ordered events.

Two clocks:
  * live   -> wall clock, poll every `poll_s`, needs OpenF1 Sponsor token.
  * replay -> a historical session played back at `speed`x, no token needed.
Both produce identical event dicts, so the rest of the pipeline is agnostic.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterator

from .openf1 import OpenF1, iso, parse_dt, session_bounds

# endpoint -> event kind. All of these carry a `date` field.
STREAMS = {
    "car_data": "telemetry",
    "intervals": "interval",
    "laps": "lap",          # keyed on date_start, finished laps appear with lap_duration
    "pit": "pit",
    "stints": "stint",
    "position": "position",
    "race_control": "race_control",
    "weather": "weather",
}


@dataclass
class Event:
    kind: str
    t: datetime
    data: dict


def _ts(kind: str, row: dict) -> datetime | None:
    key = "date_start" if kind == "lap" else "date"
    v = row.get(key)
    return parse_dt(v) if v else None


class Feed:
    def __init__(self, client: OpenF1, session_key: int, drivers: list[int] | None = None,
                 streams: dict[str, str] = STREAMS):
        self.c = client
        self.session_key = session_key
        self.drivers = drivers
        self.streams = streams
        self.session = client.sessions(session_key=session_key)[0]
        self.driver_list = client.get("drivers", session_key=session_key)
        self._laps_seen: set[tuple] = set()
        self._last_stints: datetime | None = None

    def _fetch(self, start: datetime, end: datetime) -> list[Event]:
        out: list[Event] = []
        for ep, kind in self.streams.items():
            if ep == "laps":
                # laps are keyed on date_start; a lap finishing in this window started up to ~3 min earlier
                rows = self.c.get(ep, session_key=self.session_key,
                                  date_start__gt=iso(start - timedelta(seconds=240)), date_start__lt=iso(end))
                for r in rows:
                    t = _ts(kind, r)
                    if r.get("lap_duration") is None or t is None:
                        continue
                    key = (r["driver_number"], r["lap_number"])
                    t_end = t + timedelta(seconds=r["lap_duration"])
                    if t_end <= end and key not in self._laps_seen:
                        self._laps_seen.add(key)
                        out.append(Event(kind, max(t_end, start), r))
                continue
            if ep == "stints":
                if self._last_stints is None or (end - self._last_stints).total_seconds() >= 120:
                    self._last_stints = end
                    for r in self.c.get(ep, session_key=self.session_key):
                        out.append(Event(kind, start, r))
                continue
            rows = self.c.window(ep, self.session_key, start, end)
            for r in rows:
                t = _ts(kind, r)
                if t is not None:
                    out.append(Event(kind, t, r))
        out.sort(key=lambda e: e.t)
        return out

    def snapshot(self, at: datetime) -> list[Event]:
        """Latest position/interval/stint per driver before `at` (grid order, mid-session joins)."""
        out: list[Event] = []
        for ep, kind in (("position", "position"), ("intervals", "interval")):
            latest: dict[int, dict] = {}
            for r in self.c.get(ep, session_key=self.session_key, date__lt=iso(at)):
                latest[r["driver_number"]] = r  # rows arrive in date order
            out += [Event(kind, at, r) for r in latest.values()]
        return out

    def replay(self, speed: float = 10.0, window_s: int = 30, start_offset_s: int = 0,
               max_minutes: float | None = None) -> Iterator[Event]:
        """Play a finished session as if it were live."""
        s0, s1 = session_bounds(self.session)
        cur = s0 + timedelta(seconds=start_offset_s)
        if max_minutes:
            s1 = min(s1, cur + timedelta(minutes=max_minutes))
        yield from self.snapshot(cur)
        while cur < s1:
            nxt = cur + timedelta(seconds=window_s)
            t_fetch = time.monotonic()
            events = self._fetch(cur, nxt)
            yield from events
            cur = nxt
            budget = window_s / speed - (time.monotonic() - t_fetch)
            if budget > 0:
                time.sleep(budget)

    def live(self, poll_s: float = 4.0, lookback_s: int = 60) -> Iterator[Event]:
        """Follow a session in progress. Requires an OpenF1 Sponsor token."""
        seen: set[tuple] = set()
        cur = datetime.now(timezone.utc) - timedelta(seconds=lookback_s)
        yield from self.snapshot(cur)
        while True:
            now = datetime.now(timezone.utc)
            for e in self._fetch(cur, now):
                key = (e.kind, e.data.get("driver_number"), e.t.isoformat(), e.data.get("lap_number"))
                if key in seen:
                    continue
                seen.add(key)
                yield e
            cur = now - timedelta(seconds=5)  # small overlap, dedup handles it
            time.sleep(poll_s)
