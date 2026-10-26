"""Thin OpenF1 REST client with rate limiting, retries and date-window queries.

Free tier: historical data only (2023+). Live data (session in progress)
requires the OpenF1 Sponsor tier; set OPENF1_TOKEN to the bearer token.
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import requests

BASE = "https://api.openf1.org/v1"
MIN_INTERVAL_S = 0.5  # 2 req/s, stays under the free-tier limit


class OpenF1:
    def __init__(self, token: str | None = None, session: requests.Session | None = None):
        self.token = token or os.environ.get("OPENF1_TOKEN")
        self.http = session or requests.Session()
        self._last = 0.0

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def get(self, endpoint: str, **params: Any) -> list[dict]:
        """GET /v1/<endpoint>. Keys may carry an operator suffix: date__gt, date__lt."""
        query = []
        for k, v in params.items():
            if v is None:
                continue
            if k.endswith("__gt"):
                query.append(f"{k[:-4]}>{v}")
            elif k.endswith("__lt"):
                query.append(f"{k[:-4]}<{v}")
            elif k.endswith("__gte"):
                query.append(f"{k[:-5]}>={v}")
            elif k.endswith("__lte"):
                query.append(f"{k[:-5]}<={v}")
            else:
                query.append(f"{k}={v}")
        url = f"{BASE}/{endpoint}" + ("?" + "&".join(query) if query else "")
        for attempt in range(7):
            wait = self._last + MIN_INTERVAL_S - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            r = self.http.get(url, headers=self._headers(), timeout=30)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 404:  # OpenF1 returns 404 for an empty window on some endpoints
                return []
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict):  # {"detail": "No results found."}
                return []
            return data
        raise RuntimeError(f"OpenF1 request failed after retries: {url}")

    # Convenience wrappers -------------------------------------------------
    def sessions(self, **p: Any) -> list[dict]:
        return self.get("sessions", **p)

    def race_sessions(self, year: int) -> list[dict]:
        return [s for s in self.sessions(year=year, session_name="Race") if not s.get("is_cancelled")]

    def window(self, endpoint: str, session_key: int, start: datetime, end: datetime, **p: Any) -> list[dict]:
        return self.get(endpoint, session_key=session_key, date__gt=iso(start), date__lt=iso(end), **p)


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]


def parse_dt(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def session_bounds(sess: dict) -> tuple[datetime, datetime]:
    return parse_dt(sess["date_start"]), parse_dt(sess["date_end"]) + timedelta(minutes=30)
