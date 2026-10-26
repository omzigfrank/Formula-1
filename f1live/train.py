"""Build a lap-level dataset from historical races and train the two classifiers.

    python -m f1live.train --years 2024 2025 --max-races 30
"""
from __future__ import annotations

import argparse
import bisect
from datetime import timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from .features import COMPOUNDS, LAP_FEATURES
from .models import MODEL_DIR, new_classifier
from .openf1 import OpenF1, parse_dt


def _series_at(times: list, vals: list, t) -> object:
    """Last value at or before t."""
    i = bisect.bisect_right(times, t) - 1
    return vals[i] if i >= 0 else None


def build_session(c: OpenF1, session_key: int) -> pd.DataFrame:
    laps = pd.DataFrame(c.get("laps", session_key=session_key))
    if laps.empty:
        return laps
    stints = pd.DataFrame(c.get("stints", session_key=session_key))
    pits = pd.DataFrame(c.get("pit", session_key=session_key))
    pos = pd.DataFrame(c.get("position", session_key=session_key))
    inter = pd.DataFrame(c.get("intervals", session_key=session_key))
    rc = pd.DataFrame(c.get("race_control", session_key=session_key))
    wx = pd.DataFrame(c.get("weather", session_key=session_key))

    laps = laps.dropna(subset=["lap_duration", "date_start"]).copy()
    laps["t_start"] = laps["date_start"].map(parse_dt)
    laps["t_end"] = laps["t_start"] + laps["lap_duration"].map(lambda s: timedelta(seconds=s))
    total_laps = int(laps["lap_number"].max())

    # safety-car windows from race control
    sc_windows = []
    if not rc.empty:
        rc["t"] = rc["date"].map(parse_dt)
        start = None
        for _, r in rc.sort_values("t").iterrows():
            m = (r.get("message") or "").upper()
            if "SAFETY CAR" in m and start is None and "ENDING" not in m and "IN THIS LAP" not in m:
                start = r["t"]
            elif start is not None and ("CLEAR" in m or "ENDING" in m or "IN THIS LAP" in m or r.get("flag") == "GREEN"):
                sc_windows.append((start, r["t"]))
                start = None
    rain_t, rain_v = [], []
    if not wx.empty:
        wx = wx.sort_values("date")
        rain_t = [parse_dt(x) for x in wx["date"]]
        rain_v = list(wx["rainfall"].fillna(0))

    per_driver = {}
    for n, g in pos.groupby("driver_number") if not pos.empty else []:
        g = g.sort_values("date")
        per_driver.setdefault(n, {})["pos"] = ([parse_dt(x) for x in g["date"]], list(g["position"]))
    for n, g in inter.groupby("driver_number") if not inter.empty else []:
        g = g.sort_values("date")
        per_driver.setdefault(n, {})["int"] = ([parse_dt(x) for x in g["date"]],
                                              list(pd.to_numeric(g["interval"], errors="coerce")))
    pit_laps = {}
    if not pits.empty:
        for n, g in pits.groupby("driver_number"):
            pit_laps[n] = set(g["lap_number"])

    rows = []
    for n, g in laps.groupby("driver_number"):
        g = g.sort_values("lap_number")
        st = stints[stints["driver_number"] == n].sort_values("stint_number") if not stints.empty else pd.DataFrame()
        best = None
        times = []
        pit_count = 0
        for _, lp in g.iterrows():
            ln = int(lp["lap_number"])
            lt = float(lp["lap_duration"])
            t_end = lp["t_end"]
            stint = st[(st["lap_start"] <= ln) & (st["lap_end"] >= ln)] if not st.empty else pd.DataFrame()
            if not stint.empty:
                s0 = stint.iloc[0]
                stint_lap = ln - int(s0["lap_start"]) + 1
                tyre_age = int(s0.get("tyre_age_at_start") or 0) + stint_lap
                compound = s0.get("compound") or "UNKNOWN"
            else:
                stint_lap, tyre_age, compound = ln, ln, "UNKNOWN"
            if ln - 1 in pit_laps.get(n, set()):
                pit_count += 1
            pd_ = per_driver.get(n, {})
            position = _series_at(*pd_["pos"], t_end) if "pos" in pd_ else None
            interval = _series_at(*pd_["int"], t_end) if "int" in pd_ else None
            # gap behind = interval of the car one position back at this time
            gap_behind = None
            if position is not None:
                for m, q in per_driver.items():
                    if "pos" in q and _series_at(*q["pos"], t_end) == position + 1 and "int" in q:
                        gap_behind = _series_at(*q["int"], t_end)
                        break
            in_sc = any(a <= t_end <= b for a, b in sc_windows)
            rain = bool(_series_at(rain_t, rain_v, t_end)) if rain_t else False
            if not lp.get("is_pit_out_lap") and lt < 200:
                times.append(lt)
                best = lt if best is None else min(best, lt)
            slope = _slope(times[-5:])
            row = {
                "session_key": session_key, "driver_number": n, "lap": ln,
                "lap_number": ln, "laps_remaining_frac": (total_laps - ln) / total_laps,
                "stint_lap": stint_lap, "tyre_age": tyre_age, "pit_count": pit_count,
                "last_lap_delta_best": (lt - best) if best else 0.0, "deg_slope": slope,
                "interval": interval if interval is not None and not np.isnan(interval) else 30.0,
                "gap_behind": gap_behind if gap_behind is not None and not np.isnan(gap_behind) else 30.0,
                "position": position or 20, "track_status_sc": float(in_sc), "rain": float(rain),
                "y_pit_next": float((ln + 1) in pit_laps.get(n, set())),
                "_pos": position,
            }
            for cmp_ in COMPOUNDS:
                row[f"cmp_{cmp_}"] = float(compound == cmp_)
            rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values(["driver_number", "lap"])
    nxt = df.groupby("driver_number")["_pos"].shift(-1)
    df["y_overtake_next"] = ((nxt < df["_pos"]) & df["_pos"].notna() & nxt.notna()).astype(float)
    df = df[df["lap_number"] < df.groupby("driver_number")["lap_number"].transform("max")]  # drop final lap
    return df.drop(columns=["_pos"])


def _slope(xs):
    if len(xs) < 3:
        return 0.0
    k = len(xs); mx = (k - 1) / 2; my = sum(xs) / k
    den = sum((i - mx) ** 2 for i in range(k))
    return sum((i - mx) * (y - my) for i, y in enumerate(xs)) / den if den else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int, default=[2024, 2025])
    ap.add_argument("--max-races", type=int, default=40)
    ap.add_argument("--cache", default="data/laps.parquet")
    args = ap.parse_args()
    c = OpenF1()
    cache = Path(args.cache)
    cache.parent.mkdir(exist_ok=True, parents=True)
    frames = []
    done = set()
    if cache.exists():
        old = pd.read_parquet(cache)
        frames.append(old)
        done = set(old["session_key"].unique())
    keys = [s["session_key"] for y in args.years for s in c.race_sessions(y)]
    for k in [k for k in keys if k not in done][: args.max_races]:
        print(f"building session {k} ...", flush=True)
        df = build_session(c, k)
        if not df.empty:
            frames.append(df)
            pd.concat(frames).to_parquet(cache)
    data = pd.concat(frames)
    print(f"{len(data)} lap rows from {data['session_key'].nunique()} races")
    X = data[LAP_FEATURES].to_numpy(dtype=float)
    groups = data["session_key"].to_numpy()
    MODEL_DIR.mkdir(exist_ok=True)
    for target, fname in (("y_pit_next", "pit_next_lap"), ("y_overtake_next", "overtake_next_lap")):
        y = data[target].to_numpy()
        aucs, aps = [], []
        for tr, te in GroupKFold(n_splits=min(5, len(set(groups)))).split(X, y, groups):
            m = new_classifier().fit(X[tr], y[tr])
            p = m.predict_proba(X[te])[:, 1]
            aucs.append(roc_auc_score(y[te], p)); aps.append(average_precision_score(y[te], p))
        print(f"{target}: base rate {y.mean():.3f}  CV AUC {np.mean(aucs):.3f}  AP {np.mean(aps):.3f}")
        joblib.dump(new_classifier().fit(X, y), MODEL_DIR / f"{fname}.joblib")
    print("models saved to", MODEL_DIR)


if __name__ == "__main__":
    main()
