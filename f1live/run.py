"""Run predictions against a live session or a replay.

    python -m f1live.run --replay 9839 --speed 20 --minutes 15   # 2025 Abu Dhabi GP, no token needed
    python -m f1live.run --live                                 # needs OPENF1_TOKEN (Sponsor tier)
"""
from __future__ import annotations

import argparse
import sys
import time

from .feed import Feed
from .models import Predictor
from .openf1 import OpenF1
from .state import SessionState


def render(st: SessionState, preds, every: int) -> str:
    hdr = f"{'P':>2} {'DRV':<4} {'LAP':>3} {'TYRE':<6} {'AGE':>3} {'INT':>6} {'PROJ':>7} {'PIT%':>5} {'OVT%':>5}  FLAGS"
    lines = [f"[{st.t:%H:%M:%S}] lap {st.leader_lap}  track {st.track_status}  rain={st.rain}", hdr]
    for p in preds[:every]:
        s = st.drivers[[k for k, v in st.drivers.items() if v.name == p.driver][0]]
        lines.append(
            f"{p.position or '-':>2} {p.driver:<4} {s.lap_number:>3} {p.compound[:6]:<6} {p.tyre_age:>3} "
            f"{(f'{s.interval:.1f}' if s.interval is not None else '-'):>6} "
            f"{(f'{p.projected_lap:.2f}' if p.projected_lap else '-'):>7} "
            f"{_pct(p.pit_next_lap):>5} {_pct(p.overtake_next_lap):>5}  {' '.join(p.flags)}")
    return "\n".join(lines)


def _pct(x: float) -> str:
    return "  n/a" if x != x else f"{100 * x:4.0f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", type=int, help="session_key to replay")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--speed", type=float, default=10.0)
    ap.add_argument("--minutes", type=float, default=None, help="replay only this many session minutes")
    ap.add_argument("--offset", type=int, default=0, help="replay start offset in seconds")
    ap.add_argument("--refresh", type=float, default=5.0, help="print every N session seconds")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()
    if not args.replay and not args.live:
        ap.error("pass --replay <session_key> or --live")

    c = OpenF1()
    if args.live:
        sess = c.sessions(session_key="latest")[0]
        key = sess["session_key"]
    else:
        key = args.replay
    feed = Feed(c, key)
    st = SessionState(feed.driver_list)
    pred = Predictor()
    if pred.pit is None:
        print("no trained models in ./models; run `python -m f1live.train` first (telemetry flags still work)", file=sys.stderr)
    events = feed.live() if args.live else feed.replay(args.speed, start_offset_s=args.offset, max_minutes=args.minutes)
    last_print = None
    for e in events:
        st.apply(e)
        if last_print is None or (e.t - last_print).total_seconds() >= args.refresh:
            print(render(st, pred.predict(st), args.top), flush=True)
            print(flush=True)
            last_print = e.t


if __name__ == "__main__":
    main()
