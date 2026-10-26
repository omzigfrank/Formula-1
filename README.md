# Formula-1 live predictions

Live, per-lap predictions from F1 telemetry and timing, built on the [OpenF1](https://openf1.org) API.
Runs at home on any machine with Python 3.11+. No subscription is needed to develop and test: any
finished session since 2023 can be replayed as if it were live.

What it predicts for every driver, refreshed every few seconds:

| Column | Meaning | Source |
|---|---|---|
| `PIT%` | probability the driver pits on the next lap | trained model + pit-lane speed signature from telemetry + SC bump |
| `OVT%` | probability the driver gains a position by the end of the next lap | trained model + live DRS state |
| `PROJ` | projected next lap time | last lap adjusted by the 5-lap degradation slope |
| `FLAGS` | `PIT LANE`, `DRS`, `ATTACK` (<0.5 s behind), `SC`/`VSC`, `STOPPED?` | telemetry and race control |

## Quick start

```bash
pip install -r requirements.txt

# 1. Train on history (free tier, ~2 min per race because of rate limits; resumable, cached in data/)
python -m f1live.train --years 2025 --max-races 12

# 2. Replay a past race as if live (free). 9839 = 2025 Abu Dhabi GP. --speed 20 = 20x real time.
python -m f1live.run --replay 9839 --speed 20 --offset 1500 --refresh 30

# 3. Real race weekend (requires OpenF1 Sponsor tier, EUR 9.90/mo, for live data)
export OPENF1_TOKEN=...
python -m f1live.run --live
```

Find session keys with `curl "https://api.openf1.org/v1/sessions?year=2025&session_name=Race"`.

## How it works

```
OpenF1 REST  ->  feed.py (live poll or replay clock)  ->  state.py (per-driver rolling state)
             ->  features.py (lap features + telemetry features)  ->  models.py  ->  run.py table
```

* `feed.py` polls `car_data` (speed, throttle, brake, gear, RPM, DRS at ~3.7 Hz), `intervals`,
  `position`, `laps`, `pit`, `stints`, `race_control`, `weather`. Replay and live produce the same
  event stream, so the rest of the pipeline is identical in both modes.
* `train.py` builds one row per completed lap from historical races (tire age, stint lap, degradation
  slope, interval ahead and behind, position, SC, rain, compound) and labels it with "pitted next lap"
  and "gained a position next lap". Two gradient-boosting classifiers are saved to `models/`.
* `models.py` scores the live state with those classifiers, then layers telemetry signals on top:
  a sustained 40 to 90 km/h speed band with throttle applied means the car is in the pit lane, an open
  DRS flap within 1.0 s raises the overtake probability, a safety car raises pit probability.

Cross-validated results on 12 races from 2025 (grouped by race):

| Target | Base rate | AUC | Average precision |
|---|---|---|---|
| pit next lap | 3.8% | 0.69 | 0.08 |
| overtake next lap | 12.5% | 0.73 | 0.33 |

These are honest numbers for lap-level features. The models rank well but pit timing is inherently
noisy without team strategy information. Train on more races (`--years 2023 2024 2025`) before trusting them.

## Known limits

* OpenF1 free tier is historical only and rate limited. Do not run training and a replay at the same time.
* Live `car_data` and `position` feeds on OpenF1 have had outages during sessions this season.
* Telemetry arrives 30 to 60 s behind the live broadcast. Predictions beat the US TV delay, not the trackside clock.
* `laps` rows only appear once a lap is complete, so lap-level features lag by one lap. Telemetry flags do not.

## Tests

```bash
pip install pytest && python -m pytest -q
```
