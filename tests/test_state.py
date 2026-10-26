from datetime import datetime, timedelta, timezone

from f1live.feed import Event
from f1live.features import from_state
from f1live.state import SessionState

T0 = datetime(2025, 12, 7, 13, 0, tzinfo=timezone.utc)
DRIVERS = [{"driver_number": 4, "name_acronym": "NOR", "team_name": "McLaren"},
           {"driver_number": 1, "name_acronym": "VER", "team_name": "Red Bull"}]


def ev(kind, secs, **data):
    return Event(kind, T0 + timedelta(seconds=secs), data)


def test_stint_and_tyre_age_ignore_future_stints():
    st = SessionState(DRIVERS)
    for lap in range(1, 6):
        st.apply(ev("lap", lap * 90, driver_number=4, lap_number=lap, lap_duration=90.0 + lap * 0.1))
    st.apply(ev("stint", 450, driver_number=4, lap_start=1, lap_end=20, compound="MEDIUM", tyre_age_at_start=2))
    st.apply(ev("stint", 450, driver_number=4, lap_start=21, lap_end=58, compound="HARD", tyre_age_at_start=0))
    d = st.d(4)
    assert d.compound == "MEDIUM"
    assert d.tyre_age == 2 + 5
    assert d.deg_slope() > 0.09


def test_pit_lane_signature_from_telemetry():
    st = SessionState(DRIVERS)
    st.apply(ev("lap", 90, driver_number=4, lap_number=1, lap_duration=90.0))
    for i in range(15):
        st.apply(ev("telemetry", 100 + i, driver_number=4, speed=80, throttle=20, brake=0, drs=0, n_gear=2))
    f = from_state(st, st.d(4))
    assert f["pit_speed_flag"] == 1.0


def test_positions_and_neighbours():
    st = SessionState(DRIVERS)
    st.apply(ev("position", 1, driver_number=1, position=1))
    st.apply(ev("position", 1, driver_number=4, position=2))
    st.apply(ev("interval", 2, driver_number=4, interval=0.8, gap_to_leader=0.8))
    st.apply(ev("interval", 2, driver_number=1, interval=None, gap_to_leader=0))
    assert st.ahead_of(4).number == 1
    assert st.behind(1).number == 4
    assert from_state(st, st.d(1))["gap_behind"] == 0.8


def test_safety_car_state():
    st = SessionState(DRIVERS)
    st.apply(ev("race_control", 5, category="SafetyCar", message="SAFETY CAR DEPLOYED", flag=None))
    assert st.track_status == "SC"
    st.apply(ev("race_control", 9, category="SafetyCar", message="SAFETY CAR IN THIS LAP", flag=None))
    assert st.track_status == "GREEN"
