from pathlib import Path

import numpy as np
import pandas as pd

from services.clinical_anomaly_service import (
    ClinicalTimelineService,
    flagged_hours,
    prescription_events,
    remove_invalid_values,
    rolling_z,
    stay_z_scores,
    vital_events,
)

DEMO_DIR = Path(__file__).resolve().parent.parent / "sample_data" / "mimic_demo"


def stay_with_jump(jump_at=30, hours=48, jump=60.0):
    index = pd.date_range("2150-01-01", periods=hours, freq="h")
    rng = np.random.default_rng(0)
    hr = 80 + rng.normal(0, 2, hours)
    hr[jump_at:] += jump
    frame = pd.DataFrame(index=index)
    frame["HR"] = hr
    for name, value in (("RR", 18.0), ("SPO2", 97.0), ("SBP", 120.0), ("TEMP", 37.0)):
        frame[name] = value + rng.normal(0, 0.1, hours)
    return frame


def rx_orders(doses, drug="Furosemide"):
    return pd.DataFrame({
        "starttime": pd.date_range("2150-01-01", periods=len(doses), freq="12h"),
        "drug_type": "MAIN",
        "drug": drug,
        "dose_val_rx": [str(dose) for dose in doses],
        "dose_unit_rx": "mg",
        "route": "IV",
    })


def test_sensor_errors_become_missing():
    frame = pd.DataFrame({"HR": [0, 80], "RR": [18, 0], "SPO2": [29, 97], "SBP": [10, 120], "TEMP": [37, 90]})
    clean = remove_invalid_values(frame)
    assert clean["HR"].isna().tolist() == [True, False]
    assert clean["RR"].isna().tolist() == [False, True]
    assert clean["SPO2"].isna().tolist() == [True, False]
    assert clean["TEMP"].isna().tolist() == [False, True]


def test_score_does_not_use_the_future():
    stay = stay_with_jump()
    before = rolling_z(stay["HR"], "HR").iloc[:25]
    changed = stay["HR"].copy()
    changed.iloc[25:] = 500
    after = rolling_z(changed, "HR").iloc[:25]
    pd.testing.assert_series_equal(before, after)


def test_sustained_jump_creates_event_on_second_hour():
    stay = stay_with_jump(jump_at=30)
    events = vital_events(stay, threshold=3.0)
    assert [event["sinal"] for event in events] == ["HR"]
    assert pd.Timestamp(events[0]["inicio"]) == stay.index[31]
    assert "subiu" in events[0]["descricao"]


def test_single_outlier_does_not_create_event():
    stay = stay_with_jump(jump=0)
    stay.iloc[30, stay.columns.get_loc("HR")] = 180
    assert vital_events(stay, threshold=3.0) == []


def test_high_oxygen_is_not_an_anomaly():
    z = pd.Series([6.0, 6.0, 6.0], index=pd.date_range("2150-01-01", periods=3, freq="h"))
    assert not flagged_hours(z, "SPO2", 3.0).any()
    assert flagged_hours(-z, "SPO2", 3.0).any()


def test_stable_patient_scale_has_a_floor():
    index = pd.date_range("2150-01-01", periods=20, freq="h")
    stay = pd.DataFrame({name: 80.0 for name in ("HR", "RR", "SPO2", "SBP", "TEMP")}, index=index)
    stay.iloc[10:, stay.columns.get_loc("HR")] = 82.0
    assert stay_z_scores(stay)["HR"].abs().max() < 1


def test_tenfold_dose_is_high_severity():
    events = prescription_events(rx_orders([40, 400]))
    assert len(events) == 1
    assert events[0]["gravidade"] == "alta"
    assert events[0]["razao_dose"] == 10.0


def test_same_dose_and_small_change_are_ignored():
    assert prescription_events(rx_orders([40, 40, 60])) == []


def test_saline_flush_is_ignored():
    assert prescription_events(rx_orders([3, 300], drug="Sodium Chloride 0.9%  Flush")) == []


def test_text_dose_does_not_break():
    assert prescription_events(rx_orders(["1-2", "3-6"])) == []


def test_late_high_alert_start_is_flagged():
    orders = pd.concat([rx_orders([40]), rx_orders([5], drug="Heparin").assign(starttime=pd.Timestamp("2150-01-03"))])
    events = prescription_events(orders)
    assert [event["tipo"] for event in events] == ["inicio_alta_vigilancia"]


def test_replay_reaches_the_same_events_as_the_full_timeline():
    service = ClinicalTimelineService(DEMO_DIR)
    stay_id = service.suggested_stay()
    full = service.timeline(stay_id)
    steps = list(service.replay(stay_id))
    assert [step["hora"] for step in steps] == full["horas"]
    replayed = {(event["sinal"], event["inicio"]) for event in steps[-1]["eventos_sinais_vitais"]}
    assert replayed == {(event["sinal"], event["inicio"]) for event in full["eventos_sinais_vitais"]}
    assert len(steps[-1]["eventos_prescricao"]) == len(full["eventos_prescricao"])


def test_replay_only_shows_values_known_at_that_hour():
    service = ClinicalTimelineService(DEMO_DIR)
    stay_id = service.suggested_stay()
    for step in service.replay(stay_id):
        for event in step["eventos_sinais_vitais"]:
            assert event["inicio"] <= step["hora"] and event["fim"] <= step["hora"]
        for event in step["eventos_prescricao"]:
            assert event["inicio"] <= step["hora"]


def test_exported_demo_has_no_subject_id():
    service = ClinicalTimelineService(DEMO_DIR)
    assert "subject_id" not in service.vitals.columns
    assert "subject_id" not in service.prescriptions.columns
