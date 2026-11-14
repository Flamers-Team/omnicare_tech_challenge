import pandas as pd

from services.alert_service import AlertStore, combined_risk

NOW = pd.Timestamp("2150-01-02 12:00")
GAIT_ALERT = {"alerta": True, "medida_mais_distante": {"descricao": "Inclinação média do tronco"}}


def vital(signal="HR", start="2150-01-02 10:00", end="2150-01-02 11:00"):
    return {"sinal": signal, "inicio": start, "fim": end, "descricao": f"{signal} mudou"}


def rx(severity="alta", start="2150-01-02 00:00"):
    return {"gravidade": severity, "inicio": start, "descricao": "dose mudou"}


def test_no_signal_is_low():
    risk = combined_risk([], [], NOW)
    assert (risk["pontuacao"], risk["nivel"], risk["motivos"]) == (0, "baixo", [])


def test_one_vital_is_attention():
    risk = combined_risk([vital()], [], NOW)
    assert (risk["pontuacao"], risk["nivel"]) == (2, "atencao")


def test_vital_plus_gait_is_high_and_lists_each_reason():
    risk = combined_risk([vital()], [], NOW, gait=GAIT_ALERT)
    assert (risk["pontuacao"], risk["nivel"]) == (4, "alto")
    assert {reason["fonte"] for reason in risk["motivos"]} == {"sinais_vitais", "marcha"}


def test_vital_points_are_capped():
    risk = combined_risk([vital("HR"), vital("RR"), vital("SBP")], [], NOW)
    assert risk["pontuacao"] == 4
    assert sum(reason["pontos"] for reason in risk["motivos"]) == 4


def test_old_vital_event_no_longer_counts():
    risk = combined_risk([vital(start="2150-01-01 00:00", end="2150-01-01 02:00")], [], NOW)
    assert risk["pontuacao"] == 0


def test_future_events_do_not_count():
    risk = combined_risk([vital(start="2150-01-03 00:00", end="2150-01-03 01:00")], [rx(start="2150-01-03 00:00")], NOW)
    assert risk["pontuacao"] == 0


def test_moderate_prescription_alone_stays_low():
    assert combined_risk([], [rx("moderada")], NOW)["nivel"] == "baixo"


def test_prescription_points_are_capped_and_high_comes_first():
    risk = combined_risk([], [rx("moderada"), rx("alta"), rx("alta")], NOW)
    assert risk["pontuacao"] == 2
    assert len(risk["motivos"]) == 1


def test_gait_without_alert_adds_nothing():
    assert combined_risk([], [], NOW, gait={"alerta": False})["pontuacao"] == 0


def test_store_does_not_duplicate_and_keeps_first_confirmation(tmp_path):
    store = AlertStore(tmp_path / "alertas.db")
    risk = combined_risk([vital()], [], NOW)
    first = store.add("Internação 1", risk, "laudo", "2150-01-02T12:00:00")
    second = store.add("Internação 1", risk, "laudo", "2150-01-02T12:00:00")
    assert first["id"] == second["id"]
    assert len(store.list()) == 1

    confirmed = store.confirm(first["id"], "Ana")
    assert confirmed["confirmado_por"] == "Ana" and confirmed["confirmado_em"]
    assert store.confirm(first["id"], "Bia")["confirmado_por"] == "Ana"
    assert store.confirm(999, "Ana") is None


def test_same_moment_from_another_origin_is_a_new_alert(tmp_path):
    store = AlertStore(tmp_path / "alertas.db")
    risk = combined_risk([vital()], [], NOW)
    store.add("Internação 1", risk, "laudo", "2150-01-02T12:00:00", "monitor")
    store.add("Internação 1", risk, "laudo", "2150-01-02T12:00:00", "marcha")
    assert len(store.list()) == 2


def test_clear_removes_everything_and_restarts_numbering(tmp_path):
    store = AlertStore(tmp_path / "alertas.db")
    risk = combined_risk([vital()], [], NOW)
    store.add("Internação 1", risk, "laudo", "2150-01-02T12:00:00", "monitor")
    store.add("Internação 1", risk, "laudo", "2150-01-02T12:00:00", "marcha")
    version = store.version
    assert store.count() == 2

    assert store.clear() == 2
    assert store.count() == 0 and store.list() == []
    assert store.version > version  # quem acompanha em tempo real é avisado
    assert store.add("Internação 1", risk, "laudo", "2150-01-02T12:00:00")["id"] == 1
    assert store.clear() == 1 and store.clear() == 0


def test_clear_is_permanent_after_restart(tmp_path):
    path = tmp_path / "alertas.db"
    store = AlertStore(path)
    store.add("Internação 1", combined_risk([vital()], [], NOW), "laudo", "2150-01-02T12:00:00")
    store.clear()
    assert AlertStore(path).list() == []


def test_history_survives_restart(tmp_path):
    path = tmp_path / "alertas.db"
    AlertStore(path).add("Internação 1", combined_risk([vital()], [], NOW), "laudo", "2150-01-02T12:00:00")
    assert len(AlertStore(path).list()) == 1
