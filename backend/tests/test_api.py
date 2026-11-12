import io
import json
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import config, deps
from app.main import app as fastapi_app
from services.alert_service import AlertStore


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(deps, "alerts", AlertStore(tmp_path / "alertas.db"))
    monkeypatch.setattr(config, "UPLOAD_DIR", tmp_path / "envios")
    monkeypatch.setattr(deps, "patient_context", {})
    return TestClient(fastapi_app)


@pytest.fixture()
def stay_id():
    return deps.clinical_service.suggested_stay()


def wav_bytes(seconds=1.0):
    samples = (np.sin(np.arange(int(16000 * seconds)) * 0.1) * 8000).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(samples.tobytes())
    return buffer.getvalue()


def monitor_steps(client, stay):
    response = client.get(f"/api/v1/pacientes/{stay}/monitor?intervalo=0")
    assert response.status_code == 200
    return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]


def test_home_page_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "OmniCare AI" in response.text


def test_status_is_honest_about_what_is_not_connected(client, monkeypatch):
    monkeypatch.setattr(deps.audio_service, "speech_key", "")
    status = client.get("/api/v1/status").json()
    assert status["marcha"]["ativo"] and status["sinais_vitais_prescricoes"]["ativo"]
    assert not status["azure_speech"]["ativo"]
    monkeypatch.setattr(deps.voice_signal_service, "model_path", config.BASE_DIR / "nao_existe.joblib")
    assert not client.get("/api/v1/status").json()["sinal_vocal"]["ativo"]


def test_unknown_stay_is_404(client):
    assert client.get("/api/v1/pacientes/1/linha-do-tempo").status_code == 404
    assert client.get("/api/v1/pacientes/1/monitor").status_code == 404
    assert client.post("/api/v1/pacientes/1/marcha/exemplo/normal").status_code == 404


def test_timeline_has_series_and_report(client, stay_id):
    data = client.get(f"/api/v1/pacientes/{stay_id}/linha-do-tempo").json()
    assert len(data["horas"]) == len(data["series"]["HR"])
    assert str(stay_id) in data["laudo"]
    assert json.dumps(data, allow_nan=False)


def test_all_stays_produce_valid_json(client):
    assert len(client.get("/api/v1/pacientes").json()["internacoes"]) == 140


def test_monitor_raises_alert_when_risk_rises_and_replay_does_not_duplicate(client, stay_id):
    steps = monitor_steps(client, stay_id)
    assert steps[-1] == {"fim": True}
    raised = [step["alerta"]["id"] for step in steps[:-1] if step["alerta"]]
    stored = client.get("/api/v1/alertas").json()["alertas"]
    assert raised and len(stored) == len(set(raised))
    monitor_steps(client, stay_id)
    assert len(client.get("/api/v1/alertas").json()["alertas"]) == len(stored)


def test_gait_alert_changes_risk_and_can_be_confirmed(client, stay_id):
    normal = client.post(f"/api/v1/pacientes/{stay_id}/marcha/exemplo/normal").json()
    altered = client.post(f"/api/v1/pacientes/{stay_id}/marcha/exemplo/alterada").json()
    assert not normal["resultado"]["alerta"]
    assert altered["resultado"]["alerta"]
    assert altered["risco"]["pontuacao"] == normal["risco"]["pontuacao"] + 2
    assert altered["alerta"]["origem"] == "marcha"
    assert "tronco" in altered["laudo"].lower()

    alert_id = altered["alerta"]["id"]
    assert client.post(f"/api/v1/alertas/{alert_id}/confirmar", json={"responsavel": "  "}).status_code == 422
    confirmed = client.post(f"/api/v1/alertas/{alert_id}/confirmar", json={"responsavel": "Equipe A"}).json()
    assert confirmed["confirmado_por"] == "Equipe A"
    assert client.post("/api/v1/alertas/999/confirmar", json={"responsavel": "Equipe A"}).status_code == 404


def test_unknown_example_is_404(client, stay_id):
    assert client.post(f"/api/v1/pacientes/{stay_id}/marcha/exemplo/outro").status_code == 404


def test_gait_csv_upload_runs_and_removes_temp_file(client, stay_id):
    content = (config.GAIT_EXAMPLES["normal"]).read_bytes()
    import gzip
    response = client.post(f"/api/v1/pacientes/{stay_id}/marcha", files={"arquivo": ("meu.csv", gzip.decompress(content), "text/csv")})
    assert response.status_code == 200
    assert response.json()["resultado"]["arquivo"] == "meu.csv"
    assert list(config.UPLOAD_DIR.iterdir()) == []


def test_invalid_csv_gets_clear_message_and_removes_temp_file(client, stay_id):
    response = client.post(f"/api/v1/pacientes/{stay_id}/marcha", files={"arquivo": ("ruim.csv", b"a,b\n1,2\n", "text/csv")})
    assert response.status_code == 422
    assert "colunas" in response.json()["detail"]
    assert list(config.UPLOAD_DIR.iterdir()) == []


def test_invalid_video_gets_clear_message(client, stay_id):
    response = client.post(f"/api/v1/pacientes/{stay_id}/marcha", files={"arquivo": ("ruim.mp4", b"nao e video", "video/mp4")})
    assert response.status_code == 422
    assert list(config.UPLOAD_DIR.iterdir()) == []


def test_path_traversal_name_stays_in_upload_dir(client, stay_id, tmp_path):
    response = client.post(f"/api/v1/pacientes/{stay_id}/marcha", files={"arquivo": ("../../fora.csv", b"a,b\n1,2\n", "text/csv")})
    assert response.status_code == 422
    assert not (tmp_path.parent / "fora.csv").exists()
    assert not (tmp_path / "fora.csv").exists()


def test_wrong_extension_is_rejected(client, stay_id):
    response = client.post(f"/api/v1/pacientes/{stay_id}/marcha", files={"arquivo": ("x.exe", b"1", "application/octet-stream")})
    assert response.status_code == 400
    response = client.post(f"/api/v1/pacientes/{stay_id}/voz", files={"arquivo": ("x.mp3", b"1", "audio/mpeg")})
    assert response.status_code == 400


def test_file_over_limit_is_rejected_and_removed(client, stay_id, monkeypatch):
    monkeypatch.setattr(config, "MAX_CSV_BYTES", 10)
    response = client.post(f"/api/v1/pacientes/{stay_id}/marcha", files={"arquivo": ("grande.csv", b"x" * 100, "text/csv")})
    assert response.status_code == 413
    assert list(config.UPLOAD_DIR.iterdir()) == []


def test_voice_without_azure_invents_nothing(client, stay_id, monkeypatch):
    monkeypatch.setattr(deps.audio_service, "speech_key", "")
    monkeypatch.setattr(deps.audio_service, "lang_key", "")
    # O nome não pode influenciar o resultado: a regra antiga olhava a letra C do arquivo.
    data = client.post(f"/api/v1/pacientes/{stay_id}/voz", files={"arquivo": ("p01.LC.06.wav", wav_bytes(), "audio/wav")}).json()
    result = data["resultado"]
    assert result["transcricao"] is None and result["analise_texto"] is None
    assert result["alerta"] is False
    assert result["sinal_vocal"]["soma_no_risco"] is False
    assert not {"fala", "voz"} & {reason["fonte"] for reason in data["risco"]["motivos"]}
    assert list(config.UPLOAD_DIR.iterdir()) == []


def test_broken_audio_gets_clear_message(client, stay_id):
    response = client.post(f"/api/v1/pacientes/{stay_id}/voz", files={"arquivo": ("ruim.wav", b"nao e audio", "audio/wav")})
    assert response.status_code == 422
    assert list(config.UPLOAD_DIR.iterdir()) == []


def test_simulated_surgery_route_no_longer_exists(client):
    assert client.get("/api/v1/demo/cenario-cirurgico?mode=anomaly").status_code == 404
