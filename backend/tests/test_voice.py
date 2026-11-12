import io
import wave

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import config, deps
from app.main import app as fastapi_app
from services.alert_service import AlertStore, combined_risk
from services.report_service import inpatient_report
from services.voice_features import SAMPLE_RATE, calibrate_features, extract_voice_features, split_utterances
from services.voice_signal_service import VoiceInputError, VoiceSignalService


def voice_like(seconds, seed=0):
    rng = np.random.default_rng(seed)
    time = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (0.3 * np.sin(2 * np.pi * 140 * time) + 0.1 * np.sin(2 * np.pi * 420 * time) + 0.01 * rng.standard_normal(len(time))).astype(np.float32)


def recording(count, seconds=2.5, pause=1.0):
    silence = np.zeros(int(pause * SAMPLE_RATE), dtype=np.float32)
    return np.concatenate([part for index in range(count) for part in (voice_like(seconds, index), silence)])


def wav_bytes(audio):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        output.writeframes((audio * 32767).astype(np.int16).tobytes())
    return buffer.getvalue()


class StubModel:
    def __init__(self, value):
        self.value = value
        self.seen = None

    def predict_proba(self, rows):
        self.seen = np.array(rows)
        return np.column_stack([np.full(len(rows), 1 - self.value), np.full(len(rows), self.value)])


def service_with(tmp_path, calibrated=0.8, raw=0.8):
    path = tmp_path / "modelo.joblib"
    path.touch()
    service = VoiceSignalService(path)
    service._artifact = {"modelo_calibrado": StubModel(calibrated), "modelo_sem_calibracao": StubModel(raw), "limiar_decisao": 0.5, "feature_version": 1}
    return service


def test_pauses_separate_the_sentences():
    assert len(split_utterances(recording(6))) == 6


def test_short_pause_does_not_split_a_sentence():
    assert len(split_utterances(recording(2, seconds=1.0, pause=0.1))) == 1


def test_long_speech_becomes_chunks_like_the_training_sentences():
    pieces = split_utterances(voice_like(20))
    assert len(pieces) == 8
    assert all(len(piece) <= 2.5 * SAMPLE_RATE for piece in pieces)


def test_silence_has_no_sentences():
    assert split_utterances(np.zeros(SAMPLE_RATE * 5, dtype=np.float32)) == []


def test_without_baseline_is_shown_but_never_adds_points(tmp_path):
    result = service_with(tmp_path).assess(recording(3))
    assert result["avaliado"] and result["presente"]
    assert result["soma_no_risco"] is False and "não soma pontos" in result["aviso"]
    assert result["com_gravacao_em_repouso"] is False


def test_with_baseline_uses_the_calibrated_model_with_the_persons_own_values(tmp_path):
    service = service_with(tmp_path, calibrated=0.8, raw=0.1)
    audio, base = recording(2), recording(6)
    result = service.assess(audio, base)
    assert result["com_gravacao_em_repouso"] and result["soma_no_risco"] and result["pontuacao"] == 0.8
    measures = np.array([extract_voice_features(item) for item in split_utterances(base)])
    expected = calibrate_features(np.array([extract_voice_features(item) for item in split_utterances(audio)]), measures.mean(axis=0), measures.std(axis=0))
    assert service._artifact["modelo_calibrado"].seen == pytest.approx(expected)
    assert service._artifact["modelo_sem_calibracao"].seen is None


def test_below_threshold_with_baseline_adds_no_points(tmp_path):
    result = service_with(tmp_path, calibrated=0.3).assess(recording(2), recording(6))
    assert result["presente"] is False and result["soma_no_risco"] is False


def test_short_baseline_is_refused_with_clear_message(tmp_path):
    with pytest.raises(VoiceInputError, match="ao menos 5"):
        service_with(tmp_path).assess(recording(2), recording(3))


def test_audio_without_speech_is_not_evaluated(tmp_path):
    result = service_with(tmp_path).assess(np.zeros(SAMPLE_RATE * 3, dtype=np.float32))
    assert result["avaliado"] is False and "fala" in result["motivo"]


def test_real_model_loads_and_gives_a_score():
    if not deps.voice_signal_service.available():
        pytest.skip("modelo de voz não está nesta máquina")
    result = deps.voice_signal_service.assess(recording(3), recording(6))
    assert result["avaliado"] and 0 <= result["pontuacao"] <= 1 and result["frases"] == 3 and result["frases_em_repouso"] == 6


def test_risk_counts_the_voice_signal_only_when_it_may_add_points():
    at = pd.Timestamp("2026-01-01")
    speech = {"alerta": False, "descricao": "x", "sinal_vocal": {"soma_no_risco": True, "descricao": "Sinal vocal presente"}}
    assert combined_risk([], [], at, None, speech)["pontuacao"] == 1
    speech["sinal_vocal"]["soma_no_risco"] = False
    assert combined_risk([], [], at, None, speech)["pontuacao"] == 0


def test_report_says_it_is_not_a_pain_score():
    risk = {"nivel": "baixo", "pontuacao": 1}
    speech = {"descricao": "Azure não configurado", "sinal_vocal": {"avaliado": True, "descricao": "Sinal vocal presente em relação à gravação em repouso"}}
    assert "não é nota de dor" in inpatient_report(1, "2026-01-01T00:00", [], [], None, speech, risk)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(deps, "alerts", AlertStore(tmp_path / "alertas.db"))
    monkeypatch.setattr(config, "UPLOAD_DIR", tmp_path / "envios")
    monkeypatch.setattr(deps, "patient_context", {})
    monkeypatch.setattr(deps.audio_service, "speech_key", "")
    monkeypatch.setattr(deps.audio_service, "lang_key", "")
    return TestClient(fastapi_app)


def test_route_with_baseline_adds_the_voice_point_and_removes_both_files(client, tmp_path, monkeypatch):
    monkeypatch.setattr(deps.audio_service, "voice_signal", service_with(tmp_path))
    stay = deps.clinical_service.suggested_stay()
    response = client.post(
        f"/api/v1/pacientes/{stay}/voz",
        files={"arquivo": ("consulta.wav", wav_bytes(recording(2)), "audio/wav"), "base": ("repouso.wav", wav_bytes(recording(6)), "audio/wav")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["resultado"]["sinal_vocal"]["soma_no_risco"] is True
    assert "voz" in {reason["fonte"] for reason in data["risco"]["motivos"]}
    assert "não é nota de dor" in data["laudo"]
    assert list(config.UPLOAD_DIR.iterdir()) == []


def test_route_refuses_short_baseline_and_removes_both_files(client, tmp_path, monkeypatch):
    monkeypatch.setattr(deps.audio_service, "voice_signal", service_with(tmp_path))
    stay = deps.clinical_service.suggested_stay()
    response = client.post(
        f"/api/v1/pacientes/{stay}/voz",
        files={"arquivo": ("consulta.wav", wav_bytes(recording(2)), "audio/wav"), "base": ("repouso.wav", wav_bytes(recording(2)), "audio/wav")},
    )
    assert response.status_code == 422 and "ao menos 5" in response.json()["detail"]
    assert list(config.UPLOAD_DIR.iterdir()) == []


def test_route_refuses_a_baseline_of_the_wrong_type(client, tmp_path, monkeypatch):
    monkeypatch.setattr(deps.audio_service, "voice_signal", service_with(tmp_path))
    stay = deps.clinical_service.suggested_stay()
    response = client.post(
        f"/api/v1/pacientes/{stay}/voz",
        files={"arquivo": ("consulta.wav", wav_bytes(recording(2)), "audio/wav"), "base": ("repouso.mp3", b"x", "audio/mpeg")},
    )
    assert response.status_code == 400
    assert list(config.UPLOAD_DIR.iterdir()) == []


def test_file_name_does_not_change_the_result(client, tmp_path, monkeypatch):
    monkeypatch.setattr(deps.audio_service, "voice_signal", service_with(tmp_path, raw=0.2))
    stay = deps.clinical_service.suggested_stay()
    audio = wav_bytes(recording(2))
    results = [client.post(f"/api/v1/pacientes/{stay}/voz", files={"arquivo": (name, audio, "audio/wav")}).json()["resultado"]["sinal_vocal"] for name in ("p01.LC.06.wav", "p01.LW.06.wav")]
    assert results[0] == results[1]
