import io
import json
import zipfile

import cv2
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import config, deps
from app.main import app as fastapi_app
from services.alert_service import AlertStore
from services.report_service import surgery_report
from services.surgery_video_service import CVS_FILE, DETECTOR_FILE, RULE_FILE, SurgeryVideoAnalyzer

SCENE = {"gallbladder", "tool"}
RULE = json.loads((config.BASE_DIR / "models" / RULE_FILE).read_text(encoding="utf-8"))
C2, C3 = RULE["limiares"]["C2"], RULE["limiares"]["C3"]


def frame(value=0):
    return Image.new("RGB", (64, 36), (value, value, value))


class FakeAnalyzer(SurgeryVideoAnalyzer):
    """Troca só os modelos: o valor do pixel escolhe qual pontuação cada imagem recebe."""

    def __init__(self, tmp_path, table, found=SCENE, dataset_zip=None):
        (tmp_path / RULE_FILE).write_text(json.dumps(RULE), encoding="utf-8")
        (tmp_path / DETECTOR_FILE).touch()
        (tmp_path / CVS_FILE).touch()
        super().__init__(tmp_path, dataset_zip)
        self.table, self.found = table, found

    def _predict(self, images):
        rows = [self.table[image.getpixel((0, 0))[0]] for image in images]
        return np.array(rows, dtype=float), [set(self.found) for _ in images]


REACHED = {0: (0.1, C2 + 0.05, C3 + 0.05)}
LOW_C3 = {0: (0.99, C2 + 0.05, C3 - 0.05)}


def test_deviation_when_a_criterion_never_reaches_its_threshold(tmp_path):
    result = FakeAnalyzer(tmp_path, LOW_C3).analyze_frames([frame() for _ in range(5)])
    assert result["avaliada"] and result["desvio"]
    states = {item["criterio"]: item.get("atingido") for item in result["criterios"]}
    assert states["C2"] is True and states["C3"] is False


def test_no_deviation_when_both_criteria_reach_the_threshold(tmp_path):
    result = FakeAnalyzer(tmp_path, REACHED).analyze_frames([frame() for _ in range(5)])
    assert result["avaliada"] and not result["desvio"]


def test_c1_does_not_change_the_decision_and_says_so(tmp_path):
    low = FakeAnalyzer(tmp_path, {0: (0.0, C2 + 0.05, C3 + 0.05)}).analyze_frames([frame() for _ in range(5)])
    high = FakeAnalyzer(tmp_path, {0: (1.0, C2 + 0.05, C3 + 0.05)}).analyze_frames([frame() for _ in range(5)])
    assert low["desvio"] == high["desvio"] == False  # noqa: E712
    c1 = next(item for item in low["criterios"] if item["criterio"] == "C1")
    assert c1["usado"] is False and "pico" not in c1
    assert "C1" not in low["series"]
    assert "C1 não entra na decisão" in surgery_report("Teste", low)


def test_peak_uses_the_moving_average_of_three_frames(tmp_path):
    # Um único frame alto, entre frames baixos, não basta depois da média móvel.
    table = {0: (0, 0.0, 0.0), 1: (0, 1.0, 1.0)}
    frames = [frame(0), frame(0), frame(1), frame(0), frame(0)]
    result = FakeAnalyzer(tmp_path, table).analyze_frames(frames)
    assert max(result["series"]["C2"]) == pytest.approx(1 / 3, abs=1e-3)
    assert result["desvio"]


def test_more_than_one_batch_keeps_every_frame_in_order(tmp_path):
    table = {0: (0, 0.2, 0.2), 1: (0, 0.9, 0.9)}
    frames = [frame(0)] * 40 + [frame(1)] * 3
    result = FakeAnalyzer(tmp_path, table).analyze_frames(frames)
    assert result["frames_analisados"] == 43 and len(result["series"]["C2"]) == 43
    assert result["tempo_s"][-1] == 42 * RULE["intervalo_s"]
    assert result["series"]["C2"][0] == pytest.approx(0.2) and result["series"]["C2"][-1] == pytest.approx(0.9)


def test_too_few_frames_is_not_evaluated(tmp_path):
    result = FakeAnalyzer(tmp_path, REACHED).analyze_frames([frame(), frame()])
    assert not result["avaliada"] and "segundos" in result["motivo"]
    assert "desvio" not in result


def test_video_of_something_else_is_not_evaluated(tmp_path):
    result = FakeAnalyzer(tmp_path, REACHED, found=set()).analyze_frames([frame() for _ in range(5)])
    assert not result["avaliada"] and "colecistectomia" in result["motivo"]


def write_video(path, seconds, fps=10):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 36))
    assert writer.isOpened()
    for _ in range(seconds * fps):
        writer.write(np.zeros((36, 64, 3), dtype=np.uint8))
    writer.release()


def test_video_is_sampled_every_five_seconds(tmp_path):
    path = tmp_path / "video.mp4"
    write_video(path, 31)
    result = FakeAnalyzer(tmp_path, REACHED).analyze_video(path)
    assert result["frames_analisados"] == 7  # 0, 5, 10, 15, 20, 25 e 30 segundos


def test_analysis_stops_at_the_informed_clip_moment(tmp_path):
    path = tmp_path / "video.mp4"
    write_video(path, 31)
    result = FakeAnalyzer(tmp_path, REACHED).analyze_video(path, clip_at=12)
    assert result["frames_analisados"] == 3 and result["fim_informado_s"] == 12


def test_unreadable_video_raises_value_error(tmp_path):
    path = tmp_path / "quebrado.mp4"
    path.write_bytes(b"isto nao e um video")
    with pytest.raises(ValueError):
        FakeAnalyzer(tmp_path, REACHED).analyze_video(path)


def dataset(tmp_path):
    rows = [
        {"split": "train", "video_id": 1, "frame": 125, "file_name": "1_125.jpg", "C1": 1.0, "C2": 1.0, "C3": 1.0},
        *[{"split": "test", "video_id": 162, "frame": 125 * i, "file_name": f"162_{125 * i}.jpg", "C1": 0.67, "C2": 0.33, "C3": 0.0} for i in range(4)],
        *[{"split": "test", "video_id": 166, "frame": 125 * i, "file_name": f"166_{125 * i}.jpg", "C1": 1.0, "C2": 1.0, "C3": 1.0} for i in range(4)],
    ]
    path = tmp_path / "base.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("endoscapes_omnicare/cvs_labels.csv", pd.DataFrame(rows).to_csv(index=False))
        for row in rows[1:]:
            buffer = io.BytesIO()
            frame().save(buffer, "JPEG")
            archive.writestr(f"endoscapes_omnicare/images/test/{row['file_name']}", buffer.getvalue())
    return path


def test_dataset_lists_only_test_videos_and_reports_expert_labels(tmp_path):
    analyzer = FakeAnalyzer(tmp_path, REACHED, dataset_zip=dataset(tmp_path))
    assert analyzer.dataset_videos() == [162, 166]
    result = analyzer.analyze_dataset_video(162)
    assert result["frames_analisados"] == 4
    assert result["especialistas"] == {"C1": True, "C2": False, "C3": False, "cvs_atingido": False}
    assert analyzer.analyze_dataset_video(166)["especialistas"]["cvs_atingido"] is True


def test_dataset_without_zip_lists_nothing(tmp_path):
    assert FakeAnalyzer(tmp_path, REACHED, dataset_zip=tmp_path / "nao_existe.zip").dataset_videos() == []


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(deps, "alerts", AlertStore(tmp_path / "alertas.db"))
    monkeypatch.setattr(config, "UPLOAD_DIR", tmp_path / "envios")
    return TestClient(fastapi_app)


def use(monkeypatch, tmp_path, table, **options):
    models = tmp_path / "modelos"
    models.mkdir()
    monkeypatch.setattr(deps, "surgery_service", FakeAnalyzer(models, table, **options))


def test_info_route_shows_test_performance_and_is_honest_without_models(client, tmp_path, monkeypatch):
    monkeypatch.setattr(deps, "surgery_service", SurgeryVideoAnalyzer(config.BASE_DIR / "models", tmp_path / "nao_existe.zip"))
    monkeypatch.setattr(SurgeryVideoAnalyzer, "models_available", lambda self: False)
    data = client.get("/api/v1/cirurgia").json()
    assert data["modelos_ativos"] is False and data["videos_base"] == []
    assert data["desempenho"]["videos"] == 40
    response = client.post("/api/v1/cirurgia/video", files={"arquivo": ("a.mp4", b"x", "video/mp4")})
    assert response.status_code == 503


def test_dataset_route_raises_one_attention_alert_even_if_repeated(client, tmp_path, monkeypatch):
    use(monkeypatch, tmp_path, LOW_C3, dataset_zip=dataset(tmp_path))
    for _ in range(2):
        data = client.post("/api/v1/cirurgia/base/162").json()
    assert data["alerta"]["nivel"] == "atencao" and data["alerta"]["origem"] == "cirurgia"
    assert "C3" in data["alerta"]["motivos"][0]["descricao"]
    assert len(client.get("/api/v1/alertas").json()["alertas"]) == 1
    assert client.post("/api/v1/cirurgia/base/9999").status_code == 404


def test_no_alert_without_deviation(client, tmp_path, monkeypatch):
    use(monkeypatch, tmp_path, REACHED, dataset_zip=dataset(tmp_path))
    assert client.post("/api/v1/cirurgia/base/166").json()["alerta"] is None
    assert client.get("/api/v1/alertas").json()["alertas"] == []


def test_not_evaluated_video_raises_no_alert(client, tmp_path, monkeypatch):
    use(monkeypatch, tmp_path, LOW_C3, found=set())
    path = tmp_path / "outro.mp4"
    write_video(path, 31)
    data = client.post("/api/v1/cirurgia/video", files={"arquivo": ("outro.mp4", path.read_bytes(), "video/mp4")}).json()
    assert data["resultado"]["avaliada"] is False and data["alerta"] is None
    assert "não avaliada" in data["laudo"]


def test_upload_keeps_no_file_returns_no_image_and_uses_clip_moment(client, tmp_path, monkeypatch):
    use(monkeypatch, tmp_path, LOW_C3)
    path = tmp_path / "cirurgia.mp4"
    write_video(path, 31)
    response = client.post("/api/v1/cirurgia/video", data={"fim_s": "12"}, files={"arquivo": ("../cirurgia.mp4", path.read_bytes(), "video/mp4")})
    assert response.status_code == 200
    data = response.json()
    assert data["resultado"]["frames_analisados"] == 3
    assert data["alerta"]["paciente"] == "Cirurgia, cirurgia.mp4"
    assert list(config.UPLOAD_DIR.iterdir()) == []
    text = response.text.lower()
    assert "base64" not in text and "image" not in text and ".jpg" not in text


def test_upload_rejects_other_types_and_bad_clip_moment(client, tmp_path, monkeypatch):
    use(monkeypatch, tmp_path, REACHED)
    assert client.post("/api/v1/cirurgia/video", files={"arquivo": ("a.txt", b"x", "text/plain")}).status_code == 400
    assert client.post("/api/v1/cirurgia/video", data={"fim_s": "-1"}, files={"arquivo": ("a.mp4", b"x", "video/mp4")}).status_code == 422


def test_broken_upload_returns_422_and_is_removed(client, tmp_path, monkeypatch):
    use(monkeypatch, tmp_path, REACHED)
    response = client.post("/api/v1/cirurgia/video", files={"arquivo": ("a.mp4", b"nao e video", "video/mp4")})
    assert response.status_code == 422
    assert list(config.UPLOAD_DIR.iterdir()) == []


@pytest.mark.skipif(not SurgeryVideoAnalyzer(config.BASE_DIR / "models").models_available(), reason="pesos treinados não estão nesta máquina")
def test_real_models_refuse_a_blank_video(tmp_path):
    path = tmp_path / "branco.mp4"
    write_video(path, 31)
    result = SurgeryVideoAnalyzer(config.BASE_DIR / "models").analyze_video(path)
    assert result["frames_analisados"] == 7
    assert result["avaliada"] is False and "colecistectomia" in result["motivo"]
