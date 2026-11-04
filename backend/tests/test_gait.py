import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from services.rehab_gait_service import (
    FS,
    JOINT,
    JOINTS,
    MP_LEFT_HEEL,
    MP_LEFT_HIP,
    MP_LEFT_KNEE,
    MP_LEFT_SHOULDER,
    MP_RIGHT_HEEL,
    MP_RIGHT_HIP,
    MP_RIGHT_KNEE,
    MP_RIGHT_SHOULDER,
    RehabGaitAnalyzer,
    assess_walk,
    landmarks_to_skeleton,
    load_skeleton_csv,
    longest_valid_segment,
    window_measures,
)

BACKEND = Path(__file__).resolve().parent.parent
REFERENCE = json.loads((BACKEND / "models" / "marcha_referencia.json").read_text(encoding="utf-8"))
NORMAL = BACKEND / "sample_data" / "marcha" / "caminhada_normal.csv.gz"
ALTERED = BACKEND / "sample_data" / "marcha" / "caminhada_tronco_20_graus.csv.gz"


def analyzer():
    return RehabGaitAnalyzer(BACKEND / "models" / "marcha_referencia.json")


def to_mediapipe(X, yaw_degrees=0.0):
    """Coloca um esqueleto de referência nos eixos do MediaPipe, com a câmera girada em relação ao corpo."""
    mapping = {
        MP_LEFT_HIP: "left_upper_leg", MP_RIGHT_HIP: "right_upper_leg",
        MP_LEFT_KNEE: "left_lower_leg", MP_RIGHT_KNEE: "right_lower_leg",
        MP_LEFT_HEEL: "left_foot", MP_RIGHT_HEEL: "right_foot",
    }
    yaw = np.radians(yaw_degrees)
    # Corpo x frente, y esquerda, z cima. Câmera x direita, y baixo, z longe.
    forward = np.array([np.sin(yaw), 0.0, -np.cos(yaw)])
    left = np.array([np.cos(yaw), 0.0, np.sin(yaw)])  # de frente para a câmera, a esquerda da pessoa fica à direita da imagem
    up = np.array([0.0, -1.0, 0.0])
    axes = np.stack([forward, left, up], axis=1)
    points = np.zeros((len(X), 33, 3))
    for mp_index, name in mapping.items():
        points[:, mp_index] = X[:, JOINT[name]] @ axes.T
    # O MediaPipe não tem pescoço; os dois ombros têm como ponto médio o pescoço da referência.
    neck = X[:, JOINT["neck"]] @ axes.T
    shoulder_offset = 0.18 * left
    points[:, MP_LEFT_SHOULDER] = neck + shoulder_offset
    points[:, MP_RIGHT_SHOULDER] = neck - shoulder_offset
    return points


def test_normal_example_stays_inside_reference():
    result = analyzer().analyze_skeleton(NORMAL)
    assert result["avaliada"] and not result["alerta"]
    assert result["pontuacao"] == pytest.approx(2.46, abs=0.02)


def test_altered_example_triggers_alert_and_names_trunk():
    result = analyzer().analyze_skeleton(ALTERED)
    assert result["alerta"]
    assert result["medida_mais_distante"]["medida"] == "tronco_frente_media"


def test_walk_shorter_than_window_is_not_evaluated():
    short = load_skeleton_csv(NORMAL)[: 7 * FS]
    result = assess_walk(short, REFERENCE)
    assert result == {"avaliada": False, "alerta": False, "motivo": result["motivo"]}


def test_csv_with_wrong_shape_is_rejected(tmp_path):
    bad = tmp_path / "ruim.csv"
    bad.write_text("i,a,b\n0,1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="colunas"):
        load_skeleton_csv(bad)


@pytest.mark.parametrize("yaw", [0.0, 35.0, 90.0])
def test_mediapipe_axes_recover_the_same_measures(yaw):
    X = load_skeleton_csv(NORMAL)
    points = to_mediapipe(X, yaw)
    visibility = np.ones((len(points), 33))
    recovered = landmarks_to_skeleton(points, visibility, fps=FS)
    window = 8 * FS
    expected = window_measures(X[:window])
    measured = window_measures(recovered[:window])
    for name in ("cadencia", "tronco_frente_media", "joelho_amplitude", "passo_medio", "largura_passo"):
        assert measured[name] == pytest.approx(expected[name], rel=0.05, abs=0.3), name


def test_mediapipe_resamples_to_reference_rate():
    X = load_skeleton_csv(NORMAL)[: 20 * FS]
    points = to_mediapipe(X)[::2]  # como um vídeo de 15 quadros por segundo
    recovered = landmarks_to_skeleton(points, np.ones((len(points), 33)), fps=FS / 2)
    assert len(recovered) == pytest.approx(len(X), abs=4)
    assert assess_walk(recovered, REFERENCE)["avaliada"]


def test_short_pose_gap_is_filled_and_long_gap_splits():
    times = np.arange(300) / 30
    valid = np.ones(300, dtype=bool)
    valid[100:110] = False  # 0,33 s
    assert longest_valid_segment(valid, times) == (0, 299)
    valid[150:200] = False  # 1,67 s divide em 5 s e 3,3 s
    assert longest_valid_segment(valid, times) == (0, 149)
    valid[:150] = False
    assert longest_valid_segment(valid, times) == (200, 299)


def test_hidden_legs_do_not_produce_a_result():
    points = np.zeros((300, 33, 3))
    visibility = np.ones((300, 33))
    visibility[:, MP_LEFT_HEEL] = 0
    assert landmarks_to_skeleton(points, visibility, fps=30) is None


def test_joint_count_matches_reference_format():
    assert len(JOINTS) == 23


def test_real_mediapipe_runs_on_video_without_a_person(tmp_path):
    path = tmp_path / "vazio.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
    for _ in range(20):
        writer.write(np.full((240, 320, 3), 40, dtype=np.uint8))
    writer.release()
    result = analyzer().analyze_video(path)
    assert result["avaliada"] is False and result["alerta"] is False
    assert result["motivo"] == "Nenhuma pessoa encontrada no vídeo."
    assert result["aviso"]
