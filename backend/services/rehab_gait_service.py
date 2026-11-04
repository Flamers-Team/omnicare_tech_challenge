"""Avaliação da marcha com a referência calculada no Multi-Gait-Posture."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

FS = 30
JOINTS = (
    "pelvis", "l5", "l3", "t12", "t8", "neck", "head", "right_shoulder", "right_upper_arm", "right_forearm",
    "right_hand", "left_shoulder", "left_upper_arm", "left_forearm", "left_hand", "right_upper_leg",
    "right_lower_leg", "right_foot", "right_toe", "left_upper_leg", "left_lower_leg", "left_foot", "left_toe",
)
JOINT = {name: index for index, name in enumerate(JOINTS)}
FILTER_B, FILTER_A = butter(4, 6 / (FS / 2))  # remove oscilações acima de 6 Hz

# Índices dos pontos do corpo no MediaPipe Pose.
MP_LEFT_SHOULDER, MP_RIGHT_SHOULDER = 11, 12
MP_LEFT_HIP, MP_RIGHT_HIP = 23, 24
MP_LEFT_KNEE, MP_RIGHT_KNEE = 25, 26
MP_LEFT_HEEL, MP_RIGHT_HEEL = 29, 30
MP_REQUIRED = (
    MP_LEFT_SHOULDER, MP_RIGHT_SHOULDER, MP_LEFT_HIP, MP_RIGHT_HIP,
    MP_LEFT_KNEE, MP_RIGHT_KNEE, MP_LEFT_HEEL, MP_RIGHT_HEEL,
)
MIN_VISIBILITY = 0.5
MAX_GAP_SECONDS = 0.5
MAX_VIDEO_SECONDS = 120

SENSOR_SOURCE = "Esqueleto 3D de sensores (formato Multi-Gait-Posture)"
VIDEO_SOURCE = "Vídeo com MediaPipe Pose"
VIDEO_WARNING = "A referência veio de sensores no corpo. O uso com vídeo comum ainda não foi validado."


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    u, v = a - b, c - b
    cosine = (u * v).sum(axis=1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1))
    return np.degrees(np.arccos(np.clip(cosine, -1, 1)))


def window_measures(X: np.ndarray) -> dict[str, float] | None:
    """Calcula as medidas de uma janela. Eixos: x para frente, y para a esquerda, z para cima."""
    X = filtfilt(FILTER_B, FILTER_A, X, axis=0)
    hip_l, knee_l, heel_l = (X[:, JOINT[f"left_{name}"]] for name in ("upper_leg", "lower_leg", "foot"))
    hip_r, knee_r, heel_r = (X[:, JOINT[f"right_{name}"]] for name in ("upper_leg", "lower_leg", "foot"))
    leg = np.mean([
        np.linalg.norm(hip_l - knee_l, axis=1).mean() + np.linalg.norm(knee_l - heel_l, axis=1).mean(),
        np.linalg.norm(hip_r - knee_r, axis=1).mean() + np.linalg.norm(knee_r - heel_r, axis=1).mean(),
    ])

    separation = (heel_l[:, 0] - heel_r[:, 0]) / leg
    left, _ = find_peaks(separation, distance=int(0.4 * FS), prominence=0.15)
    right, _ = find_peaks(-separation, distance=int(0.4 * FS), prominence=0.15)
    if len(left) < 2 or len(right) < 2:
        return None

    events = sorted([(i, "L") for i in left] + [(i, "R") for i in right])
    step_times, left_times, right_times = [], [], []
    for (start, start_side), (end, end_side) in zip(events, events[1:]):
        step_times.append((end - start) / FS)
        if start_side != end_side:
            (left_times if end_side == "L" else right_times).append((end - start) / FS)
    if not left_times or not right_times:
        return None

    left_lengths, right_lengths = separation[left], -separation[right]
    all_lengths = np.concatenate([left_lengths, right_lengths])
    mean_length = (left_lengths.mean() + right_lengths.mean()) / 2
    left_time, right_time = np.mean(left_times), np.mean(right_times)

    spine = X[:, JOINT["neck"]] - X[:, JOINT["pelvis"]]
    trunk_front = np.degrees(np.arctan2(spine[:, 0], spine[:, 2]))
    trunk_side = np.degrees(np.arctan2(spine[:, 1], spine[:, 2]))
    knee_left = np.ptp(joint_angle(hip_l, knee_l, heel_l))
    knee_right = np.ptp(joint_angle(hip_r, knee_r, heel_r))

    return {
        "passo_medio": mean_length,
        "assimetria_passo": abs(left_lengths.mean() - right_lengths.mean()) / mean_length,
        "assimetria_tempo": abs(left_time - right_time) / ((left_time + right_time) / 2),
        "cadencia": 60 / np.mean(step_times),
        "cv_tempo": np.std(step_times) / np.mean(step_times),
        "cv_passo": all_lengths.std() / all_lengths.mean(),
        "largura_passo": np.abs(heel_l[:, 1] - heel_r[:, 1]).mean() / leg,
        "tronco_frente_media": trunk_front.mean(),
        "tronco_frente_dp": trunk_front.std(),
        "tronco_lado_abs": abs(trunk_side.mean()),
        "tronco_lado_dp": trunk_side.std(),
        "joelho_amplitude": (knee_left + knee_right) / 2,
        "assimetria_joelho": abs(knee_left - knee_right) / ((knee_left + knee_right) / 2),
    }


def walk_windows(X: np.ndarray, reference: dict) -> list[dict[str, float]]:
    window = reference["janela_s"] * FS
    hop = reference["passo_janela_s"] * FS
    rows = []
    for start in range(0, X.shape[0] - window + 1, hop):
        measures = window_measures(X[start:start + window])
        if measures is not None:
            rows.append(measures)
    return rows


def assess_walk(X: np.ndarray, reference: dict) -> dict:
    """Compara a caminhada com a referência; a nota é a mediana entre as janelas."""
    rows = walk_windows(X, reference)
    if not rows:
        return {"avaliada": False, "alerta": False, "motivo": f"Caminhada com menos de {reference['janela_s']} segundos ou sem passos suficientes."}

    names = reference["medidas"]
    measures = pd.DataFrame(rows)[names].to_numpy()
    median = np.array([reference["mediana"][name] for name in names])
    scale = np.array([reference["escala"][name] for name in names])
    distances = np.abs((measures - median) / scale)
    score = float(np.median(distances.max(axis=1)))
    limit = reference["limiar"]
    per_measure = np.median(distances, axis=0)
    typical = np.median(measures, axis=0)

    ranked = [
        {
            "medida": name,
            "descricao": reference["descricao"][name],
            "valor": round(float(value), 3),
            "referencia": reference["mediana"][name],
            "desvio": round(float(deviation), 2),
        }
        for name, value, deviation in sorted(zip(names, typical, per_measure), key=lambda item: -item[2])
    ]
    return {
        "avaliada": True,
        "alerta": score > limit,
        "pontuacao": round(score, 2),
        "limiar": limit,
        "janelas": len(rows),
        "medidas_fora_da_faixa": [item for item in ranked if item["desvio"] > limit],
        "medida_mais_distante": ranked[0],
    }


def load_skeleton_csv(path: str | Path) -> np.ndarray:
    values = pd.read_csv(path, index_col=0).to_numpy(dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(JOINTS) * 3:
        raise ValueError(f"O CSV precisa ter {len(JOINTS) * 3} colunas de posição, no formato normalized_skeleton_3d.csv.")
    if not np.isfinite(values).all():
        raise ValueError("O CSV tem valores vazios ou inválidos.")
    return values.reshape(-1, len(JOINTS), 3)


def longest_valid_segment(valid: np.ndarray, times: np.ndarray, max_gap: float = MAX_GAP_SECONDS) -> tuple[int, int] | None:
    """Devolve o maior trecho com pose encontrada, aceitando falhas curtas."""
    indices = np.flatnonzero(valid)
    if len(indices) < 2:
        return None
    breaks = np.flatnonzero(np.diff(times[indices]) > max_gap)
    starts = np.concatenate([[0], breaks + 1])
    ends = np.concatenate([breaks, [len(indices) - 1]])
    best = int(np.argmax(times[indices[ends]] - times[indices[starts]]))
    return int(indices[starts[best]]), int(indices[ends[best]])


def landmarks_to_skeleton(landmarks: np.ndarray, visibility: np.ndarray, fps: float) -> np.ndarray | None:
    """Converte os pontos 3D do MediaPipe para o sistema de eixos da referência, a 30 amostras por segundo.

    O MediaPipe usa x para a direita da imagem, y para baixo e z para longe da câmera. A direção
    para frente vem do próprio corpo, então a câmera não precisa estar alinhada com a caminhada.
    """
    times = np.arange(len(landmarks)) / fps
    valid = (visibility[:, list(MP_REQUIRED)] >= MIN_VISIBILITY).all(axis=1) & np.isfinite(landmarks).all(axis=(1, 2))
    segment = longest_valid_segment(valid, times)
    if segment is None:
        return None
    first, last = segment
    keep = np.flatnonzero(valid[first:last + 1]) + first
    target_times = np.arange(times[keep[0]], times[keep[-1]], 1 / FS)
    if len(target_times) < 2:
        return None
    flat = landmarks[keep].reshape(len(keep), -1)
    resampled = np.column_stack([np.interp(target_times, times[keep], flat[:, column]) for column in range(flat.shape[1])])
    points = resampled.reshape(len(target_times), landmarks.shape[1], 3)

    up = np.array([0.0, -1.0, 0.0])
    left_direction = np.median(points[:, MP_LEFT_HIP] - points[:, MP_RIGHT_HIP], axis=0)
    left_direction = left_direction - left_direction.dot(up) * up
    norm = np.linalg.norm(left_direction)
    if norm < 1e-6:
        return None
    left_direction = left_direction / norm
    forward = np.cross(left_direction, up)
    axes = np.stack([forward, left_direction, up])

    X = np.zeros((len(points), len(JOINTS), 3))
    joints = {
        "pelvis": points[:, [MP_LEFT_HIP, MP_RIGHT_HIP]].mean(axis=1),
        "neck": points[:, [MP_LEFT_SHOULDER, MP_RIGHT_SHOULDER]].mean(axis=1),
        "left_upper_leg": points[:, MP_LEFT_HIP],
        "right_upper_leg": points[:, MP_RIGHT_HIP],
        "left_lower_leg": points[:, MP_LEFT_KNEE],
        "right_lower_leg": points[:, MP_RIGHT_KNEE],
        "left_foot": points[:, MP_LEFT_HEEL],
        "right_foot": points[:, MP_RIGHT_HEEL],
    }
    for name, values in joints.items():
        X[:, JOINT[name]] = values @ axes.T
    return X


class RehabGaitAnalyzer:
    def __init__(self, reference_path: str | Path):
        self.reference = json.loads(Path(reference_path).read_text(encoding="utf-8"))

    def analyze_skeleton(self, path: str | Path) -> dict:
        return {"fonte_dados": SENSOR_SOURCE, **assess_walk(load_skeleton_csv(path), self.reference)}

    def analyze_video(self, path: str | Path) -> dict:
        landmarks, visibility, fps = self._video_landmarks(path)
        base = {"fonte_dados": VIDEO_SOURCE, "aviso": VIDEO_WARNING}
        if not np.isfinite(landmarks).any():
            return {**base, "avaliada": False, "alerta": False, "motivo": "Nenhuma pessoa encontrada no vídeo."}
        X = landmarks_to_skeleton(landmarks, visibility, fps)
        if X is None:
            return {**base, "avaliada": False, "alerta": False, "motivo": "A pose não ficou visível por tempo suficiente, incluindo ombros, quadril, joelhos e calcanhares."}
        return {**base, **assess_walk(X, self.reference)}

    @staticmethod
    def _video_landmarks(path: str | Path) -> tuple[np.ndarray, np.ndarray, float]:
        import mediapipe as mp  # carregado só quando há vídeo, porque demora a iniciar

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValueError("Não foi possível abrir o vídeo.")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            capture.release()
            raise ValueError("O vídeo não informa a taxa de quadros.")
        landmarks, visibility = [], []
        try:
            with mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5) as pose:
                for _ in range(int(MAX_VIDEO_SECONDS * fps)):
                    ok, frame = capture.read()
                    if not ok:
                        break
                    result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    if result.pose_world_landmarks:
                        points = result.pose_world_landmarks.landmark
                        landmarks.append([[p.x, p.y, p.z] for p in points])
                        visibility.append([p.visibility for p in points])
                    else:
                        landmarks.append(np.full((33, 3), np.nan))
                        visibility.append(np.zeros(33))
        finally:
            capture.release()
        return np.asarray(landmarks, dtype=np.float64), np.asarray(visibility, dtype=np.float64), float(fps)
