"""Análise de vídeo cirúrgico com os modelos treinados no Endoscapes2023.

Nenhum frame é salvo ou devolvido: a saída tem só números e textos.
"""
from __future__ import annotations

import io
import json
import threading
import zipfile
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

DETECTOR_FILE = "endoscapes_yolov8s.pt"
CVS_FILE = "cvs_resnet18.pt"
RULE_FILE = "cvs_regra.json"
DATASET_ROOT = "endoscapes_omnicare"
CRITERIA = ["C1", "C2", "C3"]
CRITERIA_NAMES = {
    "C1": "Duas estruturas tubulares ligadas à vesícula",
    "C2": "Triângulo hepatocístico limpo",
    "C3": "Parte inferior da vesícula separada da placa cística",
}
STRUCTURE_NAMES = {
    "gallbladder": "Vesícula",
    "tool": "Instrumento",
    "calot_triangle": "Triângulo de Calot",
    "cystic_duct": "Ducto cístico",
    "cystic_artery": "Artéria cística",
    "cystic_plate": "Placa cística",
}
SCENE_STRUCTURES = {"gallbladder", "tool"}
RELIABLE_AP50 = 0.9
BATCH_SIZE = 32
MIN_FRAMES = 3
# Na validação, o menor valor em um vídeo real foi 34%. O corte só barra vídeos claramente fora do tema.
MIN_SCENE_SHARE = 0.25
LABEL_THRESHOLD = 0.5


class SurgeryVideoAnalyzer:
    def __init__(self, models_dir: str | Path, dataset_zip: str | Path | None = None):
        self.models_dir = Path(models_dir)
        self.dataset_zip = Path(dataset_zip) if dataset_zip else None
        self.rule = json.loads((self.models_dir / RULE_FILE).read_text(encoding="utf-8"))
        self._lock = threading.Lock()
        self._models = None
        self._labels: pd.DataFrame | None = None

    def models_available(self) -> bool:
        return (self.models_dir / DETECTOR_FILE).exists() and (self.models_dir / CVS_FILE).exists()

    def performance(self) -> dict:
        return self.rule["desempenho_teste"]

    def _load_models(self) -> tuple:
        with self._lock:
            if self._models is None:
                # Importados aqui para o servidor abrir rápido quando a aba de cirurgia não é usada.
                import torch
                from torch import nn
                from torchvision import models, transforms
                from ultralytics import YOLO

                checkpoint = torch.load(self.models_dir / CVS_FILE, map_location="cpu", weights_only=True)
                classifier = models.resnet18(weights=None)
                classifier.fc = nn.Linear(classifier.fc.in_features, len(CRITERIA))
                classifier.load_state_dict(checkpoint["state_dict"])
                classifier.eval()
                transform = transforms.Compose([
                    transforms.Resize(tuple(checkpoint["image_size"])),
                    transforms.ToTensor(),
                    transforms.Normalize(checkpoint["mean"], checkpoint["std"]),
                ])
                self._models = (classifier, transform, YOLO(str(self.models_dir / DETECTOR_FILE)), torch)
        return self._models

    def _predict(self, images: list[Image.Image]) -> tuple[np.ndarray, list[set[str]]]:
        """Devolve a pontuação de C1, C2 e C3 e as estruturas encontradas em cada imagem."""
        classifier, transform, detector, torch = self._load_models()
        with torch.no_grad():
            scores = torch.sigmoid(classifier(torch.stack([transform(image) for image in images]))).numpy()
        results = detector.predict(images, imgsz=640, conf=self.rule["confianca_detector"], verbose=False)
        found = [{result.names[int(index)] for index in result.boxes.cls.tolist()} for result in results]
        return scores, found

    def analyze_frames(self, frames: Iterable[Image.Image]) -> dict:
        step = self.rule["intervalo_s"]
        scores, detections, batch = [], [], []

        def flush() -> None:
            batch_scores, batch_found = self._predict(batch)
            scores.append(batch_scores)
            detections.extend(batch_found)
            batch.clear()

        for frame in frames:
            batch.append(frame)
            if len(batch) == BATCH_SIZE:
                flush()
        if batch:
            flush()

        if len(detections) < MIN_FRAMES:
            return {
                "avaliada": False,
                "motivo": f"É preciso ao menos {(MIN_FRAMES - 1) * step + 1} segundos de vídeo, com {MIN_FRAMES} frames a cada {step} segundos.",
                "frames_analisados": len(detections),
            }
        scene_share = sum(bool(found & SCENE_STRUCTURES) for found in detections) / len(detections)
        if scene_share < MIN_SCENE_SHARE:
            return {
                "avaliada": False,
                "motivo": f"O vídeo não parece ser de colecistectomia laparoscópica: vesícula ou instrumento apareceram em {scene_share:.0%} dos frames.",
                "frames_analisados": len(detections),
            }

        values = np.concatenate(scores)
        window = self.rule["janela_media"]
        thresholds = self.rule["limiares"]
        series, criteria = {}, []
        for index, name in enumerate(CRITERIA):
            if name not in thresholds:
                criteria.append({"criterio": name, "nome": CRITERIA_NAMES[name], "usado": False})
                continue
            smooth = pd.Series(values[:, index]).rolling(window, min_periods=1).mean()
            series[name] = [round(float(value), 4) for value in smooth]
            peak = float(smooth.max())
            criteria.append({
                "criterio": name,
                "nome": CRITERIA_NAMES[name],
                "usado": True,
                "pico": round(peak, 4),
                "limiar": round(thresholds[name], 4),
                "atingido": peak >= thresholds[name],
            })

        ap50 = self.rule["detector_ap50_teste"]
        structures = [
            {
                "classe": name,
                "nome": STRUCTURE_NAMES[name],
                "frames": sum(name in found for found in detections),
                "confiavel": ap50[name] >= RELIABLE_AP50,
            }
            for name in sorted(STRUCTURE_NAMES, key=lambda item: -ap50[item])
        ]
        used = [item for item in criteria if item["usado"]]
        return {
            "avaliada": True,
            "frames_analisados": len(detections),
            "intervalo_s": step,
            "tempo_s": [index * step for index in range(len(detections))],
            "series": series,
            "criterios": criteria,
            "desvio": not all(item["atingido"] for item in used),
            "estruturas": structures,
        }

    def _video_frames(self, path: Path, clip_at: float | None) -> Iterator[Image.Image]:
        import cv2

        capture = cv2.VideoCapture(str(path))
        try:
            fps = capture.get(cv2.CAP_PROP_FPS)
            if not capture.isOpened() or not fps or fps <= 0:
                raise ValueError("Não foi possível ler o vídeo enviado.")
            index = 0
            next_second = 0.0
            step = self.rule["intervalo_s"]
            while capture.grab():
                second = index / fps
                if clip_at is not None and second > clip_at:
                    break
                if second >= next_second:
                    ok, frame = capture.retrieve()
                    if ok:
                        yield Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    next_second += step
                index += 1
        finally:
            capture.release()

    def analyze_video(self, path: str | Path, clip_at: float | None = None) -> dict:
        """Analisa de 5 em 5 segundos até o primeiro clipe, ou até o fim se o momento não for informado."""
        result = self.analyze_frames(self._video_frames(Path(path), clip_at))
        result["fim_informado_s"] = clip_at
        return result

    def _dataset_labels(self) -> pd.DataFrame:
        with self._lock:
            if self._labels is None:
                with zipfile.ZipFile(self.dataset_zip) as archive:
                    labels = pd.read_csv(io.BytesIO(archive.read(f"{DATASET_ROOT}/cvs_labels.csv")))
                self._labels = labels[labels["split"] == "test"].sort_values(["video_id", "frame"])
        return self._labels

    def dataset_videos(self) -> list[int]:
        """Vídeos do conjunto de teste, que os modelos não viram no treino. A base fica fora do projeto."""
        if not self.dataset_zip or not self.dataset_zip.exists():
            return []
        return [int(video) for video in self._dataset_labels()["video_id"].unique()]

    def analyze_dataset_video(self, video_id: int) -> dict:
        labels = self._dataset_labels()
        frames = labels[labels["video_id"] == video_id]
        if frames.empty:
            raise KeyError(video_id)

        def images() -> Iterator[Image.Image]:
            with zipfile.ZipFile(self.dataset_zip) as archive:
                for name in frames["file_name"]:
                    with archive.open(f"{DATASET_ROOT}/images/test/{name}") as file:
                        yield Image.open(io.BytesIO(file.read())).convert("RGB")

        result = self.analyze_frames(images())
        reached = {name: bool((frames[name] >= LABEL_THRESHOLD).any()) for name in CRITERIA}
        result["especialistas"] = {**reached, "cvs_atingido": all(reached.values())}
        return result
