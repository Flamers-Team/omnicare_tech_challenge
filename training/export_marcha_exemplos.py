#!/usr/bin/env python3
"""Exporta duas caminhadas do Multi-Gait-Posture para a demonstração.

Uma é a caminhada original. A outra é a mesma caminhada com o tronco 20 graus à frente,
uma alteração simulada. As duas pessoas fazem parte da referência, então este par mostra
o funcionamento da tela e não mede desempenho.

Exemplo:
  python training/export_marcha_exemplos.py --zip "C:\\caminho\\multi-gait-posture-1.0.0.zip"
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_DIR / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from services.rehab_gait_service import JOINT, JOINTS, assess_walk  # noqa: E402

SPEED = "0.5"
TRUNK_DEGREES = 20
UPPER_BODY = [JOINT[name] for name in JOINTS[1:15]]


def lean_trunk_forward(X: np.ndarray, degrees: float) -> np.ndarray:
    out = X.copy()
    pelvis = X[:, [JOINT["pelvis"]]]
    relative = X[:, UPPER_BODY] - pelvis
    angle = np.radians(degrees)
    x, z = relative[..., 0].copy(), relative[..., 2].copy()
    relative[..., 0], relative[..., 2] = np.cos(angle) * x + np.sin(angle) * z, -np.sin(angle) * x + np.cos(angle) * z
    out[:, UPPER_BODY] = relative + pelvis
    return out


def find_walk(zip_path: Path, reference: dict) -> tuple[pd.DataFrame, np.ndarray, str]:
    with zipfile.ZipFile(zip_path) as outer:
        names = sorted(n for n in outer.namelist() if "/processed_data/participant" in n and n.endswith(".zip"))
        for name in names:
            with zipfile.ZipFile(io.BytesIO(outer.read(name))) as inner:
                for item in sorted(inner.namelist()):
                    if item.split("/")[-1] != "normalized_skeleton_3d.csv":
                        continue
                    _, sequence, speed, _ = item.split("/")[-2].split("_")
                    if sequence != "straight" or speed != SPEED:
                        continue
                    frame = pd.read_csv(io.BytesIO(inner.read(item)), index_col=0)
                    X = frame.to_numpy(dtype=np.float64).reshape(-1, len(JOINTS), 3)
                    normal = assess_walk(X, reference)
                    altered = assess_walk(lean_trunk_forward(X, TRUNK_DEGREES), reference)
                    if normal["avaliada"] and not normal["alerta"] and altered["alerta"]:
                        print("Caminhada:", item, "| normal:", normal["pontuacao"], "| alterada:", altered["pontuacao"])
                        return frame, X, item
    raise RuntimeError("Nenhuma caminhada atendeu aos critérios.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Exporta as caminhadas de exemplo da demonstração.")
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=BACKEND_DIR / "sample_data" / "marcha")
    args = parser.parse_args()

    reference = json.loads((BACKEND_DIR / "models" / "marcha_referencia.json").read_text(encoding="utf-8"))
    frame, X, source = find_walk(args.zip, reference)
    args.out.mkdir(parents=True, exist_ok=True)
    frame.round(5).to_csv(args.out / "caminhada_normal.csv.gz")
    altered = pd.DataFrame(lean_trunk_forward(X, TRUNK_DEGREES).reshape(len(X), -1), columns=frame.columns, index=frame.index)
    altered.round(5).to_csv(args.out / "caminhada_tronco_20_graus.csv.gz")
    (args.out / "LEIA-ME.txt").write_text(
        "Caminhadas derivadas de: Palermo et al. (2021), A multi-camera and multimodal dataset for posture "
        "and gait analysis (version 1.0.0), PhysioNet, https://doi.org/10.13026/fyxw-n385, CC BY 4.0.\n"
        f"Trecho de origem: {source}\n"
        f"caminhada_tronco_20_graus.csv.gz é a mesma caminhada com o tronco girado {TRUNK_DEGREES} graus à frente, uma alteração simulada.\n",
        encoding="utf-8",
    )
    print("Salvo em:", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
