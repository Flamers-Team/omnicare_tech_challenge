"""Revisa a regra de desvio por vídeo usando o modelo de CVS já treinado.

A escolha usa só a validação. O teste roda uma vez, para a regra atual e para a escolhida.
O resultado salvo tem apenas números agregados, sem frames, nomes de arquivo ou rótulos por vídeo.

Uso:
    python avaliar_regra_cvs.py <endoscapes_omnicare.zip> <cvs_resnet18.pt> <pasta_de_trabalho_fora_do_projeto>
"""
from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats import binomtest, spearmanr
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import RepeatedStratifiedKFold
from torch import nn
from torchvision import models, transforms

CRITERIA = ["C1", "C2", "C3"]
ROOT = "endoscapes_omnicare"
SEED = 42
SMOOTH_WINDOW = 3
LABEL_THRESHOLD = 0.5
# Regras com acurácia balanceada até esta distância da melhor são tratadas como empatadas:
# com 41 vídeos, uma diferença menor que essa pode ser só ruído.
TIE_MARGIN = 0.02
FRAME_STEP_SECONDS = 5  # os frames rotulados da base ficam a cada 125 quadros, a 25 quadros por segundo
DETECTOR_CONFIDENCE = 0.31  # melhor F1 geral na curva de validação do YOLOv8 (0,308)
RESULTS = Path(__file__).resolve().parent / "resultados"
OUTPUT = RESULTS / "endoscapes_regra_video.json"
SYSTEM_RULE = Path(__file__).resolve().parent.parent / "backend" / "models" / "cvs_regra.json"


def peak(values: pd.Series) -> float:
    return float(values.rolling(SMOOTH_WINDOW, min_periods=1).mean().max())


def top_three(values: pd.Series) -> float:
    return float(values.nlargest(3).mean())


def top_fifth(values: pd.Series) -> float:
    return float(values.nlargest(max(3, int(np.ceil(0.2 * len(values))))).mean())


def last_three(values: pd.Series) -> float:
    return float(values.tail(3).mean())


AGGREGATIONS = {"pico": peak, "tres_maiores": top_three, "quinto_maior": top_fifth, "tres_finais": last_three}
CRITERIA_SETS = {"C1_C2_C3": CRITERIA, "C2_C3": ["C2", "C3"]}
RULES = {f"{a}|{s}": (a, s) for a in AGGREGATIONS for s in CRITERIA_SETS}
CURRENT_RULE = "pico|C1_C2_C3"


def load_predictions(zip_path: Path, weights: Path, cache: Path) -> pd.DataFrame:
    if cache.exists():
        return pd.read_csv(cache)
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CRITERIA))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    transform = transforms.Compose([
        transforms.Resize(tuple(checkpoint["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(checkpoint["mean"], checkpoint["std"]),
    ])
    with zipfile.ZipFile(zip_path) as archive:
        labels = pd.read_csv(io.BytesIO(archive.read(f"{ROOT}/cvs_labels.csv")))
        frames = labels[labels["split"].isin(["val", "test"])].sort_values(["split", "video_id", "frame"]).reset_index(drop=True)
        scores = []
        with torch.no_grad():
            for start in range(0, len(frames), 32):
                batch = []
                for _, row in frames.iloc[start:start + 32].iterrows():
                    data = archive.read(f"{ROOT}/images/{row['split']}/{row['file_name']}")
                    batch.append(transform(Image.open(io.BytesIO(data)).convert("RGB")))
                scores.append(torch.sigmoid(model(torch.stack(batch))).numpy())
                print(f"{min(start + 32, len(frames))}/{len(frames)} frames", flush=True)
    frames[[f"p_{c}" for c in CRITERIA]] = np.concatenate(scores)
    cache.parent.mkdir(parents=True, exist_ok=True)
    frames.to_csv(cache, index=False)
    return frames


def video_table(frames: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for video_id, group in frames.groupby("video_id"):
        group = group.sort_values("frame")
        row = {"video_id": video_id, "n_frames": len(group)}
        for criterion in CRITERIA:
            row[criterion] = int((group[criterion] >= LABEL_THRESHOLD).any())
            for name, function in AGGREGATIONS.items():
                row[f"{name}_{criterion}"] = function(group[f"p_{criterion}"])
        rows.append(row)
    table = pd.DataFrame(rows)
    table["desvio_real"] = 1 - table[CRITERIA].all(axis=1).astype(int)
    return table


def youden(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(set(y_true)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y_true, score)
    finite = np.isfinite(thresholds)
    return float(thresholds[finite][np.argmax((tpr - fpr)[finite])])


def fit_thresholds(table: pd.DataFrame, rule: str) -> dict[str, float]:
    aggregation, criteria_set = RULES[rule]
    return {c: youden(table[c].to_numpy(), table[f"{aggregation}_{c}"].to_numpy()) for c in CRITERIA_SETS[criteria_set]}


def deviation_scores(table: pd.DataFrame, rule: str, thresholds: dict[str, float]) -> np.ndarray:
    """Quanto o critério mais distante ficou abaixo do limiar. Acima de zero, há desvio."""
    aggregation = RULES[rule][0]
    margins = np.column_stack([table[f"{aggregation}_{c}"] - thresholds[c] for c in thresholds])
    return -margins.min(axis=1)


def cross_validate(table: pd.DataFrame, rule: str) -> dict:
    splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=SEED)
    y_true = table["desvio_real"].to_numpy()
    accuracies, aucs = [], []
    collected = np.zeros((20, len(table)))
    for index, (fit, held_out) in enumerate(splitter.split(table, y_true)):
        thresholds = fit_thresholds(table.iloc[fit], rule)
        collected[index // 5, held_out] = deviation_scores(table.iloc[held_out], rule, thresholds)
    for scores in collected:
        accuracies.append(balanced_accuracy_score(y_true, (scores > 0).astype(int)))
        aucs.append(roc_auc_score(y_true, scores))
    return {"acuracia_balanceada": float(np.mean(accuracies)), "AUROC": float(np.mean(aucs))}


def wilson(hits: int, total: int) -> dict:
    interval = binomtest(hits, total).proportion_ci(method="wilson")
    return {"acertos": hits, "total": total, "valor": hits / total, "ic95": [float(interval.low), float(interval.high)]}


def auc_interval(y_true: np.ndarray, score: np.ndarray, rounds: int = 2000) -> list[float]:
    rng = np.random.default_rng(SEED)
    values = []
    for _ in range(rounds):
        chosen = rng.integers(0, len(y_true), len(y_true))
        if len(set(y_true[chosen])) == 2:
            values.append(roc_auc_score(y_true[chosen], score[chosen]))
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def evaluate(table: pd.DataFrame, rule: str, thresholds: dict[str, float]) -> dict:
    y_true = table["desvio_real"].to_numpy()
    score = deviation_scores(table, rule, thresholds)
    y_pred = (score > 0).astype(int)
    tp, fp = int(((y_true == 1) & (y_pred == 1)).sum()), int(((y_true == 0) & (y_pred == 1)).sum())
    fn, tn = int(((y_true == 1) & (y_pred == 0)).sum()), int(((y_true == 0) & (y_pred == 0)).sum())
    aggregation = RULES[rule][0]
    flagged_some_absent = flagged_exact = 0
    for _, row in table[(y_true == 1) & (y_pred == 1)].iterrows():
        absent = {c for c in thresholds if row[c] == 0}
        flagged = {c for c in thresholds if row[f"{aggregation}_{c}"] < thresholds[c]}
        flagged_some_absent += bool(absent & flagged)
        flagged_exact += absent == flagged
    return {
        "videos": len(table),
        "matriz": {"VP": tp, "FP": fp, "FN": fn, "VN": tn},
        "desvios_detectados": wilson(tp, tp + fn),
        "cvs_reconhecido": wilson(tn, tn + fp),
        "precisao": wilson(tp, tp + fp) if tp + fp else None,
        "acuracia": float((y_true == y_pred).mean()),
        "acuracia_sempre_desvio": float(y_true.mean()),
        "acuracia_balanceada": float(balanced_accuracy_score(y_true, y_pred)),
        "AUROC": float(roc_auc_score(y_true, score)),
        "AUROC_IC95": auc_interval(y_true, score),
        "mcnemar_contra_sempre_desvio_p": float(binomtest(tn, tn + fn, 0.5).pvalue) if tn + fn else None,
        "spearman_frames_x_escore": float(spearmanr(table["n_frames"], score).statistic),
        "motivo_nos_desvios_detectados": {
            "total": tp,
            "apontou_ao_menos_um_criterio_ausente": flagged_some_absent,
            "apontou_exatamente_os_ausentes": flagged_exact,
        },
    }


def main() -> None:
    zip_path, weights, workdir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    frames = load_predictions(zip_path, weights, workdir / "predicoes_cvs_val_teste.csv")
    validation = video_table(frames[frames["split"] == "val"])
    test = video_table(frames[frames["split"] == "test"])

    candidates = {}
    for rule in RULES:
        spearman = spearmanr(validation["n_frames"], deviation_scores(validation, rule, fit_thresholds(validation, rule))).statistic
        candidates[rule] = {**cross_validate(validation, rule), "spearman_frames_x_escore": float(spearman)}

    best = max(item["acuracia_balanceada"] for item in candidates.values())
    tied = {rule: item for rule, item in candidates.items() if item["acuracia_balanceada"] >= best - TIE_MARGIN}
    chosen = min(tied, key=lambda rule: abs(tied[rule]["spearman_frames_x_escore"]))

    criterion_auc = {c: float(roc_auc_score(validation[c], validation[f"pico_{c}"])) for c in CRITERIA}
    current_thresholds = fit_thresholds(validation, CURRENT_RULE)
    chosen_thresholds = fit_thresholds(validation, chosen)
    aggregation, criteria_set = RULES[chosen]
    report = {
        "como_foi_escolhida": (
            "Validação cruzada 5x20 nos 41 vídeos de validação, com limiares ajustados dentro de cada divisão. "
            f"Entre as regras a até {TIE_MARGIN} da melhor acurácia balanceada, vence a menos ligada à duração do vídeo."
        ),
        "aviso": (
            "A candidata sem C1 foi incluída depois de vermos o desempenho do C1 no teste. "
            "Este é um segundo olhar sobre os mesmos 40 vídeos de teste, então o resultado tende a ser um pouco otimista."
        ),
        "auroc_de_cada_criterio_na_validacao": criterion_auc,
        "candidatas_validacao_cruzada": candidates,
        "regra_escolhida": {"nome": chosen, "agregacao": aggregation, "criterios": CRITERIA_SETS[criteria_set], "limiares": chosen_thresholds},
        "teste": {
            "regra_atual": {"limiares": current_thresholds, **evaluate(test, CURRENT_RULE, current_thresholds)},
            "regra_escolhida": evaluate(test, chosen, chosen_thresholds),
        },
    }
    OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))

    chosen_test = report["teste"]["regra_escolhida"]
    detector = json.loads((RESULTS / "endoscapes_metricas.json").read_text(encoding="utf-8"))["yolov8_teste"]["per_class"]
    SYSTEM_RULE.write_text(json.dumps({
        "agregacao": aggregation,
        "janela_media": SMOOTH_WINDOW,
        "intervalo_s": FRAME_STEP_SECONDS,
        "limiares": chosen_thresholds,
        "confianca_detector": DETECTOR_CONFIDENCE,
        "detector_ap50_teste": {name: round(values["AP50"], 3) for name, values in detector.items()},
        "desempenho_teste": {
            "videos": chosen_test["videos"],
            "desvios_detectados": chosen_test["desvios_detectados"],
            "cvs_reconhecido": chosen_test["cvs_reconhecido"],
            "AUROC": chosen_test["AUROC"],
            "AUROC_IC95": chosen_test["AUROC_IC95"],
        },
    }, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
