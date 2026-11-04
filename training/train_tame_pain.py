#!/usr/bin/env python3
"""Treina o indício de dor pela voz com o TAME Pain, na CPU e sem enviar dados para fora.

Exemplo:
  python training/train_tame_pain.py --dataset-dir "C:\\caminho\\tame-pain-1.0.0"

Saídas:
  backend/models/tame_pain_logreg.joblib      modelos para o backend
  training/resultados/tame_pain_metricas.json métricas para o relatório

Decisões de avaliação:
  - treino e teste nunca compartilham participantes;
  - declarações de dor ficam de fora, porque o áudio traz a nota falada;
  - a condição de frio não entra no modelo, porque é quase a própria resposta;
  - a gravação de base (mão esquerda, água morna) serve só para calibrar. Ela é da
    mesma sessão e, em 3 das 4 ordens do experimento, vem depois de um frio.

A base tem licença restrita. O cache de medidas identifica participantes, por isso
o script recusa gravá-lo dentro do projeto. Modelos e métricas não contêm áudio.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_DIR / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from services.voice_features import (  # noqa: E402
    FEATURE_NAMES,
    FEATURE_VERSION,
    SAMPLE_RATE,
    calibrate_features,
    extract_voice_features,
)

AUDIO_ARCHIVES = ("mic1_trim_v1", "mic1_trim_v1.zip")
PAIN_STATEMENT_ID = 99999
MAX_ACTION_LABEL = 2  # 0 é a melhor qualidade e 4 é a pior
PAIN_THRESHOLD = 4  # classificação binária do próprio dataset: 1 a 3 sem dor, 4 a 10 com dor
SEVERE_THRESHOLD = 7
BASELINE_CONDITION = "LW"
MIN_BASELINE_UTTERANCES = 5
DECISION_THRESHOLD = 0.5
N_FOLDS = 5
N_REPEATS = 5
N_BOOTSTRAP = 1000
SEED = 42


class DatasetError(Exception):
    """Problema nos dados ou nos argumentos que impede um treino confiável."""


def find_dataset_dir(path: Path) -> Path:
    """Aceita a pasta com meta_audio.csv ou a pasta logo acima dela."""
    if (path / "meta_audio.csv").is_file():
        return path
    candidates = [p for p in path.iterdir() if (p / "meta_audio.csv").is_file()] if path.is_dir() else []
    if len(candidates) != 1:
        raise DatasetError(f"Não encontrei meta_audio.csv em {path}")
    return candidates[0]


def find_archive(dataset_dir: Path) -> Path:
    for name in AUDIO_ARCHIVES:
        if (dataset_dir / name).is_file():
            return dataset_dir / name
    raise DatasetError(f"Não encontrei {AUDIO_ARCHIVES[0]} em {dataset_dir}")


def verify_archive(archive: Path, dataset_dir: Path) -> bool:
    """Confere o arquivo de áudios contra o SHA256SUMS.txt, quando ele existir."""
    sums_path = dataset_dir / "SHA256SUMS.txt"
    if not sums_path.is_file():
        return False
    expected = None
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        digest, _, name = line.partition(" ")
        if name.strip().lstrip("*") in AUDIO_ARCHIVES:
            expected = digest
    if expected is None:
        return False
    actual = hashlib.sha256()
    with archive.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            actual.update(block)
    if actual.hexdigest() != expected:
        raise DatasetError(f"Hash divergente em {archive}")
    return True


def real_wav_names(names: list[str]) -> list[str]:
    """Ignora as cópias geradas pelo macOS, que também terminam em .wav."""
    return [
        n for n in names
        if n.lower().endswith(".wav") and "__MACOSX" not in n and not Path(n).name.startswith("._")
    ]


def parse_wav_name(name: str) -> tuple[str, str, int, int]:
    pid, condition, utterance_number, utterance_id = Path(name).stem.split(".")
    return pid, condition, int(utterance_number), int(utterance_id)


def extract_chunk(archive: str, names: list[str], backend_dir: str) -> list[np.ndarray]:
    """Roda em outro processo; por isso refaz o caminho de importação."""
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)
    import soundfile
    from services.voice_features import extract_voice_features as extract

    features = []
    with zipfile.ZipFile(archive) as z:
        for name in names:
            audio, sample_rate = soundfile.read(io.BytesIO(z.read(name)), dtype="float32", always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            features.append(extract(audio, sample_rate))
    return features


def load_features(archive: Path, names: list[str], cache: Path | None, workers: int) -> np.ndarray:
    if cache and cache.is_file():
        with np.load(cache, allow_pickle=False) as saved:
            if int(saved["version"]) == FEATURE_VERSION and saved["names"].tolist() == names:
                print(f"Medidas lidas do cache: {cache}")
                return saved["features"]
        print("Cache desatualizado; recalculando.")

    print(f"Extraindo medidas de {len(names)} áudios com {workers} processos...")
    chunks = [names[i:i + 200] for i in range(0, len(names), 200)]
    results = joblib.Parallel(n_jobs=workers)(
        joblib.delayed(extract_chunk)(str(archive), chunk, str(BACKEND_DIR)) for chunk in chunks
    )
    features = np.array([row for chunk in results for row in chunk], dtype=np.float64)
    if cache:
        temporary = cache.with_name(cache.name + ".part.npz")
        np.savez(temporary, version=FEATURE_VERSION, names=np.array(names), features=features)
        os.replace(temporary, cache)
    return features


def load_samples(dataset_dir: Path, archive: Path, cache: Path | None, workers: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Devolve os áudios usados e as medidas, alinhados pela mesma ordem."""
    meta = pd.read_csv(dataset_dir / "meta_audio.csv")
    required = {"PID", "COND", "UTTNUM", "UTTID", "REVISED PAIN", "ACTION LABEL"}
    if not required <= set(meta.columns):
        raise DatasetError(f"Faltam colunas em meta_audio.csv: {sorted(required - set(meta.columns))}")

    with zipfile.ZipFile(archive) as z:
        names = real_wav_names(z.namelist())
    by_key = {parse_wav_name(n): n for n in names}
    meta["key"] = list(zip(meta["PID"], meta["COND"], meta["UTTNUM"], meta["UTTID"]))
    if set(meta["key"]) != set(by_key):
        raise DatasetError(
            f"Zip e metadados não batem: {len(set(by_key) - set(meta['key']))} só no zip, "
            f"{len(set(meta['key']) - set(by_key))} só no CSV."
        )

    used = meta[
        meta["REVISED PAIN"].notna()
        & (meta["UTTID"] != PAIN_STATEMENT_ID)
        & (meta["ACTION LABEL"] <= MAX_ACTION_LABEL)
    ].sort_values(["PID", "COND", "UTTNUM"]).reset_index(drop=True)

    features = load_features(archive, [by_key[k] for k in used["key"]], cache, workers)
    used["pain"] = used["REVISED PAIN"].astype(int)
    used["has_pain"] = (used["pain"] >= PAIN_THRESHOLD).astype(int)
    used["severe"] = (used["pain"] >= SEVERE_THRESHOLD).astype(int)
    used["cold"] = used["COND"].isin(["LC", "RC"]).astype(int)
    used["is_baseline"] = used["COND"] == BASELINE_CONDITION
    return used.drop(columns="key"), features


def calibrate_by_participant(samples: pd.DataFrame, features: np.ndarray) -> np.ndarray:
    """Calibra cada pessoa só com a gravação de base dela. Sem base suficiente, vira NaN."""
    calibrated = np.full_like(features, np.nan, dtype=np.float64)
    pids = samples["PID"].to_numpy()
    is_baseline = samples["is_baseline"].to_numpy()
    for pid in np.unique(pids):
        person = pids == pid
        baseline = person & is_baseline
        if baseline.sum() >= MIN_BASELINE_UTTERANCES:
            calibrated[person] = calibrate_features(
                features[person], features[baseline].mean(axis=0), features[baseline].std(axis=0)
            )
    return calibrated


def build_model() -> Pipeline:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED),
    )


def participant_bootstrap_ci(y: np.ndarray, score: np.ndarray, groups: np.ndarray) -> list[float]:
    """Intervalo de 95% do AUROC, sorteando pessoas e não áudios."""
    rng = np.random.default_rng(SEED)
    indices = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    values = []
    for _ in range(N_BOOTSTRAP):
        chosen = np.concatenate([indices[i] for i in rng.integers(0, len(indices), len(indices))])
        if len(np.unique(y[chosen])) == 2:
            values.append(roc_auc_score(y[chosen], score[chosen]))
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def evaluate(values: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict:
    """Validação cruzada repetida, sempre com participantes novos no teste."""
    fold_aucs = []
    mean_probabilities = np.zeros(len(y))
    for repeat in range(N_REPEATS):
        splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED + repeat)
        probabilities = np.zeros(len(y))
        for train, test in splitter.split(values, y, groups):
            if set(groups[train]) & set(groups[test]):
                raise RuntimeError("Participante presente no treino e no teste.")
            model = build_model().fit(values[train], y[train])
            probabilities[test] = model.predict_proba(values[test])[:, 1]
            fold_aucs.append(roc_auc_score(y[test], probabilities[test]))
        mean_probabilities += probabilities / N_REPEATS

    predicted = (mean_probabilities >= DECISION_THRESHOLD).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y, predicted, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "audios": int(len(y)),
        "participantes": int(len(np.unique(groups))),
        "positivos": float(y.mean()),
        "AUROC_media_folds": float(np.mean(fold_aucs)),
        "AUROC_dp_folds": float(np.std(fold_aucs)),
        "AUROC_IC95_por_participante": participant_bootstrap_ci(y, mean_probabilities, groups),
        "precisao": float(precision),
        "recall": float(recall),
        "F1": float(f1),
        "acuracia_balanceada": float(balanced_accuracy_score(y, predicted)),
        "matriz": {"VP": int(tp), "FP": int(fp), "FN": int(fn), "VN": int(tn)},
    }


def run_scenarios(samples: pd.DataFrame, features: np.ndarray, calibrated: np.ndarray) -> dict:
    # Mesmos áudios nos cenários comparados: tudo menos a gravação de base.
    valid = ~np.isnan(calibrated).any(axis=1)
    target_rows = valid & ~samples["is_baseline"].to_numpy()
    rows = samples[target_rows].reset_index(drop=True)
    calibrated_rows, raw_rows = calibrated[target_rows], features[target_rows]
    y, groups = rows["has_pain"].to_numpy(), rows["PID"].to_numpy()

    shuffled = np.random.default_rng(SEED).permutation(y)
    cold_mask = rows["cold"].to_numpy() == 1

    plan = {
        "principal_dor_calibrado": ("Dor >= 4, calibrado", calibrated_rows, y, groups),
        "comparacao_dor_sem_calibracao": ("Dor >= 4, sem calibrar", raw_rows, y, groups),
        "limite_frio_x_morno_calibrado": ("Frio x morno, calibrado", calibrated_rows, rows["cold"].to_numpy(), groups),
        "limite_severa_dentro_do_frio": (
            "Dor >= 7 dentro do frio, calibrado",
            calibrated_rows[cold_mask], rows["severe"].to_numpy()[cold_mask], groups[cold_mask],
        ),
        "controle_rotulos_embaralhados": ("Rótulos embaralhados (esperado: 0,50)", calibrated_rows, shuffled, groups),
    }
    results = {}
    for key, (title, values, target, target_groups) in plan.items():
        results[key] = evaluate(values, target, target_groups)
        result = results[key]
        low, high = result["AUROC_IC95_por_participante"]
        print(f"  {title}: AUROC={result['AUROC_media_folds']:.3f} (IC95% {low:.3f} a {high:.3f}) F1={result['F1']:.3f}")
    return results


def check_output_path(path: Path) -> None:
    if path.resolve().is_relative_to(PROJECT_DIR):
        raise DatasetError(f"Grave o cache fora do projeto ({PROJECT_DIR}): ele identifica participantes.")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina o indício de dor pela voz com o TAME Pain.")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Pasta do TAME Pain.")
    parser.add_argument("--model-out", type=Path, default=BACKEND_DIR / "models" / "tame_pain_logreg.joblib")
    parser.add_argument("--metrics-out", type=Path, default=PROJECT_DIR / "training" / "resultados" / "tame_pain_metricas.json")
    parser.add_argument("--cache", type=Path, help="Cache de medidas. Padrão: ao lado da base.")
    parser.add_argument("--no-cache", action="store_true", help="Não lê nem grava o cache de medidas.")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> None:
    dataset_dir = find_dataset_dir(args.dataset_dir.resolve())
    archive = find_archive(dataset_dir)

    cache = None
    if not args.no_cache:
        cache = (args.cache or dataset_dir.parent / "tame_pain_medidas.npz").resolve()
        check_output_path(cache)

    print(f"Base: {dataset_dir}")
    checksum_ok = verify_archive(archive, dataset_dir)
    print(f"Hash do arquivo de áudios conferido: {'sim' if checksum_ok else 'não'}")

    samples, features = load_samples(dataset_dir, archive, cache, args.workers)
    calibrated = calibrate_by_participant(samples, features)
    print(f"Áudios usados: {len(samples)} de {samples['PID'].nunique()} participantes")

    print("\nAvaliação com participantes novos")
    scenarios = run_scenarios(samples, features, calibrated)

    # Os modelos finais usam todos os participantes; a estimativa de desempenho vem da validação acima.
    train_rows = ~np.isnan(calibrated).any(axis=1) & ~samples["is_baseline"].to_numpy()
    y_calibrated = samples.loc[train_rows, "has_pain"].to_numpy()
    artifact = {
        "modelo_calibrado": build_model().fit(calibrated[train_rows], y_calibrated),
        "modelo_sem_calibracao": build_model().fit(features, samples["has_pain"].to_numpy()),
        "feature_names": FEATURE_NAMES,
        "feature_version": FEATURE_VERSION,
        "sample_rate": SAMPLE_RATE,
        "limiar_decisao": DECISION_THRESHOLD,
        "classe_positiva": f"dor autorrelatada >= {PAIN_THRESHOLD} em 10",
        "versoes": {"sklearn": sklearn.__version__, "numpy": np.__version__},
    }
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    temporary_model = args.model_out.with_name(args.model_out.name + ".part")
    joblib.dump(artifact, temporary_model)
    os.replace(temporary_model, args.model_out)

    metrics = {
        "gerado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": "TAME Pain 1.0.0",
        "hash_do_audio_conferido": checksum_ok,
        "amostra": {
            "audios_usados": int(len(samples)),
            "participantes": int(samples["PID"].nunique()),
            "filtro": f"frases lidas, sem declarações de dor, ACTION LABEL <= {MAX_ACTION_LABEL}",
            "gravacao_de_base": f"condição {BASELINE_CONDITION}, usada só para calibrar",
        },
        "validacao": {"tipo": "StratifiedGroupKFold por participante", "folds": N_FOLDS, "repeticoes": N_REPEATS, "bootstrap": N_BOOTSTRAP},
        "modelo": "StandardScaler + LogisticRegression(class_weight='balanced')",
        "cenarios": scenarios,
        "limitacoes": [
            "Dor induzida em laboratório por água fria, em pessoas saudáveis de 18 a 33 anos; não são pacientes.",
            "Distúrbios neurológicos foram critério de exclusão, então a base não cobre disartria; fadiga também não tem rótulo.",
            "A dor quase coincide com a condição de frio; compare o cenário principal com limite_frio_x_morno_calibrado.",
            "O modelo não separa bem dor severa de dor menor; não deve exibir nota de dor.",
            "Cada frase herda a nota da declaração de dor seguinte; o rótulo não é medido frase a frase.",
            "A gravação de base é da mesma sessão e do mesmo microfone, e em 3 das 4 ordens do experimento vem depois de um frio; o AUROC calibrado tende a ser otimista para uso real.",
            "Falantes fluentes em inglês, em sala controlada; o modelo não foi validado em português nem em outros microfones.",
            "Indício de apoio à decisão, não diagnóstico.",
        ],
    }
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\nModelo:   {args.model_out}")
    print(f"Métricas: {args.metrics_out}")


def main(argv: list[str] | None = None) -> int:
    try:
        run(parse_args(argv))
    except DatasetError as error:
        print(f"ERRO: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
