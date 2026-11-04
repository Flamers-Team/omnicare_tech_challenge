"""Avalia o sinal vocal por gravação, como o sistema usa, e confere o corte em frases.

Cada gravação é o conjunto de frases de uma pessoa em uma condição. A nota da gravação é a média das notas
das frases, sempre com pessoas fora do treino. O resultado salvo tem só números agregados.

Uso:
    python avaliar_voz_gravacao.py <pasta do TAME Pain>
"""
from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import joblib
import numpy as np
import soundfile
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

TRAINING_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TRAINING_DIR))
import train_tame_pain as training  # noqa: E402

from services.voice_features import SAMPLE_RATE, extract_voice_features, split_utterances  # noqa: E402

OUTPUT = TRAINING_DIR / "resultados" / "voz_gravacao_metricas.json"
MODEL = training.BACKEND_DIR / "models" / "tame_pain_logreg.joblib"
PARITY_PEOPLE = 12
SILENCE_SECONDS = 1.0


def summarize(y: np.ndarray, score: np.ndarray, groups: np.ndarray) -> dict:
    predicted = score >= training.DECISION_THRESHOLD
    return {
        "gravacoes": int(len(y)),
        "com_dor": int(y.sum()),
        "AUROC": float(roc_auc_score(y, score)),
        "AUROC_IC95_por_pessoa": training.participant_bootstrap_ci(y, score, groups),
        "sinal_presente_com_dor": {"acertos": int((predicted & (y == 1)).sum()), "total": int(y.sum())},
        "sinal_ausente_sem_dor": {"acertos": int((~predicted & (y == 0)).sum()), "total": int((y == 0).sum())},
    }


def recording_level(values: np.ndarray, samples) -> dict:
    """Validação cruzada por pessoa, com a nota média por gravação."""
    groups = samples["PID"].to_numpy()
    y_frames = samples["has_pain"].to_numpy()
    probabilities = np.zeros(len(samples))
    for repeat in range(training.N_REPEATS):
        splitter = StratifiedGroupKFold(n_splits=training.N_FOLDS, shuffle=True, random_state=training.SEED + repeat)
        for fit, held_out in splitter.split(values, y_frames, groups):
            model = training.build_model().fit(values[fit], y_frames[fit])
            probabilities[held_out] += model.predict_proba(values[held_out])[:, 1] / training.N_REPEATS
    table = samples.assign(score=probabilities).groupby(["PID", "COND"]).agg(score=("score", "mean"), pain=("pain", "mean"), cold=("cold", "first"), frases=("score", "size")).reset_index()
    table = table[table["frases"] >= training.MIN_BASELINE_UTTERANCES]
    pids = table["PID"].to_numpy()
    return {
        "dor_media_4_ou_mais": summarize((table["pain"] >= training.PAIN_THRESHOLD).astype(int).to_numpy(), table["score"].to_numpy(), pids),
        "frio_contra_morno": summarize(table["cold"].to_numpy(), table["score"].to_numpy(), pids),
    }


def split_check(archive: Path, names: list[str], samples) -> dict:
    """Junta as frases de cada pessoa com pausas e confere se o corte encontra as mesmas frases."""
    artifact = joblib.load(MODEL)
    rng = np.random.default_rng(training.SEED)
    people = rng.choice(samples["PID"].unique(), PARITY_PEOPLE, replace=False)
    counts, differences = [], []
    silence = np.zeros(int(SILENCE_SECONDS * SAMPLE_RATE), dtype=np.float32)
    with zipfile.ZipFile(archive) as z:
        for person in people:
            rows = samples[(samples["PID"] == person) & (samples["COND"] == "RC")]
            clips = []
            for key in rows[["PID", "COND", "UTTNUM", "UTTID"]].itertuples(index=False):
                audio, rate = soundfile.read(io.BytesIO(z.read(names[tuple(key)])), dtype="float32", always_2d=False)
                assert rate == SAMPLE_RATE
                clips.append(audio.mean(axis=1) if audio.ndim > 1 else audio)
            if not clips:
                continue
            joined = np.concatenate([part for clip in clips for part in (clip, silence)])
            found = split_utterances(joined)
            direct = artifact["modelo_sem_calibracao"].predict_proba([extract_voice_features(clip) for clip in clips])[:, 1].mean()
            cut = artifact["modelo_sem_calibracao"].predict_proba([extract_voice_features(clip) for clip in found])[:, 1].mean()
            counts.append((len(clips), len(found)))
            differences.append(abs(float(direct) - float(cut)))
    pairs = np.array(counts)
    return {
        "gravacoes_montadas": len(pairs),
        "frases_reais_mediana": float(np.median(pairs[:, 0])),
        "trechos_encontrados_mediana": float(np.median(pairs[:, 1])),
        "razao_trechos_por_frase_mediana": float(np.median(pairs[:, 1] / pairs[:, 0])),
        "diferenca_absoluta_na_nota_mediana": float(np.median(differences)),
        "diferenca_absoluta_na_nota_maior": float(np.max(differences)),
    }


def main() -> None:
    dataset = training.find_dataset_dir(Path(sys.argv[1]).resolve())
    archive = training.find_archive(dataset)
    samples, features = training.load_samples(dataset, archive, dataset.parent / "tame_pain_medidas.npz", 1)
    calibrated = training.calibrate_by_participant(samples, features)
    usable = ~np.isnan(calibrated).any(axis=1) & ~samples["is_baseline"].to_numpy()
    rows = samples[usable].reset_index(drop=True)

    with zipfile.ZipFile(archive) as z:
        names = {training.parse_wav_name(name): name for name in training.real_wav_names(z.namelist())}

    report = {
        "como_foi_feito": "Média da nota das frases por pessoa e condição, sem a gravação de base, em validação cruzada por pessoa (5 divisões, 5 repetições).",
        "com_gravacao_em_repouso": recording_level(calibrated[usable], rows),
        "sem_gravacao_em_repouso": recording_level(features[usable], rows),
        "conferencia_do_corte_em_frases": split_check(archive, names, samples),
        "aviso": "A dor quase coincide com o frio nesta base, e as gravações em repouso são da mesma sessão. Os números não valem para consultas reais nem para português.",
    }
    OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
