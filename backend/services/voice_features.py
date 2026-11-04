"""Medidas de voz usadas no treino e na inferência do indício de dor."""
from __future__ import annotations

import librosa
import numpy as np

SAMPLE_RATE = 16000
N_MFCC = 13
MIN_SAMPLES = 2048
FEATURE_VERSION = 1
SILENCE_TOP_DB = 30
MIN_PEAK = 0.005
MERGE_GAP_SECONDS = 0.3
MIN_UTTERANCE_SECONDS = 0.5
MAX_UTTERANCE_SECONDS = 4.0  # 99% das frases usadas no treino têm menos que isso, com mediana de 2,5 s
CHUNK_SECONDS = 2.5

# A duração fica de fora de propósito: nos dados de treino ela depende da frase
# lida e não seria comparável em uma consulta longa.
FEATURE_NAMES = (
    [f"mfcc{i}_media" for i in range(N_MFCC)]
    + [f"mfcc{i}_dp" for i in range(N_MFCC)]
    + [f"delta{i}_media" for i in range(N_MFCC)]
    + [f"delta{i}_dp" for i in range(N_MFCC)]
    + ["rms_media", "rms_dp", "zcr_media", "zcr_dp", "centroide_media", "centroide_dp", "f0_mediana", "f0_dp"]
)


def load_audio(path: str) -> np.ndarray:
    """Carrega o áudio em mono, na taxa usada no treino."""
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return audio


def extract_voice_features(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Resume um áudio em um vetor com as medidas de FEATURE_NAMES."""
    if sample_rate != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
    if len(audio) < MIN_SAMPLES:
        raise ValueError(f"Áudio curto demais: {len(audio)} amostras, mínimo de {MIN_SAMPLES}.")

    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    frames = mfcc.shape[1]
    width = min(9, frames if frames % 2 else frames - 1)
    delta = librosa.feature.delta(mfcc, width=width) if width >= 3 else np.zeros_like(mfcc)
    rms = librosa.feature.rms(y=audio)[0]
    zcr = librosa.feature.zero_crossing_rate(audio)[0]
    centroid = librosa.feature.spectral_centroid(y=audio, sr=SAMPLE_RATE)[0]
    f0 = librosa.yin(audio, fmin=65, fmax=400, sr=SAMPLE_RATE)

    features = np.concatenate([
        mfcc.mean(axis=1),
        mfcc.std(axis=1),
        delta.mean(axis=1),
        delta.std(axis=1),
        [rms.mean(), rms.std(), zcr.mean(), zcr.std(), centroid.mean(), centroid.std(), np.median(f0), f0.std()],
    ])
    return np.nan_to_num(features)


def split_utterances(audio: np.ndarray) -> list[np.ndarray]:
    """Corta a gravação em trechos parecidos com as frases do treino, de cerca de 2,5 segundos."""
    # O corte de silêncio é relativo ao trecho mais alto, então um áudio mudo pareceria todo fala.
    if len(audio) == 0 or np.abs(audio).max() < MIN_PEAK:
        return []
    pieces: list[list[int]] = []
    for start, end in librosa.effects.split(audio, top_db=SILENCE_TOP_DB):
        if pieces and start - pieces[-1][1] < MERGE_GAP_SECONDS * SAMPLE_RATE:
            pieces[-1][1] = int(end)
        else:
            pieces.append([int(start), int(end)])

    utterances = []
    chunk = int(CHUNK_SECONDS * SAMPLE_RATE)
    for start, end in pieces:
        if end - start <= MAX_UTTERANCE_SECONDS * SAMPLE_RATE:
            utterances.append(audio[start:end])
        else:
            utterances.extend(audio[position:position + chunk] for position in range(start, end, chunk))
    shortest = max(MIN_SAMPLES, int(MIN_UTTERANCE_SECONDS * SAMPLE_RATE))
    return [item for item in utterances if len(item) >= shortest]


def calibrate_features(values: np.ndarray, baseline_mean: np.ndarray, baseline_std: np.ndarray) -> np.ndarray:
    """Expressa as medidas em relação à gravação de base da própria pessoa."""
    scaled = (values - baseline_mean) / np.maximum(baseline_std, 1e-6)
    return np.clip(scaled, -10, 10)
