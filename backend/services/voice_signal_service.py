"""Sinal vocal com o modelo treinado no TAME Pain. É um apoio e não mede dor."""
from __future__ import annotations

import threading
from pathlib import Path

import joblib
import numpy as np

from services.voice_features import FEATURE_VERSION, calibrate_features, extract_voice_features, split_utterances

MIN_BASELINE_UTTERANCES = 5  # mesmo mínimo usado no treino
LIMITS = "Treinado em inglês, com dor induzida por água fria em adultos saudáveis. Não foi validado em português nem com outros microfones, e não é uma nota de dor."
NO_BASELINE_WARNING = "Sem a gravação em repouso, o resultado é mais fraco e serve só de informação: não soma pontos no risco."


class VoiceInputError(ValueError):
    """Problema na gravação que a pessoa consegue resolver. A mensagem vai para a tela."""


class VoiceSignalService:
    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        self._lock = threading.Lock()
        self._artifact = None

    def available(self) -> bool:
        return self.model_path.exists()

    def _load(self) -> dict:
        with self._lock:
            if self._artifact is None:
                artifact = joblib.load(self.model_path)
                if artifact["feature_version"] != FEATURE_VERSION:
                    raise RuntimeError("O modelo de voz foi treinado com outra versão das medidas.")
                self._artifact = artifact
        return self._artifact

    @staticmethod
    def _measures(audio: np.ndarray) -> np.ndarray:
        utterances = split_utterances(audio)
        return np.array([extract_voice_features(item) for item in utterances]) if utterances else np.empty((0, 0))

    def assess(self, audio: np.ndarray, baseline: np.ndarray | None = None) -> dict:
        artifact = self._load()
        measures = self._measures(audio)
        if not len(measures):
            return {"avaliado": False, "motivo": "Não encontrei trechos de fala no áudio."}

        if baseline is None:
            scores = artifact["modelo_sem_calibracao"].predict_proba(measures)[:, 1]
            base_count = 0
        else:
            base = self._measures(baseline)
            base_count = len(base)
            if base_count < MIN_BASELINE_UTTERANCES:
                raise VoiceInputError(
                    f"A gravação em repouso tem {base_count} frases reconhecidas e precisa de ao menos {MIN_BASELINE_UTTERANCES}. "
                    "Grave de novo, com uma pausa de 1 segundo entre as frases."
                )
            calibrated = calibrate_features(measures, base.mean(axis=0), base.std(axis=0))
            scores = artifact["modelo_calibrado"].predict_proba(calibrated)[:, 1]

        score = float(scores.mean())
        limit = float(artifact["limiar_decisao"])
        present = score >= limit
        has_baseline = baseline is not None
        return {
            "avaliado": True,
            "com_gravacao_em_repouso": has_baseline,
            "frases": int(len(scores)),
            "frases_em_repouso": base_count,
            "pontuacao": round(score, 2),
            "limiar": limit,
            "presente": present,
            "soma_no_risco": present and has_baseline,
            "descricao": f"Sinal vocal {'presente' if present else 'ausente'} em relação à {'gravação em repouso' if has_baseline else 'referência geral'}",
            "aviso": LIMITS if has_baseline else f"{NO_BASELINE_WARNING} {LIMITS}",
        }
