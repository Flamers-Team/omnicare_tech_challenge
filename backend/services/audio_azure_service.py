import os
import re
import tempfile
import threading
import unicodedata
from pathlib import Path

import azure.cognitiveservices.speech as speechsdk
import librosa
import numpy as np
import soundfile as sf
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

from services.voice_features import load_audio

load_dotenv()

NEGATIVE_SPEECH_SCORE = 0.6
TEXT_CHUNK_CHARS = 4000  # o Azure Language aceita até 5.120 caracteres por documento
# Graves geram alerta. Leves só aparecem na tela, para não alertar por qualquer queixa.
SEVERE_TERMS = (
    "falta de ar", "dificuldade para respirar", "dificuldade de respirar", "não consigo respirar",
    "dor no peito", "aperto no peito", "tontura", "desmaio", "desmaiei", "palpitação",
    "confusão", "suor frio", "dor muito forte", "dor forte", "sangramento",
)
MILD_TERMS = (
    "cansaço", "cansado", "cansada", "fadiga", "tosse", "febre", "dor de cabeça", "dor de barriga",
    "dor de garganta", "dor abdominal", "náusea", "enjoo",
)
NEGATORS = {"nao", "sem", "nega", "negou", "nunca", "nenhum", "nenhuma", "ausencia", "ninguem"}
NEGATION_WINDOW = 4  # palavras antes do termo, dentro da mesma frase


def normalize(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text.lower()) if unicodedata.category(c) != "Mn")


def _is_negated(normalized: str, start: int) -> bool:
    clause = re.split(r"[.,;:!?]", normalized[:start])[-1]
    return any(word in NEGATORS for word in clause.split()[-NEGATION_WINDOW:])


def _find(normalized: str, terms: tuple[str, ...]) -> list[str]:
    found = []
    for term in terms:
        matches = re.finditer(rf"\b{re.escape(normalize(term))}\b", normalized)
        if any(not _is_negated(normalized, m.start()) for m in matches):
            found.append(term)
    return found


def find_terms(text: str) -> dict[str, list[str]]:
    """Termos citados e não negados: "não tenho falta de ar" não conta."""
    normalized = normalize(text)
    severe = _find(normalized, SEVERE_TERMS)
    if "dor muito forte" in severe and "dor forte" in severe:
        severe.remove("dor forte")
    return {"graves": severe, "leves": _find(normalized, MILD_TERMS)}


def split_text(text: str, size: int = TEXT_CHUNK_CHARS) -> list[str]:
    chunks, current = [], ""
    for word in text.split():
        if current and len(current) + len(word) + 1 > size:
            chunks.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        chunks.append(current)
    return chunks


class AudioAzureService:
    def __init__(self, voice_signal=None):
        self.voice_signal = voice_signal
        self.speech_key = os.getenv("AZURE_SPEECH_KEY", "")
        self.speech_region = os.getenv("AZURE_SPEECH_REGION", "")
        self.speech_language = os.getenv("AZURE_SPEECH_LANGUAGE", "pt-BR")
        self.lang_key = os.getenv("AZURE_LANGUAGE_KEY", "")
        self.lang_endpoint = os.getenv("AZURE_LANGUAGE_ENDPOINT", "")

    def speech_configured(self) -> bool:
        return bool(self.speech_key and self.speech_region and "sua_chave" not in self.speech_key)

    def language_configured(self) -> bool:
        return bool(self.lang_key and self.lang_endpoint and "sua_chave" not in self.lang_key)

    def _transcribe(self, audio: np.ndarray, duration_s: float) -> str | None:
        """Reconhecimento contínuo: recognize_once para na primeira pausa (até ~30 s)."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            # O SDK só aceita WAV PCM. Regravar em 16 kHz mono 16 bits evita falha com áudio float ou 24 bits.
            wav_path = str(Path(folder) / "consulta.wav")
            sf.write(wav_path, audio, 16000, subtype="PCM_16")
            speech_config = speechsdk.SpeechConfig(subscription=self.speech_key, region=self.speech_region)
            speech_config.speech_recognition_language = self.speech_language
            audio_config = speechsdk.audio.AudioConfig(filename=wav_path)
            recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

            parts: list[str] = []
            failure: list[str] = []
            done = threading.Event()

            def on_recognized(evt):
                if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech and evt.result.text:
                    parts.append(evt.result.text)

            def on_canceled(evt):
                details = evt.cancellation_details
                if details.reason == speechsdk.CancellationReason.Error:
                    failure.append(f"{details.error_code}: {details.error_details}")
                done.set()

            recognizer.recognized.connect(on_recognized)
            recognizer.canceled.connect(on_canceled)
            recognizer.session_stopped.connect(lambda evt: done.set())
            recognizer.start_continuous_recognition()
            finished = done.wait(timeout=max(60.0, duration_s * 2))
            recognizer.stop_continuous_recognition()
            # No Windows o SDK segura o arquivo até o reconhecedor ser descartado.
            del recognizer, audio_config
            if failure:
                raise RuntimeError(f"Azure Speech respondeu: {failure[0]}")
            if not finished:
                raise RuntimeError("Azure Speech não terminou a tempo.")
            return " ".join(parts) or None

    def _analyze_text(self, text: str) -> dict:
        client = TextAnalyticsClient(endpoint=self.lang_endpoint, credential=AzureKeyCredential(self.lang_key))
        language = self.speech_language.split("-")[0]
        chunks = split_text(text)
        sentiments = client.analyze_sentiment(documents=chunks, language=language)
        phrases = client.extract_key_phrases(documents=chunks, language=language)
        if any(item.is_error for item in [*sentiments, *phrases]):
            raise RuntimeError("Azure Text Analytics devolveu erro para o texto enviado.")
        weights = [len(chunk) for chunk in chunks]
        total = sum(weights)
        scores = {
            label: sum(getattr(item.confidence_scores, label) * w for item, w in zip(sentiments, weights)) / total
            for label in ("negative", "positive", "neutral")
        }
        dominant = max(scores, key=scores.get)
        key_phrases = list(dict.fromkeys(phrase for item in phrases for phrase in item.key_phrases))
        terms = find_terms(text)
        return {
            "sentimento_dominante": dominant,
            "score_negativo": round(scores["negative"], 2),
            "termos_criticos": terms["graves"],
            "termos_leves": terms["leves"],
            "frases_chave": key_phrases,
        }

    def _voice_signal(self, audio: np.ndarray, baseline_path: str | None) -> dict:
        if self.voice_signal is None or not self.voice_signal.available():
            return {"avaliado": False, "motivo": "O modelo treinado com o TAME Pain não está em backend/models."}
        baseline = load_audio(baseline_path) if baseline_path else None
        return self.voice_signal.assess(audio, baseline)

    def process_audio(self, wav_path: str, filename: str, baseline_path: str | None = None) -> dict:
        y, sample_rate = librosa.load(wav_path, sr=16000)
        result = {
            "acustica": {
                "arquivo": filename,
                "duracao_s": round(float(librosa.get_duration(y=y, sr=sample_rate)), 2),
                "rms_energy": round(float(np.mean(librosa.feature.rms(y=y))), 4),
                "zero_crossing_rate": round(float(np.mean(librosa.feature.zero_crossing_rate(y))), 4),
                "centroide_hz": round(float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sample_rate))), 1),
            },
            "sinal_vocal": self._voice_signal(y, baseline_path),
            "azure": {
                "speech_configurado": self.speech_configured(),
                "language_configurado": self.language_configured(),
                "erro": None,
            },
            "transcricao": None,
            "analise_texto": None,
        }

        if self.speech_configured():
            try:
                result["transcricao"] = self._transcribe(y, result["acustica"]["duracao_s"])
            except Exception as error:
                result["azure"]["erro"] = f"Falha na transcrição: {error}"
        if result["transcricao"] and self.language_configured():
            try:
                result["analise_texto"] = self._analyze_text(result["transcricao"])
            except Exception as error:
                result["azure"]["erro"] = f"Falha na análise do texto: {error}"

        analysis = result["analise_texto"]
        negative = bool(analysis) and analysis["sentimento_dominante"] == "negative" and analysis["score_negativo"] >= NEGATIVE_SPEECH_SCORE
        critical = bool(analysis) and bool(analysis["termos_criticos"])
        if analysis:
            severe = ", ".join(analysis["termos_criticos"]) or "nenhum"
            mild = ", ".join(analysis["termos_leves"]) or "nenhum"
            description = f"sentimento {analysis['sentimento_dominante']} na transcrição; sintomas graves: {severe}; outras queixas: {mild}"
        elif result["azure"]["erro"]:
            description = result["azure"]["erro"]
        else:
            description = "Azure não configurado; sem transcrição"
        result["alerta"] = negative or critical
        result["descricao"] = description
        return result
