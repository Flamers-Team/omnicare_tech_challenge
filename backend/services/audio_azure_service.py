import os

import azure.cognitiveservices.speech as speechsdk
import librosa
import numpy as np
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

from services.voice_features import load_audio

load_dotenv()

NEGATIVE_SPEECH_SCORE = 0.6


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

    def _transcribe(self, wav_path: str) -> str | None:
        speech_config = speechsdk.SpeechConfig(subscription=self.speech_key, region=self.speech_region)
        speech_config.speech_recognition_language = self.speech_language
        audio_config = speechsdk.audio.AudioConfig(filename=wav_path)
        recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
        result = recognizer.recognize_once_async().get()
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            return result.text
        if result.reason == speechsdk.ResultReason.NoMatch:
            return None
        raise RuntimeError(f"Azure Speech respondeu: {result.reason}")

    def _analyze_text(self, text: str) -> dict:
        client = TextAnalyticsClient(endpoint=self.lang_endpoint, credential=AzureKeyCredential(self.lang_key))
        language = self.speech_language.split("-")[0]
        sentiment = client.analyze_sentiment(documents=[text], language=language)[0]
        phrases = client.extract_key_phrases(documents=[text], language=language)[0]
        if sentiment.is_error or phrases.is_error:
            raise RuntimeError("Azure Text Analytics devolveu erro para o texto enviado.")
        return {
            "sentimento_dominante": sentiment.sentiment,
            "score_negativo": round(sentiment.confidence_scores.negative, 2),
            "termos_criticos": list(phrases.key_phrases),
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
                result["transcricao"] = self._transcribe(wav_path)
            except Exception as error:
                result["azure"]["erro"] = f"Falha na transcrição: {error}"
        if result["transcricao"] and self.language_configured():
            try:
                result["analise_texto"] = self._analyze_text(result["transcricao"])
            except Exception as error:
                result["azure"]["erro"] = f"Falha na análise do texto: {error}"

        analysis = result["analise_texto"]
        negative = bool(analysis) and analysis["sentimento_dominante"] == "negative" and analysis["score_negativo"] >= NEGATIVE_SPEECH_SCORE
        if analysis:
            terms = ", ".join(analysis["termos_criticos"]) or "nenhum termo"
            description = f"sentimento {analysis['sentimento_dominante']} na transcrição; termos: {terms}"
        elif result["azure"]["erro"]:
            description = result["azure"]["erro"]
        else:
            description = "Azure não configurado; sem transcrição"
        result["alerta"] = negative
        result["descricao"] = description
        return result
