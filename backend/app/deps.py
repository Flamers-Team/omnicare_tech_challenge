"""Serviços compartilhados. Os routers acessam `deps.<nome>` na hora da chamada, o que permite trocá-los nos testes."""
from app import config
from services.alert_service import AlertStore
from services.audio_azure_service import AudioAzureService
from services.clinical_anomaly_service import ClinicalTimelineService
from services.rehab_gait_service import RehabGaitAnalyzer
from services.surgery_video_service import SurgeryVideoAnalyzer
from services.voice_signal_service import VoiceSignalService

clinical_service = ClinicalTimelineService(config.SAMPLE_DIR / "mimic_demo")
gait_service = RehabGaitAnalyzer(config.MODELS_DIR / "marcha_referencia.json")
voice_signal_service = VoiceSignalService(config.MODELS_DIR / "tame_pain_logreg.joblib")
audio_service = AudioAzureService(voice_signal_service)
surgery_service = SurgeryVideoAnalyzer(config.MODELS_DIR, config.ENDOSCAPES_ZIP)
alerts = AlertStore(config.ALERTS_DB)
patient_context: dict[int, dict] = {}
