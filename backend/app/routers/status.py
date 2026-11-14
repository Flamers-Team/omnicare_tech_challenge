from fastapi import APIRouter

from app import config, deps

router = APIRouter(prefix="/api/v1", tags=["status"])


@router.get("/status")
def system_status():
    return {
        "sinais_vitais_prescricoes": {"ativo": True, "fonte": "MIMIC-IV Demo 2.2"},
        "marcha": {"ativo": True, "exemplos": sorted(config.GAIT_EXAMPLES)},
        "sinal_vocal": {"ativo": deps.voice_signal_service.available()},
        "azure_speech": {"ativo": deps.audio_service.speech_configured()},
        "azure_language": {"ativo": deps.audio_service.language_configured()},
        "cirurgia": {"ativo": deps.surgery_service.models_available()},
        "limpeza_permitida": config.ALLOW_RESET,
    }
