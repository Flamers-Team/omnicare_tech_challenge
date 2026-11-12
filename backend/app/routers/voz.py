from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app import config, deps
from app.monitoring import require_stay, store_context_and_alert
from app.uploads import save_upload
from services.voice_signal_service import VoiceInputError

router = APIRouter(prefix="/api/v1/pacientes/{stay_id}/voz", tags=["voz"])


@router.post("")
def voice_upload(stay_id: int, arquivo: UploadFile = File(...), base: UploadFile | None = File(None)):
    """`base` é a gravação da mesma pessoa em repouso. É opcional, mas sem ela o sinal vocal não soma pontos."""
    require_stay(stay_id)
    path = save_upload(arquivo, config.AUDIO_TYPES, config.MAX_AUDIO_BYTES)
    base_path = None
    try:
        if base is not None:
            base_path = save_upload(base, config.AUDIO_TYPES, config.MAX_AUDIO_BYTES)
        result = deps.audio_service.process_audio(str(path), Path(arquivo.filename).name, str(base_path) if base_path else None)
    except HTTPException:
        raise
    except VoiceInputError as error:
        raise HTTPException(422, str(error)) from error
    except Exception as error:
        raise HTTPException(422, "Não foi possível ler o áudio enviado.") from error
    finally:
        path.unlink(missing_ok=True)
        if base_path:
            base_path.unlink(missing_ok=True)
    return store_context_and_alert(stay_id, "fala", result)
