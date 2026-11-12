from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app import config, deps
from app.monitoring import require_stay, store_context_and_alert
from app.uploads import save_upload

router = APIRouter(prefix="/api/v1/pacientes/{stay_id}/marcha", tags=["marcha"])


@router.post("/exemplo/{nome}")
def gait_example(stay_id: int, nome: str):
    require_stay(stay_id)
    if nome not in config.GAIT_EXAMPLES:
        raise HTTPException(404, "Exemplo não encontrado.")
    result = deps.gait_service.analyze_skeleton(config.GAIT_EXAMPLES[nome])
    result["arquivo"] = config.GAIT_EXAMPLES[nome].name
    return store_context_and_alert(stay_id, "marcha", result)


@router.post("")
def gait_upload(stay_id: int, arquivo: UploadFile = File(...)):
    require_stay(stay_id)
    is_csv = Path(arquivo.filename or "").suffix.lower() in config.GAIT_CSV
    path = save_upload(arquivo, config.GAIT_CSV | config.GAIT_VIDEO, config.MAX_CSV_BYTES if is_csv else config.MAX_VIDEO_BYTES)
    try:
        result = deps.gait_service.analyze_skeleton(path) if is_csv else deps.gait_service.analyze_video(path)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    finally:
        path.unlink(missing_ok=True)
    result["arquivo"] = Path(arquivo.filename).name
    return store_context_and_alert(stay_id, "marcha", result)
