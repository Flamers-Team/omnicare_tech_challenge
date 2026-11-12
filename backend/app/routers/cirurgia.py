from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app import config, deps
from app.monitoring import surgery_payload
from app.uploads import save_upload
from services.surgery_video_service import CVS_FILE, DETECTOR_FILE

router = APIRouter(prefix="/api/v1/cirurgia", tags=["cirurgia"])


def require_surgery_models() -> None:
    if not deps.surgery_service.models_available():
        raise HTTPException(503, f"Os modelos de cirurgia não estão em backend/models. Copie {DETECTOR_FILE} e {CVS_FILE} para essa pasta.")


@router.get("")
def surgery_info():
    return {
        "modelos_ativos": deps.surgery_service.models_available(),
        "videos_base": deps.surgery_service.dataset_videos(),
        "desempenho": deps.surgery_service.performance(),
    }


@router.post("/base/{video_id}")
def surgery_dataset_video(video_id: int):
    require_surgery_models()
    if video_id not in deps.surgery_service.dataset_videos():
        raise HTTPException(404, "Vídeo não encontrado no conjunto de teste local.")
    return surgery_payload(f"Cirurgia, vídeo {video_id} do teste", deps.surgery_service.analyze_dataset_video(video_id))


@router.post("/video")
def surgery_upload(arquivo: UploadFile = File(...), fim_s: float | None = Form(None, gt=0)):
    require_surgery_models()
    path = save_upload(arquivo, config.GAIT_VIDEO, config.MAX_VIDEO_BYTES)
    try:
        analysis = deps.surgery_service.analyze_video(path, fim_s)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    finally:
        path.unlink(missing_ok=True)
    return surgery_payload(f"Cirurgia, {Path(arquivo.filename).name}", analysis)
