import asyncio
import json
import os
import uuid
from datetime import date
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from services.alert_service import LEVEL_ORDER, AlertStore, combined_risk, surgery_risk
from services.audio_azure_service import AudioAzureService
from services.clinical_anomaly_service import ClinicalTimelineService
from services.rehab_gait_service import RehabGaitAnalyzer
from services.report_service import inpatient_report, surgery_report
from services.surgery_video_service import CVS_FILE, DETECTOR_FILE, SurgeryVideoAnalyzer
from services.voice_signal_service import VoiceInputError, VoiceSignalService

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend_js"
UPLOAD_DIR = BASE_DIR / "temp_uploads"
GAIT_EXAMPLES = {
    "normal": BASE_DIR / "sample_data" / "marcha" / "caminhada_normal.csv.gz",
    "alterada": BASE_DIR / "sample_data" / "marcha" / "caminhada_tronco_20_graus.csv.gz",
}

GAIT_CSV = {".csv"}
GAIT_VIDEO = {".mp4", ".avi", ".mov", ".mkv"}
AUDIO_TYPES = {".wav"}
MAX_CSV_BYTES = 100 * 1024 * 1024
MAX_VIDEO_BYTES = 500 * 1024 * 1024
MAX_AUDIO_BYTES = 50 * 1024 * 1024
ALERT_STREAM_SECONDS = 1.0

app = FastAPI(title="OmniCare AI - Tech Challenge Fase 4")

clinical_service = ClinicalTimelineService(BASE_DIR / "sample_data" / "mimic_demo")
gait_service = RehabGaitAnalyzer(BASE_DIR / "models" / "marcha_referencia.json")
voice_signal_service = VoiceSignalService(BASE_DIR / "models" / "tame_pain_logreg.joblib")
audio_service = AudioAzureService(voice_signal_service)
# A base tem licença restrita e fica fora do projeto. Em outro computador, informe o caminho em ENDOSCAPES_ZIP.
surgery_service = SurgeryVideoAnalyzer(
    BASE_DIR / "models",
    os.getenv("ENDOSCAPES_ZIP") or Path.home() / "Downloads" / "endoscapes_omnicare.zip",
)
alerts = AlertStore(BASE_DIR / "dados_locais" / "alertas.db")
patient_context: dict[int, dict] = {}


class Confirmation(BaseModel):
    responsavel: str = Field(min_length=2, max_length=80)


def save_upload(upload: UploadFile, allowed: set[str], max_bytes: int) -> Path:
    """Salva o envio com um nome gerado aqui, sem usar o caminho que veio do navegador."""
    extension = Path(upload.filename or "").suffix.lower()
    if extension not in allowed:
        raise HTTPException(400, f"Tipo de arquivo não aceito. Use: {', '.join(sorted(allowed))}.")
    UPLOAD_DIR.mkdir(exist_ok=True)
    target = UPLOAD_DIR / f"{uuid.uuid4().hex}{extension}"
    size = 0
    try:
        with target.open("wb") as output:
            while chunk := upload.file.read(1024 * 1024):
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(413, f"Arquivo maior que o limite de {max_bytes // (1024 * 1024)} MB.")
                output.write(chunk)
    except HTTPException:
        target.unlink(missing_ok=True)
        raise
    return target


def require_stay(stay_id: int) -> None:
    if stay_id not in clinical_service.stays():
        raise HTTPException(404, "Internação não encontrada.")


def current_risk(stay_id: int, timeline: dict, at: pd.Timestamp) -> dict:
    context = patient_context.get(stay_id, {})
    return combined_risk(timeline["eventos_sinais_vitais"], timeline["eventos_prescricao"], at, context.get("marcha"), context.get("fala"))


def timeline_payload(stay_id: int) -> dict:
    try:
        timeline = clinical_service.timeline(stay_id)
    except KeyError as error:
        raise HTTPException(404, "Internação não encontrada.") from error
    context = patient_context.get(stay_id, {})
    at = pd.Timestamp(timeline["horas"][-1])
    risk = current_risk(stay_id, timeline, at)
    report = inpatient_report(stay_id, timeline["horas"][-1], timeline["eventos_sinais_vitais"], timeline["eventos_prescricao"], context.get("marcha"), context.get("fala"), risk)
    return {**timeline, "marcha": context.get("marcha"), "fala": context.get("fala"), "risco": risk, "laudo": report}


def raise_alert_if_needed(stay_id: int, risk: dict, report: str, occurred_at: str, origin: str = "monitor") -> dict | None:
    if risk["nivel"] == "baixo":
        return None
    return alerts.add(f"Internação {stay_id}", risk, report, occurred_at, origin)


def store_context_and_alert(stay_id: int, key: str, result: dict) -> dict:
    patient_context.setdefault(stay_id, {})[key] = result
    payload = timeline_payload(stay_id)
    alert = raise_alert_if_needed(stay_id, payload["risco"], payload["laudo"], payload["horas"][-1], key)
    return {"resultado": result, "risco": payload["risco"], "laudo": payload["laudo"], "alerta": alert}


def sse(data: dict, event: str = "message") -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.get("/api/v1/status")
def system_status():
    return {
        "sinais_vitais_prescricoes": {"ativo": True, "fonte": "MIMIC-IV Demo 2.2"},
        "marcha": {"ativo": True, "exemplos": sorted(GAIT_EXAMPLES)},
        "sinal_vocal": {"ativo": voice_signal_service.available()},
        "azure_speech": {"ativo": audio_service.speech_configured()},
        "azure_language": {"ativo": audio_service.language_configured()},
        "cirurgia": {"ativo": surgery_service.models_available()},
    }


@app.get("/api/v1/pacientes")
def list_patients():
    return {"sugerida": clinical_service.suggested_stay(), "internacoes": clinical_service.summaries()}


@app.get("/api/v1/pacientes/{stay_id}/linha-do-tempo")
def patient_timeline(stay_id: int):
    return timeline_payload(stay_id)


@app.get("/api/v1/pacientes/{stay_id}/monitor")
async def patient_monitor(stay_id: int, intervalo: float = Query(0.3, ge=0, le=5)):
    """Reproduz a internação hora a hora, como se os dados chegassem agora."""
    require_stay(stay_id)

    async def stream():
        previous_level = "baixo"
        for step in clinical_service.replay(stay_id):
            context = patient_context.get(stay_id, {})
            at = pd.Timestamp(step["hora"])
            risk = combined_risk(step["eventos_sinais_vitais"], step["eventos_prescricao"], at, context.get("marcha"), context.get("fala"))
            alert = None
            # Só avisa quando o risco sobe, para a equipe não receber o mesmo aviso a cada hora.
            if LEVEL_ORDER[risk["nivel"]] > LEVEL_ORDER[previous_level]:
                report = inpatient_report(stay_id, step["hora"], step["eventos_sinais_vitais"], step["eventos_prescricao"], context.get("marcha"), context.get("fala"), risk)
                alert = raise_alert_if_needed(stay_id, risk, report, step["hora"])
            previous_level = risk["nivel"]
            yield sse({"hora": step["hora"], "valores": step["valores"], "novos_eventos": step["novos_eventos"], "risco": risk, "alerta": alert})
            await asyncio.sleep(intervalo)
        yield sse({"fim": True}, event="fim")

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/api/v1/pacientes/{stay_id}/marcha/exemplo/{nome}")
def gait_example(stay_id: int, nome: str):
    require_stay(stay_id)
    if nome not in GAIT_EXAMPLES:
        raise HTTPException(404, "Exemplo não encontrado.")
    result = gait_service.analyze_skeleton(GAIT_EXAMPLES[nome])
    result["arquivo"] = GAIT_EXAMPLES[nome].name
    return store_context_and_alert(stay_id, "marcha", result)


@app.post("/api/v1/pacientes/{stay_id}/marcha")
def gait_upload(stay_id: int, arquivo: UploadFile = File(...)):
    require_stay(stay_id)
    is_csv = Path(arquivo.filename or "").suffix.lower() in GAIT_CSV
    path = save_upload(arquivo, GAIT_CSV | GAIT_VIDEO, MAX_CSV_BYTES if is_csv else MAX_VIDEO_BYTES)
    try:
        result = gait_service.analyze_skeleton(path) if is_csv else gait_service.analyze_video(path)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    finally:
        path.unlink(missing_ok=True)
    result["arquivo"] = Path(arquivo.filename).name
    return store_context_and_alert(stay_id, "marcha", result)


@app.post("/api/v1/pacientes/{stay_id}/voz")
def voice_upload(stay_id: int, arquivo: UploadFile = File(...), base: UploadFile | None = File(None)):
    """`base` é a gravação da mesma pessoa em repouso. É opcional, mas sem ela o sinal vocal não soma pontos."""
    require_stay(stay_id)
    path = save_upload(arquivo, AUDIO_TYPES, MAX_AUDIO_BYTES)
    base_path = None
    try:
        if base is not None:
            base_path = save_upload(base, AUDIO_TYPES, MAX_AUDIO_BYTES)
        result = audio_service.process_audio(str(path), Path(arquivo.filename).name, str(base_path) if base_path else None)
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


def require_surgery_models() -> None:
    if not surgery_service.models_available():
        raise HTTPException(503, f"Os modelos de cirurgia não estão em backend/models. Copie {DETECTOR_FILE} e {CVS_FILE} para essa pasta.")


def surgery_payload(label: str, analysis: dict) -> dict:
    report = surgery_report(label, analysis)
    alert = None
    if analysis["avaliada"] and analysis["desvio"]:
        # A data do dia evita duplicar o alerta quando a mesma análise é repetida.
        alert = alerts.add(label, surgery_risk(analysis), report, date.today().isoformat(), "cirurgia")
    return {"resultado": analysis, "laudo": report, "alerta": alert}


@app.get("/api/v1/cirurgia")
def surgery_info():
    return {
        "modelos_ativos": surgery_service.models_available(),
        "videos_base": surgery_service.dataset_videos(),
        "desempenho": surgery_service.performance(),
    }


@app.post("/api/v1/cirurgia/base/{video_id}")
def surgery_dataset_video(video_id: int):
    require_surgery_models()
    if video_id not in surgery_service.dataset_videos():
        raise HTTPException(404, "Vídeo não encontrado no conjunto de teste local.")
    return surgery_payload(f"Cirurgia, vídeo {video_id} do teste", surgery_service.analyze_dataset_video(video_id))


@app.post("/api/v1/cirurgia/video")
def surgery_upload(arquivo: UploadFile = File(...), fim_s: float | None = Form(None, gt=0)):
    require_surgery_models()
    path = save_upload(arquivo, GAIT_VIDEO, MAX_VIDEO_BYTES)
    try:
        analysis = surgery_service.analyze_video(path, fim_s)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    finally:
        path.unlink(missing_ok=True)
    return surgery_payload(f"Cirurgia, {Path(arquivo.filename).name}", analysis)


@app.get("/api/v1/alertas")
def list_alerts():
    return {"alertas": alerts.list()}


@app.get("/api/v1/alertas/stream")
async def alert_stream():
    async def stream():
        seen = -1
        while True:
            if alerts.version != seen:
                seen = alerts.version
                yield sse({"alertas": alerts.list()})
            await asyncio.sleep(ALERT_STREAM_SECONDS)

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/api/v1/alertas/{alert_id}/confirmar")
def confirm_alert(alert_id: int, body: Confirmation):
    responsible = body.responsavel.strip()
    if len(responsible) < 2:
        raise HTTPException(422, "Informe o nome de quem confirmou.")
    alert = alerts.confirm(alert_id, responsible)
    if alert is None:
        raise HTTPException(404, "Alerta não encontrado.")
    return alert


@app.get("/")
def home():
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
