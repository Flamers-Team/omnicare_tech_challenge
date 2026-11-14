import asyncio

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app import config, deps
from app.monitoring import sse
from app.schemas import Confirmation

router = APIRouter(prefix="/api/v1/alertas", tags=["alertas"])


@router.get("")
def list_alerts():
    return {"alertas": deps.alerts.list()}


@router.delete("")
def clear_alerts(incluir_resultados: bool = False):
    """Apaga o histórico de alertas. Com `incluir_resultados`, também esquece a marcha e a fala analisadas em cada internação.

    Os dados clínicos (sinais vitais e prescrições da base) nunca são tocados.
    """
    if not config.ALLOW_RESET:
        raise HTTPException(403, "A limpeza está desativada neste ambiente.")
    deleted = deps.alerts.clear()
    if incluir_resultados:
        deps.patient_context.clear()
    return {"alertas_apagados": deleted, "resultados_limpos": incluir_resultados}


@router.get("/stream")
async def alert_stream():
    async def stream():
        seen = -1
        while True:
            if deps.alerts.version != seen:
                seen = deps.alerts.version
                yield sse({"alertas": deps.alerts.list()})
            await asyncio.sleep(config.ALERT_STREAM_SECONDS)

    return StreamingResponse(stream(), media_type="text/event-stream")


@router.post("/{alert_id}/confirmar")
def confirm_alert(alert_id: int, body: Confirmation):
    responsible = body.responsavel.strip()
    if len(responsible) < 2:
        raise HTTPException(422, "Informe o nome de quem confirmou.")
    alert = deps.alerts.confirm(alert_id, responsible)
    if alert is None:
        raise HTTPException(404, "Alerta não encontrado.")
    return alert
