import asyncio

import pandas as pd
from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from app import deps
from app.monitoring import raise_alert_if_needed, require_stay, sse, timeline_payload
from services.alert_service import LEVEL_ORDER, combined_risk
from services.report_service import inpatient_report

router = APIRouter(prefix="/api/v1/pacientes", tags=["pacientes"])


@router.get("")
def list_patients():
    return {"sugerida": deps.clinical_service.suggested_stay(), "internacoes": deps.clinical_service.summaries()}


@router.get("/{stay_id}/linha-do-tempo")
def patient_timeline(stay_id: int):
    return timeline_payload(stay_id)


@router.get("/{stay_id}/monitor")
async def patient_monitor(stay_id: int, intervalo: float = Query(0.3, ge=0, le=5)):
    """Reproduz a internação hora a hora, como se os dados chegassem agora."""
    require_stay(stay_id)

    async def stream():
        previous_level = "baixo"
        for step in deps.clinical_service.replay(stay_id):
            context = deps.patient_context.get(stay_id, {})
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
