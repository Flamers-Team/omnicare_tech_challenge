"""Regras de montagem da resposta de uma internação e de criação de alertas."""
import json
from datetime import date

import pandas as pd
from fastapi import HTTPException

from app import deps
from services.alert_service import combined_risk, surgery_risk
from services.report_service import inpatient_report, surgery_report


def sse(data: dict, event: str = "message") -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def require_stay(stay_id: int) -> None:
    if stay_id not in deps.clinical_service.stays():
        raise HTTPException(404, "Internação não encontrada.")


def current_risk(stay_id: int, timeline: dict, at: pd.Timestamp) -> dict:
    context = deps.patient_context.get(stay_id, {})
    return combined_risk(timeline["eventos_sinais_vitais"], timeline["eventos_prescricao"], at, context.get("marcha"), context.get("fala"))


def timeline_payload(stay_id: int) -> dict:
    try:
        timeline = deps.clinical_service.timeline(stay_id)
    except KeyError as error:
        raise HTTPException(404, "Internação não encontrada.") from error
    context = deps.patient_context.get(stay_id, {})
    at = pd.Timestamp(timeline["horas"][-1])
    risk = current_risk(stay_id, timeline, at)
    report = inpatient_report(stay_id, timeline["horas"][-1], timeline["eventos_sinais_vitais"], timeline["eventos_prescricao"], context.get("marcha"), context.get("fala"), risk)
    return {**timeline, "marcha": context.get("marcha"), "fala": context.get("fala"), "risco": risk, "laudo": report}


def raise_alert_if_needed(stay_id: int, risk: dict, report: str, occurred_at: str, origin: str = "monitor") -> dict | None:
    if risk["nivel"] == "baixo":
        return None
    return deps.alerts.add(f"Internação {stay_id}", risk, report, occurred_at, origin)


def store_context_and_alert(stay_id: int, key: str, result: dict) -> dict:
    deps.patient_context.setdefault(stay_id, {})[key] = result
    payload = timeline_payload(stay_id)
    alert = raise_alert_if_needed(stay_id, payload["risco"], payload["laudo"], payload["horas"][-1], key)
    return {"resultado": result, "risco": payload["risco"], "laudo": payload["laudo"], "alerta": alert}


def surgery_payload(label: str, analysis: dict) -> dict:
    report = surgery_report(label, analysis)
    alert = None
    if analysis["avaliada"] and analysis["desvio"]:
        # A data do dia evita duplicar o alerta quando a mesma análise é repetida.
        alert = deps.alerts.add(label, surgery_risk(analysis), report, date.today().isoformat(), "cirurgia")
    return {"resultado": analysis, "laudo": report, "alerta": alert}
