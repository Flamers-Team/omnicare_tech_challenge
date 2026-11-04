"""Laudos em texto, montados só com o que foi realmente medido."""
from __future__ import annotations

CLOSING = "Apoio à decisão; não substitui a avaliação da equipe."
LEVEL_NAMES = {"baixo": "baixo", "atencao": "de atenção", "alto": "alto"}


def _decimal(value: float) -> str:
    return f"{value:.2f}".replace(".", ",")


def _gait_section(gait: dict | None) -> str:
    if not gait:
        return "Marcha: sem avaliação."
    if not gait["avaliada"]:
        return f"Marcha: não avaliada. {gait['motivo']}"
    score = _decimal(gait["pontuacao"])
    limit = _decimal(gait["limiar"])
    if gait["alerta"]:
        main = gait["medida_mais_distante"]
        return f"Marcha: fora da referência (nota {score}, limiar {limit}). Maior desvio: {main['descricao'].lower()}, valor {_decimal(main['valor'])}."
    return f"Marcha: dentro da referência (nota {score}, limiar {limit})."


def _speech_section(speech: dict | None) -> str:
    if not speech:
        return "Fala: sem avaliação."
    text = f"Fala: {speech['descricao']}."
    voice = speech.get("sinal_vocal") or {}
    if voice.get("avaliado"):
        text += f" Voz: {voice['descricao'].lower()} (apoio, não é nota de dor)."
    return text


def inpatient_report(stay_id: int, at: str, vital_events: list[dict], prescription_events: list[dict], gait: dict | None, speech: dict | None, risk: dict) -> str:
    vitals = "; ".join(event["descricao"] for event in vital_events[-3:]) or "sem desvio sustentado"
    rx = "; ".join(event["descricao"] for event in prescription_events[-3:]) or "sem mudança destacada"
    return " ".join([
        f"Internação {stay_id}, referência {at.replace('T', ' ')[:16]}.",
        f"Risco {LEVEL_NAMES[risk['nivel']]} ({risk['pontuacao']} {'ponto' if risk['pontuacao'] == 1 else 'pontos'}).",
        f"Sinais vitais: {vitals}.",
        f"Prescrições: {rx}.",
        _gait_section(gait),
        _speech_section(speech),
        CLOSING,
    ])


def surgery_report(label: str, analysis: dict) -> str:
    if not analysis["avaliada"]:
        return f"{label}: não avaliada. {analysis['motivo']} {CLOSING}"
    used = [item for item in analysis["criterios"] if item["usado"]]
    skipped = [item["criterio"] for item in analysis["criterios"] if not item["usado"]]
    measures = "; ".join(
        f"{item['criterio']} com pico {_decimal(item['pico'])} (limiar {_decimal(item['limiar'])}), {'atingiu' if item['atingido'] else 'abaixo do limiar'}"
        for item in used
    )
    seen = ", ".join(f"{item['nome'].lower()} em {item['frames']}" for item in analysis["estruturas"] if item["confiavel"])
    conclusion = (
        "Indício de desvio: recomenda-se revisar o vídeo antes do primeiro clipe."
        if analysis["desvio"]
        else f"Sem indício de desvio nos critérios avaliados, o que não confirma a Visão Crítica completa, porque {' e '.join(skipped)} não {'entra' if len(skipped) == 1 else 'entram'} na decisão."
    )
    return " ".join([
        f"{label}: {analysis['frames_analisados']} frames, um a cada {analysis['intervalo_s']} segundos.",
        f"{measures}.",
        conclusion,
        f"Estruturas com detecção mais confiável: {seen}.",
        CLOSING,
    ])
