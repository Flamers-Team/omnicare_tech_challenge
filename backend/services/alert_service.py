"""Pontuação de risco combinada e histórico de alertas com confirmação."""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Pesos definidos pelo projeto. Não foram validados clinicamente.
POINTS_PER_VITAL = 2
MAX_VITAL_POINTS = 4
POINTS_HIGH_PRESCRIPTION = 2
POINTS_MODERATE_PRESCRIPTION = 1
MAX_PRESCRIPTION_POINTS = 2
POINTS_GAIT = 2
POINTS_NEGATIVE_SPEECH = 1
POINTS_VOICE_SIGNAL = 1
# Só chega a "atenção": no teste, 4 de 14 cirurgias com a Visão Crítica atingida também geraram o aviso.
POINTS_SURGERY_DEVIATION = 2
ATTENTION_POINTS = 2
HIGH_POINTS = 4
VITAL_RECENCY = pd.Timedelta(hours=6)
PRESCRIPTION_RECENCY = pd.Timedelta(hours=24)
RISK_NOTE = "Pontuação de apoio à decisão, com pesos definidos pelo projeto e sem validação clínica."
LEVEL_ORDER = {"baixo": 0, "atencao": 1, "alto": 2}


def risk_level(points: int) -> str:
    if points >= HIGH_POINTS:
        return "alto"
    return "atencao" if points >= ATTENTION_POINTS else "baixo"


def combined_risk(
    vital_events: list[dict],
    prescription_events: list[dict],
    at: pd.Timestamp,
    gait: dict | None = None,
    speech: dict | None = None,
) -> dict:
    """Soma o que está ativo no instante `at` e guarda o motivo de cada ponto."""
    reasons: list[dict] = []
    active_vitals: dict[str, dict] = {}
    for event in vital_events:
        started = pd.Timestamp(event["inicio"]) <= at
        recent = at - pd.Timestamp(event["fim"]) <= VITAL_RECENCY
        if started and recent:
            active_vitals[event["sinal"]] = event
    vital_points = 0
    for event in active_vitals.values():
        points = min(POINTS_PER_VITAL, MAX_VITAL_POINTS - vital_points)
        vital_points += points
        reasons.append({"fonte": "sinais_vitais", "descricao": event["descricao"], "pontos": points})

    prescription_points = 0
    recent_rx = [
        event for event in prescription_events
        if pd.Timedelta(0) <= at - pd.Timestamp(event["inicio"]) <= PRESCRIPTION_RECENCY
    ]
    for event in sorted(recent_rx, key=lambda item: item["gravidade"] != "alta"):
        value = POINTS_HIGH_PRESCRIPTION if event["gravidade"] == "alta" else POINTS_MODERATE_PRESCRIPTION
        points = min(value, MAX_PRESCRIPTION_POINTS - prescription_points)
        if points <= 0:
            break
        prescription_points += points
        reasons.append({"fonte": "prescricao", "descricao": event["descricao"], "pontos": points})

    if gait and gait.get("alerta"):
        main_measure = gait["medida_mais_distante"]
        reasons.append({
            "fonte": "marcha",
            "descricao": f"Marcha fora da referência; maior desvio: {main_measure['descricao'].lower()}",
            "pontos": POINTS_GAIT,
        })
    if speech and speech.get("alerta"):
        reasons.append({"fonte": "fala", "descricao": speech["descricao"], "pontos": POINTS_NEGATIVE_SPEECH})
    voice = (speech or {}).get("sinal_vocal") or {}
    if voice.get("soma_no_risco"):
        reasons.append({"fonte": "voz", "descricao": voice["descricao"], "pontos": POINTS_VOICE_SIGNAL})

    total = sum(reason["pontos"] for reason in reasons)
    return {"pontuacao": total, "nivel": risk_level(total), "motivos": reasons, "observacao": RISK_NOTE}


def surgery_risk(analysis: dict) -> dict:
    missing = [item["criterio"] for item in analysis["criterios"] if item["usado"] and not item["atingido"]]
    reason = {"fonte": "cirurgia", "descricao": f"Indício de desvio: {' e '.join(missing)} abaixo do limiar antes do primeiro clipe", "pontos": POINTS_SURGERY_DEVIATION}
    return {"pontuacao": POINTS_SURGERY_DEVIATION, "nivel": risk_level(POINTS_SURGERY_DEVIATION), "motivos": [reason], "observacao": RISK_NOTE}


class AlertStore:
    """Guarda os alertas em SQLite. Repetir a mesma demonstração não duplica o alerta."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.version = 0
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._db:
            self._db.execute(
                """
                CREATE TABLE IF NOT EXISTS alertas (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    paciente TEXT NOT NULL,
                    nivel TEXT NOT NULL,
                    pontuacao INTEGER NOT NULL,
                    motivos TEXT NOT NULL,
                    laudo TEXT NOT NULL,
                    ocorrido_em TEXT NOT NULL,
                    origem TEXT NOT NULL,
                    criado_em TEXT NOT NULL,
                    confirmado_por TEXT,
                    confirmado_em TEXT,
                    UNIQUE (paciente, nivel, ocorrido_em, origem)
                )
                """
            )

    @staticmethod
    def _as_dict(row: sqlite3.Row) -> dict:
        item = dict(row)
        item["motivos"] = json.loads(item["motivos"])
        return item

    def add(self, patient: str, risk: dict, report: str, occurred_at: str, origin: str = "monitor") -> dict:
        with self._lock, self._db:
            self._db.execute(
                """
                INSERT OR IGNORE INTO alertas (paciente, nivel, pontuacao, motivos, laudo, ocorrido_em, origem, criado_em)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    patient,
                    risk["nivel"],
                    risk["pontuacao"],
                    json.dumps(risk["motivos"], ensure_ascii=False),
                    report,
                    occurred_at,
                    origin,
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                ),
            )
            row = self._db.execute(
                "SELECT * FROM alertas WHERE paciente = ? AND nivel = ? AND ocorrido_em = ? AND origem = ?",
                (patient, risk["nivel"], occurred_at, origin),
            ).fetchone()
            self.version += 1
        return self._as_dict(row)

    def list(self, limit: int = 100) -> list[dict]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM alertas ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [self._as_dict(row) for row in rows]

    def confirm(self, alert_id: int, responsible: str) -> dict | None:
        """Registra quem viu o alerta. Devolve None se ele não existir; o primeiro registro é mantido."""
        with self._lock, self._db:
            self._db.execute(
                "UPDATE alertas SET confirmado_por = ?, confirmado_em = ? WHERE id = ? AND confirmado_por IS NULL",
                (responsible, datetime.now(timezone.utc).isoformat(timespec="seconds"), alert_id),
            )
            row = self._db.execute("SELECT * FROM alertas WHERE id = ?", (alert_id,)).fetchone()
            self.version += 1
        return self._as_dict(row) if row else None
