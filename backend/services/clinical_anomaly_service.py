"""Anomalias em séries de sinais vitais e na evolução de prescrições (MIMIC-IV Demo)."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# itemids do MIMIC-IV. A pressão arterial invasiva tem prioridade sobre a não invasiva.
VITAL_ITEMS = {
    220045: "HR",
    220210: "RR",
    220277: "SPO2",
    220050: "SBP_INV",
    220179: "SBP_NI",
    223761: "TEMP_F",
}
VITALS = ("HR", "RR", "SPO2", "SBP", "TEMP")
VITAL_LABELS = {
    "HR": "frequência cardíaca",
    "RR": "frequência respiratória",
    "SPO2": "oxigenação",
    "SBP": "pressão sistólica",
    "TEMP": "temperatura",
}
VALID_RANGE = {"HR": (20, 250), "RR": (4, 70), "SPO2": (50, 100), "SBP": (40, 280), "TEMP": (30, 43)}

# Variação mínima considerada, para pacientes muito estáveis não gerarem alerta por ruído.
MIN_SCALE = {"HR": 5.0, "RR": 2.0, "SPO2": 1.5, "SBP": 8.0, "TEMP": 0.3}
ONE_SIDED_LOW = {"SPO2"}  # oxigenação alta não é sinal de risco
BASELINE_WINDOW = "24h"
BASELINE_MIN_POINTS = 6
SUSTAINED_POINTS = 2
EVENT_GAP = pd.Timedelta(hours=3)
DEFAULT_Z_THRESHOLD = 3.0  # escolhido em training/analyze_mimic_demo.py

# Regras fixas herdadas do projeto; continuam como comparação e explicação.
RULES = {"HR": lambda s: s > 100, "RR": lambda s: s > 20, "SBP": lambda s: s < 90, "SPO2": lambda s: s < 92}

NON_DRUG_PATTERN = re.compile(r"sodium chloride|dextrose|flush|^bag$|sterile water|lactated|syringe|vial", re.I)
SLIDING_SCALE_PATTERN = re.compile(r"insulin", re.I)  # dose ajustada de rotina pela glicemia
# Lista simplificada do projeto, não é uma lista clínica validada.
HIGH_ALERT_PATTERN = re.compile(
    r"heparin|warfarin|enoxaparin|morphine|hydromorphone|fentanyl|oxycodone|norepinephrine|vasopressin|"
    r"phenylephrine|epinephrine|dopamine|amiodarone|digoxin|propofol|midazolam",
    re.I,
)
DOSE_CHANGE_RATIO = 2.0
LARGE_DOSE_CHANGE_RATIO = 5.0
LATE_START = pd.Timedelta(hours=24)
MIN_DEMO_HOURS = 36
MAX_DEMO_HOURS = 120


def load_hourly_vitals(mimic_dir: Path) -> pd.DataFrame:
    """Lê o chartevents e devolve uma linha por internação e hora, com os erros de sensor removidos."""
    chart = pd.read_csv(
        Path(mimic_dir) / "icu" / "chartevents.csv.gz",
        usecols=["subject_id", "hadm_id", "stay_id", "charttime", "itemid", "valuenum"],
        parse_dates=["charttime"],
    )
    chart = chart[chart["itemid"].isin(VITAL_ITEMS)].dropna(subset=["valuenum"])
    chart["vital"] = chart["itemid"].map(VITAL_ITEMS)
    chart["hour"] = chart["charttime"].dt.floor("h")
    wide = chart.pivot_table(
        index=["subject_id", "hadm_id", "stay_id", "hour"], columns="vital", values="valuenum", aggfunc="median"
    ).reset_index()
    wide.columns.name = None
    for column in VITAL_ITEMS.values():
        if column not in wide:
            wide[column] = np.nan
    wide["SBP"] = wide["SBP_INV"].where(wide["SBP_INV"].notna(), wide["SBP_NI"])
    wide["TEMP"] = (wide["TEMP_F"] - 32) * 5 / 9
    wide = wide[["subject_id", "hadm_id", "stay_id", "hour", *VITALS]].sort_values(["stay_id", "hour"]).reset_index(drop=True)
    return remove_invalid_values(wide)


def remove_invalid_values(wide: pd.DataFrame) -> pd.DataFrame:
    out = wide.copy()
    for vital, (low, high) in VALID_RANGE.items():
        out[vital] = out[vital].where(out[vital].between(low, high))
    return out


def rolling_z(series: pd.Series, vital: str) -> pd.Series:
    """Distância de cada valor até as 24 h anteriores do próprio paciente, sem olhar o futuro."""
    clean = series.dropna()
    if clean.empty:
        return pd.Series(np.nan, index=series.index)
    window = clean.rolling(BASELINE_WINDOW, closed="left", min_periods=BASELINE_MIN_POINTS)
    median = window.median()
    mad = window.apply(lambda values: np.median(np.abs(values - np.median(values))), raw=True)
    scale = (1.4826 * mad).clip(lower=MIN_SCALE[vital])
    return ((clean - median) / scale).reindex(series.index)


def stay_z_scores(stay: pd.DataFrame) -> pd.DataFrame:
    """Recebe uma internação indexada pela hora e devolve a distância de cada sinal."""
    return pd.DataFrame({vital: rolling_z(stay[vital], vital) for vital in VITALS}, index=stay.index)


def flagged_hours(z: pd.Series, vital: str, threshold: float) -> pd.Series:
    """Marca as horas em que o desvio se manteve, para um valor isolado não disparar alerta."""
    outside = (-z if vital in ONE_SIDED_LOW else z.abs()) >= threshold
    valid = z.dropna().index
    sustained = outside.loc[valid].rolling(SUSTAINED_POINTS).sum() >= SUSTAINED_POINTS
    return sustained.reindex(z.index, fill_value=False).astype(bool)


def event_starts(flags: pd.Series) -> list[pd.Timestamp]:
    """Junta horas próximas em um evento e devolve quando cada evento começa."""
    hours = flags.index[flags.to_numpy()]
    starts: list[pd.Timestamp] = []
    previous = None
    for hour in hours:
        if previous is None or hour - previous > EVENT_GAP:
            starts.append(hour)
        previous = hour
    return starts


def decimal_text(value: float) -> str:
    return f"{value:.1f}".replace(".", ",")


def vital_events(stay: pd.DataFrame, threshold: float = DEFAULT_Z_THRESHOLD) -> list[dict]:
    """Lista os eventos de uma internação, do mais antigo ao mais recente."""
    z_scores = stay_z_scores(stay)
    events = []
    for vital in VITALS:
        flags = flagged_hours(z_scores[vital], vital, threshold)
        for start in event_starts(flags):
            end = start
            for hour in flags.index[flags.to_numpy()]:
                if hour >= start and hour - end <= EVENT_GAP:
                    end = hour
            segment = z_scores.loc[start:end, vital]
            peak = segment.abs().idxmax()
            value = float(stay.loc[peak, vital])
            baseline = stay.loc[:start, vital].iloc[:-1].tail(24).median()
            events.append({
                "tipo": "sinal_vital",
                "sinal": vital,
                "descricao": f"{VITAL_LABELS[vital].capitalize()} {'subiu' if value > baseline else 'caiu'} para {decimal_text(value)} (referência do paciente: {decimal_text(baseline)})",
                "inicio": start.isoformat(),
                "fim": end.isoformat(),
                "valor": round(value, 1),
                "referencia_paciente": round(float(baseline), 1),
                "desvio_z": round(float(segment.loc[peak]), 1),
            })
    return sorted(events, key=lambda event: event["inicio"])


def rule_flags(stay: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({vital: rule(stay[vital]).fillna(False) for vital, rule in RULES.items()}, index=stay.index)


def load_prescriptions(mimic_dir: Path) -> pd.DataFrame:
    rx = pd.read_csv(
        Path(mimic_dir) / "hosp" / "prescriptions.csv.gz",
        usecols=["subject_id", "hadm_id", "starttime", "stoptime", "drug_type", "drug", "dose_val_rx", "dose_unit_rx", "route"],
        parse_dates=["starttime", "stoptime"],
    )
    return rx.sort_values(["hadm_id", "starttime"]).reset_index(drop=True)


def prescription_events(rx: pd.DataFrame, admission_start: pd.Timestamp | None = None) -> list[dict]:
    """Encontra mudanças fortes de dose e inícios tardios de medicamento de alta vigilância em uma internação."""
    main = rx[(rx["drug_type"] == "MAIN") & ~rx["drug"].str.contains(NON_DRUG_PATTERN)].copy()
    main["dose"] = pd.to_numeric(main["dose_val_rx"], errors="coerce")
    events = []

    for (drug, route, unit), orders in main.dropna(subset=["dose"]).groupby(["drug", "route", "dose_unit_rx"]):
        if SLIDING_SCALE_PATTERN.search(drug):
            continue
        orders = orders.sort_values("starttime")
        for previous, current in zip(orders.itertuples(), orders.iloc[1:].itertuples()):
            if previous.dose <= 0 or current.dose <= 0 or current.starttime == previous.starttime:
                continue
            ratio = current.dose / previous.dose
            if max(ratio, 1 / ratio) < DOSE_CHANGE_RATIO:
                continue
            large = max(ratio, 1 / ratio) >= LARGE_DOSE_CHANGE_RATIO
            events.append({
                "tipo": "mudanca_de_dose",
                "gravidade": "alta" if large else "moderada",
                "medicamento": drug,
                "descricao": f"{drug}: dose mudou de {previous.dose:g} para {current.dose:g} {unit} ({route})",
                "inicio": current.starttime.isoformat(),
                "razao_dose": round(float(ratio), 2),
            })

    first_order = admission_start if admission_start is not None else rx["starttime"].min()
    for drug, orders in main.groupby("drug"):
        first = orders["starttime"].min()
        if HIGH_ALERT_PATTERN.search(drug) and first - first_order >= LATE_START:
            events.append({
                "tipo": "inicio_alta_vigilancia",
                "gravidade": "moderada",
                "medicamento": drug,
                "descricao": f"{drug} iniciado após as primeiras 24 h da internação",
                "inicio": first.isoformat(),
            })
    return sorted(events, key=lambda event: event["inicio"])


class ClinicalTimelineService:
    """Serve a linha do tempo de cada internação a partir dos dados exportados do MIMIC-IV Demo."""

    def __init__(self, data_dir: str | Path = "sample_data/mimic_demo", threshold: float = DEFAULT_Z_THRESHOLD):
        self.data_dir = Path(data_dir)
        self.threshold = threshold
        self.vitals = pd.read_csv(self.data_dir / "vitais_por_hora.csv.gz", parse_dates=["hour"])
        self.prescriptions = pd.read_csv(self.data_dir / "prescricoes.csv.gz", parse_dates=["starttime", "stoptime", "admittime"])
        self._summaries: list[dict] | None = None

    def stays(self) -> list[int]:
        return sorted(self.vitals["stay_id"].unique().tolist())

    def timeline(self, stay_id: int) -> dict:
        rows = self.vitals[self.vitals["stay_id"] == stay_id]
        if rows.empty:
            raise KeyError(stay_id)
        stay = rows.set_index("hour").sort_index()
        hadm_id = int(rows["hadm_id"].iloc[0])
        admission_rx = self.prescriptions[self.prescriptions["hadm_id"] == hadm_id]
        window = (stay.index.min() - pd.Timedelta(hours=12), stay.index.max())
        rx_events = [
            event for event in prescription_events(admission_rx, admission_rx["admittime"].iloc[0] if len(admission_rx) else None)
            if window[0] <= pd.Timestamp(event["inicio"]) <= window[1]
        ]
        latest = stay[list(VITALS)].ffill().iloc[-1]
        series = stay[list(VITALS)].round(1).astype(object).where(stay[list(VITALS)].notna(), None)
        return {
            "stay_id": int(stay_id),
            "hadm_id": hadm_id,
            "horas": [hour.isoformat() for hour in stay.index],
            "series": {vital: series[vital].tolist() for vital in VITALS},
            "eventos_sinais_vitais": vital_events(stay, self.threshold),
            "eventos_prescricao": rx_events,
            "ultimos_valores": {vital: (None if pd.isna(latest[vital]) else round(float(latest[vital]), 1)) for vital in VITALS},
            "fonte": "MIMIC-IV Clinical Database Demo 2.2",
        }

    def replay(self, stay_id: int):
        """Percorre a internação hora a hora e entrega só o que já era conhecido naquele momento."""
        full = self.timeline(stay_id)
        stay = self.vitals[self.vitals["stay_id"] == stay_id].set_index("hour").sort_index()
        starts = {pd.Timestamp(event["inicio"]) for event in full["eventos_sinais_vitais"]}
        known_vitals: dict[tuple[str, str], dict] = {}
        pending_rx = sorted(full["eventos_prescricao"], key=lambda event: event["inicio"])
        known_rx: list[dict] = []
        for hour, row in stay.iterrows():
            new_events: list[dict] = []
            if hour in starts:
                for event in vital_events(stay.loc[:hour], self.threshold):
                    key = (event["sinal"], event["inicio"])
                    if key not in known_vitals:
                        new_events.append(event)
                    known_vitals[key] = event
            while pending_rx and pd.Timestamp(pending_rx[0]["inicio"]) <= hour:
                event = pending_rx.pop(0)
                known_rx.append(event)
                new_events.append(event)
            yield {
                "hora": hour.isoformat(),
                "valores": {vital: (None if pd.isna(row[vital]) else round(float(row[vital]), 1)) for vital in VITALS},
                "novos_eventos": new_events,
                "eventos_sinais_vitais": list(known_vitals.values()),
                "eventos_prescricao": list(known_rx),
            }

    def summaries(self) -> list[dict]:
        """Resume cada internação; o resultado fica guardado porque não muda durante a execução."""
        if self._summaries is None:
            items = []
            for stay_id in self.stays():
                timeline = self.timeline(stay_id)
                vitals = timeline["eventos_sinais_vitais"]
                rx = timeline["eventos_prescricao"]
                items.append({
                    "stay_id": stay_id,
                    "horas": len(timeline["horas"]),
                    "eventos_sinais_vitais": len(vitals),
                    "eventos_prescricao": len(rx),
                    "eventos_prescricao_alta": sum(event["gravidade"] == "alta" for event in rx),
                    "eventos_proximos": self._close_events(vitals, rx),
                })
            self._summaries = items
        return self._summaries

    @staticmethod
    def _close_events(vitals: list[dict], rx: list[dict], gap: pd.Timedelta = pd.Timedelta(hours=24)) -> int:
        return sum(
            abs(pd.Timestamp(v["inicio"]) - pd.Timestamp(p["inicio"])) <= gap and p["gravidade"] == "alta"
            for v in vitals for p in rx
        )

    def suggested_stay(self) -> int:
        """Escolhe para a demonstração uma internação curta em que sinais vitais e prescrição mudaram perto um do outro."""
        items = self.summaries()
        short = [item for item in items if MIN_DEMO_HOURS <= item["horas"] <= MAX_DEMO_HOURS] or items
        return max(short, key=lambda item: (item["eventos_proximos"], item["eventos_sinais_vitais"], item["stay_id"]))["stay_id"]
