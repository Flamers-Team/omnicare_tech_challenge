#!/usr/bin/env python3
"""Avalia as anomalias em sinais vitais e prescrições com o MIMIC-IV Demo e exporta os dados da demo.

Exemplo:
  python training/analyze_mimic_demo.py --mimic-dir "C:\\caminho\\mimic-iv-demo-2.2"

A base não tem rótulos de anomalia. Por isso a avaliação mede:
  - carga de alertas: eventos por paciente por dia nos dados reais;
  - detecção de alterações simuladas, descontando o que já dispararia por acaso.

Saídas:
  backend/sample_data/mimic_demo/   dados usados pela demo
  training/resultados/clinico_metricas.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GroupKFold

PROJECT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_DIR / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from services.alert_service import LEVEL_ORDER, combined_risk  # noqa: E402
from services.clinical_anomaly_service import (  # noqa: E402
    DEFAULT_Z_THRESHOLD,
    NON_DRUG_PATTERN,
    SLIDING_SCALE_PATTERN,
    VITALS,
    vital_events,
    event_starts,
    flagged_hours,
    load_hourly_vitals,
    load_prescriptions,
    prescription_events,
    rolling_z,
    rule_flags,
    stay_z_scores,
)

SEED = 42
MIN_HOURS = 36
THRESHOLDS = (3.0, 4.0, 5.0, 6.0)
RAMP_HOURS = 2
HOLD_HOURS = 6
DETECTION_WINDOW = pd.Timedelta(hours=HOLD_HOURS)
SIMULATED_CHANGE = {
    "moderada": {"HR": 25, "RR": 8, "SPO2": -6, "SBP": -25, "TEMP": 1.2},
    "forte": {"HR": 40, "RR": 14, "SPO2": -10, "SBP": -40, "TEMP": 2.0},
}
IF_QUANTILES = (0.90, 0.95, 0.97, 0.99)
MAX_EVENTS_PER_PATIENT_DAY = 1.0
MAX_CHANCE_RATE = 0.05
PRESCRIPTION_DOSE_ERROR = 10
DOSE_ERROR_CASES = 300


def stay_tables(hourly: pd.DataFrame) -> dict[int, pd.DataFrame]:
    return {int(stay_id): rows.set_index("hour").sort_index() for stay_id, rows in hourly.groupby("stay_id")}


def inject(series: pd.Series, start: pd.Timestamp, change: float) -> pd.Series:
    """Soma uma alteração que cresce por 2 h e se mantém por 6 h."""
    out = series.copy()
    elapsed = (out.index - start) / pd.Timedelta(hours=1)
    ramp = np.clip((elapsed + 1) / RAMP_HOURS, 0, 1)
    active = (elapsed >= 0) & (elapsed < HOLD_HOURS)
    out[active] = out[active] + change * ramp[active]
    return out


def choose_injections(stays: dict[int, pd.DataFrame], rng: np.random.Generator) -> list[dict]:
    plan = []
    for stay_id, stay in stays.items():
        for vital in VITALS:
            valid = stay[vital].dropna()
            if len(valid) < MIN_HOURS:
                continue
            first, last = valid.index[0] + pd.Timedelta(hours=24), valid.index[-1] - pd.Timedelta(hours=HOLD_HOURS)
            candidates = valid.index[(valid.index >= first) & (valid.index <= last)]
            if len(candidates):
                plan.append({"stay": stay_id, "vital": vital, "start": candidates[rng.integers(len(candidates))]})
    return plan


def started_in_window(flags: pd.Series, start: pd.Timestamp) -> bool:
    return any(start <= hour < start + DETECTION_WINDOW for hour in event_starts(flags))


def patient_days(stays: dict[int, pd.DataFrame]) -> float:
    return sum((s.index.max() - s.index.min()) / pd.Timedelta(days=1) for s in stays.values())


def evaluate_z(stays: dict[int, pd.DataFrame], plan: list[dict], threshold: float) -> dict:
    events = 0
    for stay in stays.values():
        z_scores = stay_z_scores(stay)
        events += sum(len(event_starts(flagged_hours(z_scores[v], v, threshold))) for v in VITALS)

    def flags_for(series: pd.Series, vital: str) -> pd.Series:
        return flagged_hours(rolling_z(series, vital), vital, threshold)

    return summarize(stays, plan, events, flags_for)


def evaluate_rules(stays: dict[int, pd.DataFrame], plan: list[dict]) -> dict:
    events = sum(
        len(event_starts(column))
        for stay in stays.values()
        for _, column in rule_flags(stay).items()
    )
    rule_plan = [item for item in plan if item["vital"] != "TEMP"]  # as regras herdadas não olham a temperatura

    def flags_for(series: pd.Series, vital: str) -> pd.Series:
        frame = pd.DataFrame({v: (series if v == vital else pd.Series(np.nan, index=series.index)) for v in VITALS})
        return rule_flags(frame)[vital]

    return summarize(stays, rule_plan, events, flags_for)


def summarize(stays, plan, events, flags_for) -> dict:
    result = {"eventos": int(events), "eventos_por_paciente_dia": round(events / patient_days(stays), 2), "deteccao": {}}
    baseline_hits = 0
    for item in plan:
        baseline_hits += started_in_window(flags_for(stays[item["stay"]][item["vital"]], item["vital"]), item["start"])
    result["disparo_por_acaso"] = round(baseline_hits / max(len(plan), 1), 3)
    for severity, changes in SIMULATED_CHANGE.items():
        hits = 0
        for item in plan:
            series = inject(stays[item["stay"]][item["vital"]], item["start"], changes[item["vital"]])
            hits += started_in_window(flags_for(series, item["vital"]), item["start"])
        result["deteccao"][severity] = round(hits / max(len(plan), 1), 3)
    result["casos_simulados"] = len(plan)
    return result


def if_features(stay: pd.DataFrame, z_scores: pd.DataFrame) -> np.ndarray:
    return np.nan_to_num(np.column_stack([z_scores[v].clip(-15, 15) for v in VITALS]))


def evaluate_isolation_forest(stays: dict[int, pd.DataFrame], plan: list[dict], subjects: dict[int, int], quantile: float) -> dict:
    ids = list(stays)
    z_by_stay = {i: stay_z_scores(stays[i]) for i in ids}
    groups = np.array([subjects[i] for i in ids])
    events, by_item = 0, {severity: 0 for severity in SIMULATED_CHANGE}
    chance = 0
    for train_idx, test_idx in GroupKFold(n_splits=5).split(ids, groups=groups):
        train_x = np.vstack([if_features(stays[ids[i]], z_by_stay[ids[i]]) for i in train_idx])
        model = IsolationForest(n_estimators=100, random_state=SEED).fit(train_x)
        threshold = np.quantile(-model.score_samples(train_x), quantile)

        def flags_of(stay_id: int, z_scores: pd.DataFrame) -> pd.Series:
            scores = -model.score_samples(if_features(stays[stay_id], z_scores))
            over = pd.Series(scores >= threshold, index=stays[stay_id].index)
            return (over & over.shift(1, fill_value=False)).astype(bool)

        test_ids = {ids[i] for i in test_idx}
        for stay_id in test_ids:
            events += len(event_starts(flags_of(stay_id, z_by_stay[stay_id])))
        for item in plan:
            if item["stay"] not in test_ids:
                continue
            chance += started_in_window(flags_of(item["stay"], z_by_stay[item["stay"]]), item["start"])
            for severity, changes in SIMULATED_CHANGE.items():
                z_scores = z_by_stay[item["stay"]].copy()
                series = inject(stays[item["stay"]][item["vital"]], item["start"], changes[item["vital"]])
                z_scores[item["vital"]] = rolling_z(series, item["vital"])
                by_item[severity] += started_in_window(flags_of(item["stay"], z_scores), item["start"])
    return {
        "quantil": quantile,
        "eventos": int(events),
        "eventos_por_paciente_dia": round(events / patient_days(stays), 2),
        "disparo_por_acaso": round(chance / max(len(plan), 1), 3),
        "deteccao": {severity: round(hits / max(len(plan), 1), 3) for severity, hits in by_item.items()},
        "casos_simulados": len(plan),
    }


def evaluate_prescriptions(rx: pd.DataFrame, admissions: pd.DataFrame) -> dict:
    starts = admissions.set_index("hadm_id")["admittime"]
    days = ((admissions["dischtime"] - admissions["admittime"]) / pd.Timedelta(days=1)).sum()
    counts: dict[str, int] = {}
    high = 0
    total = 0
    for hadm_id, orders in rx.groupby("hadm_id"):
        for event in prescription_events(orders, starts.get(hadm_id)):
            counts[event["tipo"]] = counts.get(event["tipo"], 0) + 1
            high += event["gravidade"] == "alta"
            total += 1

    # Teste: uma ordem do mesmo medicamento recebe uma dose 10 vezes maior.
    # Só conta como detectado o evento de gravidade alta que não existia antes da alteração.
    main = rx[(rx["drug_type"] == "MAIN") & ~rx["drug"].str.contains(NON_DRUG_PATTERN)].copy()
    main["dose"] = pd.to_numeric(main["dose_val_rx"], errors="coerce")
    main = main[main["dose"] > 0]
    main = main[~main["drug"].str.contains(SLIDING_SCALE_PATTERN)]
    repeated = main.groupby(["hadm_id", "drug", "route", "dose_unit_rx"]).filter(lambda orders: orders["starttime"].nunique() >= 2)
    sample = repeated.sample(n=min(DOSE_ERROR_CASES, len(repeated)), random_state=SEED)
    detected = 0
    for row in sample.itertuples():
        orders = rx[rx["hadm_id"] == row.hadm_id].copy()
        before = {event["descricao"] for event in prescription_events(orders, starts.get(row.hadm_id))}
        orders.loc[row.Index, "dose_val_rx"] = str(row.dose * PRESCRIPTION_DOSE_ERROR)
        detected += any(
            event["gravidade"] == "alta" and event["medicamento"] == row.drug and event["descricao"] not in before
            for event in prescription_events(orders, starts.get(row.hadm_id))
        )
    return {
        "internacoes": int(rx["hadm_id"].nunique()),
        "dias_de_internacao": round(float(days), 1),
        "eventos": counts,
        "eventos_de_gravidade_alta": int(high),
        "eventos_por_internacao_dia": round(total / float(days), 2),
        "eventos_alta_por_internacao_dia": round(high / float(days), 3),
        "dose_10x_simulada": {"casos": int(len(sample)), "detectados": int(detected), "taxa": round(detected / max(len(sample), 1), 3)},
    }


def choose_threshold(results: dict[float, dict]) -> float:
    """Maior detecção entre os limiares com carga de alertas aceitável, olhando só a metade de ajuste."""
    allowed = {
        threshold: result for threshold, result in results.items()
        if result["eventos_por_paciente_dia"] <= MAX_EVENTS_PER_PATIENT_DAY and result["disparo_por_acaso"] <= MAX_CHANCE_RATE
    }
    if not allowed:
        raise RuntimeError("Nenhum limiar respeitou a carga de alertas definida.")
    return max(allowed, key=lambda threshold: allowed[threshold]["deteccao"]["moderada"])

def combined_alert_load(stays: dict[int, pd.DataFrame], rx: pd.DataFrame, admissions: pd.DataFrame, hadm_by_stay: dict[int, int], threshold: float) -> dict:
    """Conta quantos avisos a equipe receberia só com sinais vitais e prescrições, como faz o monitoramento."""
    starts = admissions.set_index("hadm_id")["admittime"]
    rises = {"atencao": 0, "alto": 0}
    hours_at_level = {"baixo": 0, "atencao": 0, "alto": 0}
    for stay_id, stay in stays.items():
        hadm_id = hadm_by_stay[stay_id]
        vitals = vital_events(stay, threshold)
        orders = rx[rx["hadm_id"] == hadm_id]
        rx_events = prescription_events(orders, starts.get(hadm_id)) if len(orders) else []
        previous = "baixo"
        for hour in stay.index:
            level = combined_risk(vitals, rx_events, hour)["nivel"]
            hours_at_level[level] += 1
            if LEVEL_ORDER[level] > LEVEL_ORDER[previous]:
                rises[level] += 1
            previous = level
    days = patient_days(stays)
    total_hours = sum(hours_at_level.values())
    return {
        "avisos_por_paciente_dia": {level: round(count / days, 2) for level, count in rises.items()},
        "fracao_das_horas": {level: round(count / total_hours, 3) for level, count in hours_at_level.items()},
    }


def export_demo_data(hourly: pd.DataFrame, rx: pd.DataFrame, admissions: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    hourly.drop(columns="subject_id").to_csv(out_dir / "vitais_por_hora.csv.gz", index=False)
    kept = rx[rx["hadm_id"].isin(hourly["hadm_id"].unique())].drop(columns="subject_id")
    kept.merge(admissions[["hadm_id", "admittime"]], on="hadm_id", how="left").to_csv(out_dir / "prescricoes.csv.gz", index=False)
    (out_dir / "LEIA-ME.txt").write_text(
        "Dados derivados do MIMIC-IV Clinical Database Demo 2.2, https://physionet.org/content/mimic-iv-demo/2.2/.\n"
        "Licença: Open Database License (ODbL). Estes arquivos derivados seguem a mesma licença.\n"
        "Gerados por training/analyze_mimic_demo.py: medições agrupadas por hora, erros de sensor removidos e subject_id excluído.\n",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> None:
    mimic_dir = args.mimic_dir.resolve()
    hourly = load_hourly_vitals(mimic_dir)
    rx = load_prescriptions(mimic_dir)
    admissions = pd.read_csv(mimic_dir / "hosp" / "admissions.csv.gz", usecols=["hadm_id", "admittime", "dischtime"], parse_dates=["admittime", "dischtime"])
    admissions = admissions[admissions["hadm_id"].isin(rx["hadm_id"].unique())]
    print(f"Internações em UTI: {hourly['stay_id'].nunique()} | pacientes: {hourly['subject_id'].nunique()} | horas: {len(hourly)}")

    rng = np.random.default_rng(SEED)
    stays = stay_tables(hourly)
    subjects = {int(k): int(v) for k, v in hourly.groupby("stay_id")["subject_id"].first().items()}
    people = np.array(sorted(set(subjects.values())))
    rng.shuffle(people)
    tuning_people = set(people[: len(people) // 2].tolist())
    tuning = {k: v for k, v in stays.items() if subjects[k] in tuning_people}
    evaluation = {k: v for k, v in stays.items() if subjects[k] not in tuning_people}
    tuning_plan = choose_injections(tuning, rng)
    evaluation_plan = choose_injections(evaluation, rng)

    print("\nAjuste do limiar (metade dos pacientes):")
    tuning_results = {}
    for threshold in THRESHOLDS:
        tuning_results[threshold] = evaluate_z(tuning, tuning_plan, threshold)
        r = tuning_results[threshold]
        print(f"  z >= {threshold}: {r['eventos_por_paciente_dia']} eventos por paciente/dia | moderada {r['deteccao']['moderada']} | forte {r['deteccao']['forte']} | acaso {r['disparo_por_acaso']}")

    chosen = choose_threshold(tuning_results)
    if chosen != DEFAULT_Z_THRESHOLD:
        raise RuntimeError(f"Atualize DEFAULT_Z_THRESHOLD no serviço para {chosen}.")

    print("\nIsolation Forest na metade de ajuste:")
    if_tuning = {quantile: evaluate_isolation_forest(tuning, tuning_plan, subjects, quantile) for quantile in IF_QUANTILES}
    for quantile, r in if_tuning.items():
        print(f"  quantil {quantile}: {r['eventos_por_paciente_dia']} eventos por paciente/dia | moderada {r['deteccao']['moderada']} | forte {r['deteccao']['forte']} | acaso {r['disparo_por_acaso']}")
    target_load = tuning_results[chosen]["eventos_por_paciente_dia"]
    if_quantile = min(if_tuning, key=lambda quantile: abs(if_tuning[quantile]["eventos_por_paciente_dia"] - target_load))

    print(f"\nTeste na outra metade (z >= {chosen}; Isolation Forest no quantil {if_quantile}, carga de alertas parecida):")
    comparison = {
        "z_do_proprio_paciente": evaluate_z(evaluation, evaluation_plan, chosen),
        "regras_fixas": evaluate_rules(evaluation, evaluation_plan),
        "isolation_forest": evaluate_isolation_forest(evaluation, evaluation_plan, subjects, if_quantile),
    }
    for name, r in comparison.items():
        print(f"  {name}: {r['eventos_por_paciente_dia']} eventos por paciente/dia | moderada {r['deteccao']['moderada']} | forte {r['deteccao']['forte']} | acaso {r['disparo_por_acaso']} | casos {r['casos_simulados']}")

    prescriptions = evaluate_prescriptions(rx, admissions)
    print("\nPrescrições:", json.dumps(prescriptions, ensure_ascii=False))

    hadm_by_stay = {int(k): int(v) for k, v in hourly.groupby("stay_id")["hadm_id"].first().items()}
    alert_load = combined_alert_load(evaluation, rx, admissions, hadm_by_stay, chosen)
    print("\nAlerta combinado, sem marcha nem fala:", json.dumps(alert_load, ensure_ascii=False))

    export_demo_data(hourly, rx, admissions, args.demo_out)
    metrics = {
        "gerado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "base": "MIMIC-IV Clinical Database Demo 2.2",
        "amostra": {
            "pacientes": int(hourly["subject_id"].nunique()),
            "internacoes_uti": int(hourly["stay_id"].nunique()),
            "horas_com_medida": int(len(hourly)),
            "pacientes_para_ajuste": len(tuning_people),
            "pacientes_para_teste": int(len(people) - len(tuning_people)),
        },
        "limiar_escolhido": chosen,
        "criterio_do_limiar": f"maior detecção com até {MAX_EVENTS_PER_PATIENT_DAY} evento por paciente/dia e até {MAX_CHANCE_RATE:.0%} de disparo por acaso, na metade de ajuste",
        "isolation_forest_ajuste": {str(k): v for k, v in if_tuning.items()},
        "ajuste_limiar": {str(k): v for k, v in tuning_results.items()},
        "sinais_vitais_teste": comparison,
        "prescricoes": prescriptions,
        "alerta_combinado_teste": alert_load,
        "limitacoes": [
            "A base não tem rótulos de anomalia; não é possível medir acerto clínico real.",
            "As alterações são simuladas sobre dados reais; a detecção mostrada é um teste de engenharia.",
            "Pacientes de UTI em uma amostra de 100 pessoas de um único hospital dos EUA.",
            "Muitas mudanças de dose são condutas corretas; o evento pede revisão, não indica erro.",
            "A lista de medicamentos de alta vigilância é simplificada e não foi validada clinicamente.",
            "Apoio à decisão, não diagnóstico.",
        ],
    }
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nMétricas: {args.metrics_out}\nDados da demo: {args.demo_out}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Avalia anomalias clínicas com o MIMIC-IV Demo.")
    parser.add_argument("--mimic-dir", type=Path, required=True)
    parser.add_argument("--demo-out", type=Path, default=BACKEND_DIR / "sample_data" / "mimic_demo")
    parser.add_argument("--metrics-out", type=Path, default=PROJECT_DIR / "training" / "resultados" / "clinico_metricas.json")
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    sys.exit(main())
