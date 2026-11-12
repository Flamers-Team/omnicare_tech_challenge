export const num = (value, digits = 2) => Number(value).toLocaleString("pt-BR", { maximumFractionDigits: digits });
export const pointsLabel = (points) => (points === 1 ? "1 ponto" : `${points} pontos`);
export const levelName = (level) => ({ baixo: "baixo", atencao: "de atenção", alto: "alto" })[level] ?? level;
export const levelTone = (level) => ({ baixo: "ok", atencao: "warn", alto: "crit" })[level] ?? "";
export const formatDate = (iso) => (iso ? iso.replace("T", " ").slice(0, 16) : "");

const SOURCES = {
  sinais_vitais: "Sinais vitais", prescricao: "Prescrição", marcha: "Marcha", fala: "Fala",
  voz: "Voz", monitor: "Monitoramento", cirurgia: "Cirurgia",
};
export const sourceName = (source) => SOURCES[source] ?? source;

export function eventType(event) {
  if (event.tipo === "sinal_vital") return "Sinal vital";
  return event.gravidade === "alta" ? "Prescrição, gravidade alta" : "Prescrição";
}

export const VITALS = [
  { key: "HR", name: "Frequência cardíaca", short: "FC", unit: "bpm" },
  { key: "RR", name: "Frequência respiratória", short: "FR", unit: "irpm" },
  { key: "SPO2", name: "Oxigenação", short: "SpO₂", unit: "%" },
  { key: "SBP", name: "Pressão sistólica", short: "PAS", unit: "mmHg" },
  { key: "TEMP", name: "Temperatura", short: "Temp", unit: "°C" },
];
