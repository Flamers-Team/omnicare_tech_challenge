import { reactive } from "./vue.js";
import { api } from "./api.js";
import { VITALS } from "./format.js";

/** Estado compartilhado entre as telas. */
export const store = reactive({
  tab: "internacao",
  status: null,
  loadError: "",
  // internação
  stays: [],
  suggested: null,
  stayId: null,
  running: false,
  hours: [],
  series: {},
  vitalEvents: [],
  rxEvents: [],
  risk: null,
  report: "",
  gait: null,
  voice: null,
  voiceBase: null,
  // cirurgia
  surgeryInfo: null,
  surgery: null,
  surgeryVideo: null,
  // alertas
  alerts: [],
  toasts: [],
  // aparência
  theme: "auto",
});

export function emptySeries() {
  return Object.fromEntries(VITALS.map((vital) => [vital.key, []]));
}

export function applyContext(payload) {
  store.risk = payload.risco;
  store.report = payload.laudo;
}

let monitorSource = null;

export async function loadTimeline() {
  store.loadError = "";
  stopMonitor();
  try {
    const data = await api(`/api/v1/pacientes/${store.stayId}/linha-do-tempo`);
    store.hours = data.horas;
    store.series = data.series;
    store.vitalEvents = data.eventos_sinais_vitais;
    store.rxEvents = data.eventos_prescricao;
    store.gait = data.marcha;
    store.voice = data.fala;
    applyContext(data);
  } catch (error) {
    store.loadError = error.message;
  }
}

export function startMonitor(speed) {
  store.hours = [];
  store.series = emptySeries();
  store.vitalEvents = [];
  store.rxEvents = [];
  store.risk = null;
  store.report = "";
  store.running = true;
  monitorSource = new EventSource(`/api/v1/pacientes/${store.stayId}/monitor?intervalo=${speed}`);
  monitorSource.onmessage = (message) => {
    const step = JSON.parse(message.data);
    store.hours.push(step.hora);
    VITALS.forEach((vital) => store.series[vital.key].push(step.valores[vital.key]));
    step.novos_eventos.forEach((event) => (event.tipo === "sinal_vital" ? store.vitalEvents : store.rxEvents).push(event));
    store.risk = step.risco;
  };
  monitorSource.addEventListener("fim", async () => {
    stopMonitor();
    await loadTimeline();
  });
  monitorSource.onerror = () => {
    store.loadError = "A conexão com o monitoramento foi perdida.";
    stopMonitor();
  };
}

export function stopMonitor() {
  monitorSource?.close();
  monitorSource = null;
  store.running = false;
}

/** Recebe os alertas em tempo real e avisa quando chega um novo. */
export function connectAlerts() {
  let first = true;
  const source = new EventSource("/api/v1/alertas/stream");
  source.onmessage = (message) => {
    const alerts = JSON.parse(message.data).alertas;
    if (!first) {
      const known = new Set(store.alerts.map((alert) => alert.id));
      alerts.filter((alert) => !known.has(alert.id)).forEach(pushToast);
    }
    first = false;
    store.alerts = alerts;
  };
}

function pushToast(alert) {
  // Mostra o motivo que mais pesou no risco.
  const main = [...(alert.motivos ?? [])].sort((a, b) => b.pontos - a.pontos)[0];
  const toast = { id: alert.id, nivel: alert.nivel, titulo: alert.paciente, texto: main?.descricao ?? "" };
  store.toasts.push(toast);
  setTimeout(() => dismissToast(toast.id), 9000);
}

export function dismissToast(id) {
  const index = store.toasts.findIndex((toast) => toast.id === id);
  if (index >= 0) store.toasts.splice(index, 1);
}

export async function boot() {
  try {
    store.status = await api("/api/v1/status");
    store.surgeryInfo = await api("/api/v1/cirurgia");
    store.surgeryVideo = store.surgeryInfo.videos_base[0] ?? null;
    const patients = await api("/api/v1/pacientes");
    store.stays = patients.internacoes;
    store.suggested = patients.sugerida;
    store.stayId = patients.sugerida;
    await loadTimeline();
  } catch (error) {
    store.loadError = `Não foi possível falar com o servidor. ${error.message}`;
  }
  connectAlerts();
}
