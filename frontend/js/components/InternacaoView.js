import { computed, ref } from "../vue.js";
import { loadTimeline, startMonitor, stopMonitor, store } from "../store.js";
import { levelName, levelTone, pointsLabel } from "../format.js";
import Icon from "./Icon.js";
import RiskCard from "./RiskCard.js";
import VitalsPanel from "./VitalsPanel.js";
import EventsFeed from "./EventsFeed.js";
import GaitPanel from "./GaitPanel.js";
import VoicePanel from "./VoicePanel.js";

export default {
  name: "InternacaoView",
  components: { Icon, RiskCard, VitalsPanel, EventsFeed, GaitPanel, VoicePanel },
  setup() {
    const speed = ref(0.25);
    const openAlerts = computed(() => store.alerts.filter((alert) => !alert.confirmado_por).length);
    const eventCount = computed(() => store.vitalEvents.length + store.rxEvents.length);
    return { store, speed, openAlerts, eventCount, loadTimeline, startMonitor, stopMonitor, levelName, levelTone, pointsLabel };
  },
  template: `
    <div class="stack">
      <section class="card">
        <div class="row">
          <div class="field">
            <label for="stay">Internação (MIMIC-IV Demo)</label>
            <select id="stay" class="select" v-model.number="store.stayId" @change="loadTimeline" :disabled="store.running">
              <option v-for="item in store.stays" :key="item.stay_id" :value="item.stay_id">
                {{ item.stay_id }}: {{ item.horas }} h, {{ item.eventos_sinais_vitais }} sinais, {{ item.eventos_prescricao }} prescrições{{ item.stay_id === store.suggested ? ' (sugerida)' : '' }}
              </option>
            </select>
          </div>
          <div class="field">
            <label for="speed">Velocidade do replay</label>
            <select id="speed" class="select" v-model.number="speed" :disabled="store.running">
              <option :value="0.6">1 hora a cada 0,6 s</option>
              <option :value="0.25">1 hora a cada 0,25 s</option>
              <option :value="0.08">1 hora a cada 0,08 s</option>
            </select>
          </div>
          <button v-if="!store.running" class="btn primary" @click="startMonitor(speed)" :disabled="!store.stayId" data-testid="start-monitor">
            <Icon name="play" :size="16" /> Iniciar monitoramento
          </button>
          <button v-else class="btn danger" @click="stopMonitor" data-testid="stop-monitor"><Icon name="stop" :size="16" /> Parar</button>
          <span class="small muted" v-if="store.running" role="status"><span class="spinner"></span> Hora {{ store.hours.length }}</span>
        </div>
      </section>

      <div class="kpis">
        <div class="card kpi">
          <div class="kpi-label">Nível de risco</div>
          <div class="kpi-value" :class="store.risk ? levelTone(store.risk.nivel) : ''" data-testid="kpi-level">{{ store.risk ? levelName(store.risk.nivel) : '–' }}</div>
          <div class="kpi-sub">{{ store.risk ? pointsLabel(store.risk.pontuacao) : 'sem dados' }}</div>
        </div>
        <div class="card kpi">
          <div class="kpi-label">Eventos</div>
          <div class="kpi-value">{{ eventCount }}</div>
          <div class="kpi-sub">{{ store.vitalEvents.length }} sinais · {{ store.rxEvents.length }} prescrições</div>
        </div>
        <div class="card kpi">
          <div class="kpi-label">Horas monitoradas</div>
          <div class="kpi-value">{{ store.hours.length }}</div>
          <div class="kpi-sub">{{ store.running ? 'em andamento' : 'internação carregada' }}</div>
        </div>
        <div class="card kpi">
          <div class="kpi-label">Alertas abertos</div>
          <div class="kpi-value" :class="openAlerts ? 'crit' : ''">{{ openAlerts }}</div>
          <div class="kpi-sub">aguardando confirmação</div>
        </div>
      </div>

      <RiskCard />

      <div class="grid-main">
        <VitalsPanel />
        <EventsFeed />
      </div>

      <div class="grid-two">
        <GaitPanel />
        <VoicePanel />
      </div>

      <section class="card" v-if="store.report" aria-labelledby="report-title">
        <h2 id="report-title" style="margin-bottom: 8px">Laudo</h2>
        <p class="small" style="margin: 0" data-testid="report">{{ store.report }}</p>
      </section>
    </div>`,
};
