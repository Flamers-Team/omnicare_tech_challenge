import { ref } from "../vue.js";
import { postJson } from "../api.js";
import { store } from "../store.js";
import { formatDate, levelName, levelTone, pointsLabel, sourceName } from "../format.js";

export default {
  name: "AlertsView",
  setup() {
    const responsible = ref("");
    const error = ref("");
    try { responsible.value = localStorage.getItem("omnicare-responsavel") ?? ""; } catch { /* sem armazenamento */ }

    async function confirm(alert) {
      error.value = "";
      try {
        await postJson(`/api/v1/alertas/${alert.id}/confirmar`, { responsavel: responsible.value });
        try { localStorage.setItem("omnicare-responsavel", responsible.value); } catch { /* sem armazenamento */ }
      } catch (e) {
        error.value = e.message;
      }
    }
    const accent = (level) => ({ ok: "accent-ok", warn: "accent-warn", crit: "accent-crit" })[levelTone(level)] ?? "";
    return { store, responsible, error, confirm, accent, formatDate, levelName, levelTone, pointsLabel, sourceName };
  },
  template: `
    <div class="stack">
      <section class="card">
        <div class="card-head">
          <div>
            <h2>Histórico de alertas</h2>
            <p class="small muted">Os alertas chegam em tempo real. Uma pessoa da equipe confirma que viu.</p>
          </div>
        </div>
        <div class="field" style="max-width: 360px">
          <label for="responsible">Seu nome, para registrar a confirmação</label>
          <input id="responsible" class="input" type="text" v-model="responsible" maxlength="80" autocomplete="name" />
        </div>
        <p class="notice error" v-if="error" role="alert" style="margin-top: 10px">{{ error }}</p>
        <p class="small muted" style="margin: 10px 0 0">As datas do MIMIC-IV são deslocadas pela própria base para proteger os pacientes.</p>
      </section>

      <section class="card" :class="accent(alert.nivel)" v-for="alert in store.alerts" :key="alert.id" data-testid="alert-card">
        <div class="card-head">
          <div>
            <h3>{{ alert.paciente }}</h3>
            <p class="small muted">{{ formatDate(alert.ocorrido_em) }} · origem: {{ sourceName(alert.origem) }}</p>
          </div>
          <span class="pill" :class="levelTone(alert.nivel)">risco {{ levelName(alert.nivel) }} · {{ pointsLabel(alert.pontuacao) }}</span>
        </div>
        <ul class="list">
          <li v-for="(reason, index) in alert.motivos" :key="index" class="reason"><span class="reason-pts">+{{ reason.pontos }}</span><span>{{ reason.descricao }}</span></li>
        </ul>
        <p class="small" v-if="alert.confirmado_por" style="margin: 12px 0 0"><span class="pill ok">confirmado</span> por {{ alert.confirmado_por }} em {{ formatDate(alert.confirmado_em) }}.</p>
        <button v-else class="btn sm" style="margin-top: 12px" @click="confirm(alert)" :disabled="responsible.trim().length < 2" data-testid="confirm-alert">Confirmar que vi</button>
      </section>
      <p class="empty card" v-if="!store.alerts.length">Nenhum alerta ainda. Inicie o monitoramento de uma internação.</p>
    </div>`,
};
