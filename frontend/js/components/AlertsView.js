import { ref } from "../vue.js";
import { postJson } from "../api.js";
import { clearAlerts, store } from "../store.js";
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

    // Limpeza do histórico de teste
    const dialog = ref(null);
    const includeResults = ref(true);
    const clearing = ref(false);
    const clearError = ref("");
    const cleared = ref(null);

    function openDialog() {
      clearError.value = "";
      includeResults.value = true;
      dialog.value.showModal();
    }
    function closeDialog() {
      dialog.value.close();
    }
    async function doClear() {
      clearing.value = true;
      clearError.value = "";
      try {
        cleared.value = await clearAlerts(includeResults.value);
        closeDialog();
      } catch (e) {
        clearError.value = e.message;
      } finally {
        clearing.value = false;
      }
    }

    const accent = (level) => ({ ok: "accent-ok", warn: "accent-warn", crit: "accent-crit" })[levelTone(level)] ?? "";
    return {
      store, responsible, error, confirm, accent, formatDate, levelName, levelTone, pointsLabel, sourceName,
      dialog, includeResults, clearing, clearError, cleared, openDialog, closeDialog, doClear,
    };
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

      <section class="card accent-info" v-if="store.status?.limpeza_permitida" aria-labelledby="demo-title">
        <div class="card-head">
          <div>
            <h2 id="demo-title">Vai gravar uma demonstração?</h2>
            <p class="small muted">Você pode apagar o histórico de alertas de teste e começar do zero.</p>
          </div>
          <button class="btn" @click="openDialog" data-testid="open-clear">Limpar alertas de teste…</button>
        </div>
        <p class="small" style="margin: 0">
          Hoje há <strong data-testid="alert-count">{{ store.alerts.length }}</strong>
          {{ store.alerts.length === 1 ? 'alerta guardado' : 'alertas guardados' }} neste servidor.
          A limpeza <strong>não apaga</strong> os dados clínicos da base (sinais vitais e prescrições).
        </p>
        <p class="small" v-if="cleared" style="margin: 8px 0 0" role="status" data-testid="clear-result">
          <span class="pill ok">pronto</span>
          {{ cleared.alertas_apagados }} {{ cleared.alertas_apagados === 1 ? 'alerta apagado' : 'alertas apagados' }}{{ cleared.resultados_limpos ? ', e a marcha e a fala analisadas foram esquecidas' : '' }}.
        </p>
      </section>

      <dialog ref="dialog" class="dialog" aria-labelledby="clear-title" data-testid="clear-dialog">
        <h2 id="clear-title">Apagar os alertas de teste?</h2>
        <p style="margin: 10px 0 0">
          Serão apagados <strong>{{ store.alerts.length }}</strong> {{ store.alerts.length === 1 ? 'alerta' : 'alertas' }} do histórico.
          <strong>Esta ação não pode ser desfeita.</strong>
        </p>
        <label class="check" for="include-results">
          <input id="include-results" type="checkbox" v-model="includeResults" />
          <span>
            Esquecer também a marcha e a fala que foram analisadas nesta sessão
            <span class="small muted" style="display: block">Recomendado antes de gravar: o risco da internação volta ao estado inicial.</span>
          </span>
        </label>
        <p class="small muted" style="margin: 0">Os dados clínicos da base (sinais vitais e prescrições) não são afetados.</p>
        <p class="notice error" v-if="clearError" role="alert" style="margin-top: 12px">{{ clearError }}</p>
        <div class="dialog-actions">
          <button class="btn" @click="closeDialog" :disabled="clearing" data-testid="clear-cancel">Cancelar</button>
          <button class="btn danger" @click="doClear" :disabled="clearing" data-testid="clear-confirm">
            <span class="spinner" v-if="clearing"></span> Apagar alertas
          </button>
        </div>
      </dialog>

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
