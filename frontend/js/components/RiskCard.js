import { store } from "../store.js";
import { levelName, levelTone, pointsLabel, sourceName } from "../format.js";

export default {
  name: "RiskCard",
  setup() {
    const accent = (level) => ({ ok: "accent-ok", warn: "accent-warn", crit: "accent-crit" })[levelTone(level)] ?? "";
    return { store, levelName, levelTone, pointsLabel, sourceName, accent };
  },
  template: `
    <section class="card" :class="store.risk ? accent(store.risk.nivel) : ''" aria-labelledby="risk-title" aria-live="polite">
      <template v-if="store.risk">
        <div class="card-head">
          <div>
            <h2 id="risk-title">Risco {{ levelName(store.risk.nivel) }}</h2>
            <p class="small muted">{{ pointsLabel(store.risk.pontuacao) }} · cada ponto tem um motivo</p>
          </div>
          <span class="pill" :class="levelTone(store.risk.nivel)">{{ levelName(store.risk.nivel) }}</span>
        </div>
        <ul class="list" v-if="store.risk.motivos.length">
          <li v-for="(reason, index) in store.risk.motivos" :key="index" class="reason">
            <span class="reason-pts">+{{ reason.pontos }}</span>
            <span><strong>{{ sourceName(reason.fonte) }}</strong>: {{ reason.descricao }}</span>
          </li>
        </ul>
        <p class="small muted" v-else>Nenhum sinal ativo neste momento.</p>
        <p class="small muted" style="margin: 10px 0 0">{{ store.risk.observacao }}</p>
      </template>
      <template v-else>
        <h2 id="risk-title">Risco</h2>
        <p class="small muted">Inicie o monitoramento ou escolha uma internação.</p>
      </template>
    </section>`,
};
