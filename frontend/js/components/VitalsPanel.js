import { computed, nextTick, onBeforeUnmount, onMounted, watch } from "../vue.js";
import { store } from "../store.js";
import { VITALS, num } from "../format.js";
import { createVitalChart, updateVitalChart } from "../charts.js";

export default {
  name: "VitalsPanel",
  setup() {
    const charts = {};

    function build() {
      VITALS.forEach((vital) => {
        charts[vital.key]?.destroy();
        const canvas = document.getElementById(`chart-${vital.key}`);
        if (canvas) charts[vital.key] = createVitalChart(canvas, vital);
      });
      draw();
    }

    function draw() {
      VITALS.forEach((vital) => {
        const chart = charts[vital.key];
        if (!chart) return;
        const marks = new Set(store.vitalEvents.filter((event) => event.sinal === vital.key).map((event) => event.inicio));
        updateVitalChart(chart, store.hours, store.series[vital.key] || [], marks);
      });
    }

    function lastValue(key) {
      const values = store.series[key] || [];
      for (let i = values.length - 1; i >= 0; i -= 1) {
        if (values[i] !== null && values[i] !== undefined) return `${num(values[i])} ${VITALS.find((v) => v.key === key).unit}`;
      }
      return "–";
    }

    const lastRows = computed(() => {
      const first = Math.max(0, store.hours.length - 24);
      const rows = [];
      for (let i = store.hours.length - 1; i >= first; i -= 1) {
        const values = {};
        VITALS.forEach((vital) => { values[vital.key] = store.series[vital.key]?.[i]; });
        rows.push({ index: i + 1, values });
      }
      return rows;
    });

    onMounted(async () => { await nextTick(); build(); });
    onBeforeUnmount(() => Object.values(charts).forEach((chart) => chart.destroy()));
    watch(() => [store.hours.length, store.vitalEvents.length], draw);
    watch(() => store.hours, draw);
    // Troca de tema: as cores dos gráficos vêm do CSS e precisam ser lidas de novo.
    watch(() => store.theme, async () => { await nextTick(); build(); });

    return { VITALS, store, lastValue, lastRows, num };
  },
  template: `
    <section class="card" aria-labelledby="vitals-title">
      <div class="card-head">
        <div>
          <h2 id="vitals-title">Sinais vitais, hora a hora</h2>
          <p class="small muted">Os pontos vermelhos marcam desvios sustentados em relação às 24 horas anteriores do próprio paciente. Falhas na linha são períodos de mais de 6 horas sem medição.</p>
        </div>
      </div>
      <div class="chart-box" v-for="vital in VITALS" :key="vital.key">
        <div class="chart-head"><span>{{ vital.name }} ({{ vital.unit }})</span><strong>{{ lastValue(vital.key) }}</strong></div>
        <div class="chart-wrap"><canvas :id="'chart-' + vital.key" :aria-label="vital.name" role="img"></canvas></div>
      </div>
      <details>
        <summary>Ver as últimas 24 horas em tabela</summary>
        <div class="table-scroll">
          <table>
            <thead><tr><th>Hora</th><th v-for="vital in VITALS" :key="vital.key">{{ vital.short }}</th></tr></thead>
            <tbody>
              <tr v-for="row in lastRows" :key="row.index">
                <td>{{ row.index }}</td>
                <td v-for="vital in VITALS" :key="vital.key">{{ row.values[vital.key] === null || row.values[vital.key] === undefined ? '–' : num(row.values[vital.key]) }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </details>
    </section>`,
};
