import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "../vue.js";
import { api, postFile } from "../api.js";
import { store } from "../store.js";
import { num } from "../format.js";
import { createSurgeryChart } from "../charts.js";
import SurgeryAnatomy from "./SurgeryAnatomy.js";

export default {
  name: "SurgeryView",
  components: { SurgeryAnatomy },
  setup() {
    const busy = ref(false);
    const error = ref("");
    const clip = ref("");
    const charts = {};

    const result = computed(() => store.surgery?.resultado);
    const names = computed(() => Object.keys(result.value?.series ?? {}));
    const level = computed(() => {
      if (!result.value?.avaliada) return "";
      return result.value.desvio ? "accent-warn" : "accent-ok";
    });
    const headline = computed(() => {
      if (!result.value?.avaliada) return "Não avaliada";
      return result.value.desvio ? "Indício de desvio: revisar o vídeo" : "Sem indício de desvio nos critérios avaliados";
    });
    const agrees = computed(() => result.value.desvio === !result.value.especialistas.cvs_atingido);
    const criterion = (name) => result.value.criterios.find((item) => item.criterio === name);

    async function draw() {
      Object.values(charts).forEach((chart) => chart.destroy());
      Object.keys(charts).forEach((key) => delete charts[key]);
      if (!result.value?.avaliada) return;
      await nextTick();
      names.value.forEach((name) => {
        const canvas = document.getElementById(`surgery-chart-${name}`);
        if (canvas) charts[name] = createSurgeryChart(canvas, result.value, name, criterion(name).limiar);
      });
    }

    async function run(request) {
      busy.value = true;
      error.value = "";
      try {
        store.surgery = await request();
        await draw();
      } catch (e) {
        error.value = e.message;
      } finally {
        busy.value = false;
      }
    }

    const analyzeDataset = () => run(() => api(`/api/v1/cirurgia/base/${store.surgeryVideo}`, { method: "POST" }));
    async function upload(event) {
      const file = event.target.files[0];
      if (!file) return;
      await run(() => postFile("/api/v1/cirurgia/video", file, { fim_s: clip.value }));
      event.target.value = "";
    }

    onMounted(draw);
    onBeforeUnmount(() => Object.values(charts).forEach((chart) => chart.destroy()));
    watch(() => store.theme, draw);

    return { store, busy, error, clip, result, names, level, headline, agrees, criterion, analyzeDataset, upload, num };
  },
  template: `
    <div class="stack">
      <section class="card" v-if="store.surgeryInfo" aria-labelledby="surgery-title">
        <div class="card-head">
          <div>
            <h2 id="surgery-title">Analisar uma cirurgia</h2>
            <p class="small muted">Avalia a Visão Crítica de Segurança em vídeo gravado de colecistectomia laparoscópica, um frame a cada 5 segundos. Nenhuma imagem é salva ou exibida.</p>
          </div>
        </div>
        <p class="notice error" v-if="!store.surgeryInfo.modelos_ativos">Os modelos treinados não estão em backend/models, então a análise está desligada.</p>
        <template v-else>
          <div class="row" v-if="store.surgeryInfo.videos_base.length" style="margin-bottom: 14px">
            <div class="field">
              <label for="surgery-video">Vídeo do teste do Endoscapes2023 (base local)</label>
              <select id="surgery-video" class="select" v-model.number="store.surgeryVideo" :disabled="busy">
                <option v-for="video in store.surgeryInfo.videos_base" :key="video" :value="video">Vídeo {{ video }}</option>
              </select>
            </div>
            <button class="btn primary" @click="analyzeDataset" :disabled="busy">Analisar</button>
          </div>
          <p class="small muted" v-else>A base local não foi encontrada. Ainda é possível enviar um vídeo.</p>
          <div class="row">
            <div class="field">
              <label for="surgery-clip">Segundo do primeiro clipe (opcional)</label>
              <input id="surgery-clip" class="input number" type="number" min="1" step="1" v-model="clip" :disabled="busy" />
            </div>
            <div class="field">
              <label for="surgery-file">Ou envie um vídeo</label>
              <input id="surgery-file" class="file" type="file" accept=".mp4,.avi,.mov,.mkv" @change="upload" :disabled="busy" />
            </div>
          </div>
          <p class="small" v-if="busy" style="margin-top: 10px"><span class="spinner"></span> Analisando...</p>
          <p class="notice error" v-if="error" role="alert" style="margin-top: 10px">{{ error }}</p>
        </template>
      </section>
      <div class="card" v-else><div class="skeleton" style="width: 60%"></div></div>

      <template v-if="store.surgery">
        <section class="card" :class="level" aria-live="polite">
          <h2>{{ headline }}</h2>
          <p class="small" v-if="!result.avaliada" style="margin-top: 6px">{{ result.motivo }}</p>
          <p class="small" v-if="result.especialistas" style="margin-top: 6px">Marcação dos especialistas neste vídeo: {{ result.especialistas.cvs_atingido ? 'Visão Crítica atingida' : 'desvio, Visão Crítica não atingida' }}. O modelo {{ agrees ? 'concordou' : 'discordou' }}.</p>
          <p class="small muted" v-if="store.surgery.alerta" style="margin: 6px 0 0">Alerta registrado na aba Alertas.</p>
        </section>

        <div class="grid-main" v-if="result.avaliada">
          <section class="card">
            <h2>Pontuação dos critérios ao longo do vídeo</h2>
            <p class="small muted">Média de 3 frames. O alerta olha o maior valor de cada critério. A pontuação não é uma probabilidade, e pode ser alta antes de o critério ser atingido.</p>
            <div class="chart-box" v-for="name in names" :key="name">
              <div class="chart-head"><span>{{ name }} · limiar {{ num(criterion(name).limiar) }} (tracejado)</span><strong>pico {{ num(criterion(name).pico) }}</strong></div>
              <div class="chart-wrap"><canvas :id="'surgery-chart-' + name" :aria-label="'Pontuação de ' + name" role="img"></canvas></div>
            </div>
            <details>
              <summary>Ver em tabela</summary>
              <div class="table-scroll">
                <table>
                  <thead><tr><th>Segundo</th><th v-for="name in names" :key="name">{{ name }}</th></tr></thead>
                  <tbody><tr v-for="(second, index) in result.tempo_s" :key="second"><td>{{ second }}</td><td v-for="name in names" :key="name">{{ num(result.series[name][index]) }}</td></tr></tbody>
                </table>
              </div>
            </details>
          </section>
          <div class="stack">
            <section class="card">
              <h2 style="margin-bottom: 10px">Critérios</h2>
              <ul class="list">
                <li v-for="item in result.criterios" :key="item.criterio">
                  <span class="pill" :class="!item.usado ? '' : (item.atingido ? 'ok' : 'warn')">{{ !item.usado ? 'não usado' : (item.atingido ? 'atingiu o limiar' : 'abaixo do limiar') }}</span><br />
                  <strong>{{ item.criterio }}</strong>: {{ item.nome }}
                </li>
              </ul>
            </section>
            <section class="card">
              <h2>Estruturas detectadas</h2>
              <p class="small muted">Frames em que cada estrutura apareceu, de {{ result.frames_analisados }}. Não detectar uma estrutura não prova que ela está ausente.</p>
              <table><tbody><tr v-for="item in result.estruturas" :key="item.classe"><td>{{ item.nome }}<span class="muted" v-if="!item.confiavel"> (menos confiável)</span></td><td>{{ item.frames }}</td></tr></tbody></table>
            </section>
          </div>
        </div>
      </template>

      <!-- Desenho fixo: fica abaixo das estruturas detectadas e antes do laudo, com ou sem análise feita. -->
      <SurgeryAnatomy />

      <section class="card" v-if="store.surgery">
        <h2 style="margin-bottom: 8px">Laudo</h2>
        <p class="small" style="margin: 0">{{ store.surgery.laudo }}</p>
      </section>

      <section class="card" v-if="store.surgeryInfo">
        <h2 style="margin-bottom: 8px">Quanto confiar neste alerta</h2>
        <p class="small">Em {{ store.surgeryInfo.desempenho.videos }} vídeos de teste, o alerta pegou {{ store.surgeryInfo.desempenho.desvios_detectados.acertos }} de {{ store.surgeryInfo.desempenho.desvios_detectados.total }} desvios e não disparou em {{ store.surgeryInfo.desempenho.cvs_reconhecido.acertos }} de {{ store.surgeryInfo.desempenho.cvs_reconhecido.total }} cirurgias com a Visão Crítica atingida. AUROC {{ num(store.surgeryInfo.desempenho.AUROC) }}, com margem de {{ num(store.surgeryInfo.desempenho.AUROC_IC95[0]) }} a {{ num(store.surgeryInfo.desempenho.AUROC_IC95[1]) }}.</p>
        <p class="small muted" style="margin: 0">O C1 não entra na decisão, então um desvio só em C1 passa sem aviso. Base de um único centro, sem validação clínica. Apoio à decisão; não substitui a avaliação da equipe.</p>
      </section>
    </div>`,
};
