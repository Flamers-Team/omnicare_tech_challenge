import { ref } from "../vue.js";
import { api, postFile } from "../api.js";
import { applyContext, store } from "../store.js";
import { num } from "../format.js";

export default {
  name: "GaitPanel",
  setup() {
    const busy = ref(false);
    const error = ref("");

    async function run(request) {
      busy.value = true;
      error.value = "";
      try {
        const result = await request();
        store.gait = result.resultado;
        applyContext(result);
      } catch (e) {
        error.value = e.message;
      } finally {
        busy.value = false;
      }
    }

    const example = (name) => run(() => api(`/api/v1/pacientes/${store.stayId}/marcha/exemplo/${name}`, { method: "POST" }));
    async function upload(event) {
      const file = event.target.files[0];
      if (!file) return;
      await run(() => postFile(`/api/v1/pacientes/${store.stayId}/marcha`, file));
      event.target.value = "";
    }
    return { busy, error, store, example, upload, num };
  },
  template: `
    <section class="card" aria-labelledby="gait-title">
      <div class="card-head">
        <div>
          <h2 id="gait-title">Marcha</h2>
          <p class="small muted">Compara a caminhada com a referência de 14 adultos saudáveis usando andador.</p>
        </div>
      </div>
      <div class="row" style="margin-bottom: 12px">
        <button class="btn" @click="example('normal')" :disabled="busy">Exemplo normal</button>
        <button class="btn" @click="example('alterada')" :disabled="busy">Tronco 20° à frente (simulado)</button>
      </div>
      <div class="field">
        <label for="gait-file">Ou envie um CSV de esqueleto ou um vídeo da caminhada</label>
        <input id="gait-file" class="file" type="file" accept=".csv,.mp4,.avi,.mov,.mkv" @change="upload" :disabled="busy" />
      </div>
      <p class="small" v-if="busy" style="margin-top: 10px"><span class="spinner"></span> Analisando...</p>
      <p class="notice error" v-if="error" role="alert" style="margin-top: 10px">{{ error }}</p>
      <div v-if="store.gait" style="margin-top: 14px" data-testid="gait-result">
        <p class="small muted">{{ store.gait.fonte_dados }}<span v-if="store.gait.arquivo"> · {{ store.gait.arquivo }}</span></p>
        <template v-if="store.gait.avaliada">
          <p>
            <span class="pill" :class="store.gait.alerta ? 'warn' : 'ok'">{{ store.gait.alerta ? 'Fora da referência' : 'Dentro da referência' }}</span>
            <span class="small"> nota {{ num(store.gait.pontuacao) }}, limiar {{ num(store.gait.limiar) }}, {{ store.gait.janelas }} janelas</span>
          </p>
          <p class="small">Maior desvio: {{ store.gait.medida_mais_distante.descricao }}. Valor {{ num(store.gait.medida_mais_distante.valor) }}; referência {{ num(store.gait.medida_mais_distante.referencia) }}.</p>
        </template>
        <p class="small" v-else>Não avaliada: {{ store.gait.motivo }}</p>
        <p class="small muted" v-if="store.gait.aviso">{{ store.gait.aviso }}</p>
      </div>
    </section>`,
};
