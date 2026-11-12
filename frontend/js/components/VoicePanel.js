import { computed, markRaw, ref } from "../vue.js";
import { postFile } from "../api.js";
import { useRecorder } from "../audio.js";
import { applyContext, store } from "../store.js";
import { num } from "../format.js";
import Icon from "./Icon.js";

export default {
  name: "VoicePanel",
  components: { Icon },
  setup() {
    const busy = ref(false);
    const error = ref("");
    const target = ref(null); // "base" ou "consulta", conforme o que está sendo gravado

    async function analyze(file) {
      busy.value = true;
      error.value = "";
      try {
        const result = await postFile(`/api/v1/pacientes/${store.stayId}/voz`, file, { base: store.voiceBase });
        store.voice = result.resultado;
        applyContext(result);
      } catch (e) {
        error.value = e.message;
      } finally {
        busy.value = false;
      }
    }

    const recorder = useRecorder(async (wav) => {
      if (target.value === "base") store.voiceBase = markRaw(wav);
      else await analyze(wav);
    });

    function toggle(kind) {
      if (recorder.recording.value) {
        recorder.stop();
      } else {
        target.value = kind;
        recorder.start();
      }
    }

    async function uploadConsulta(event) {
      const file = event.target.files[0];
      if (!file) return;
      await analyze(file);
      event.target.value = "";
    }
    const pickBase = (event) => { store.voiceBase = event.target.files[0] ? markRaw(event.target.files[0]) : null; };

    const severe = computed(() => store.voice?.analise_texto?.termos_criticos ?? []);
    const mild = computed(() => store.voice?.analise_texto?.termos_leves ?? []);
    const sentimentTone = computed(() => ({ negative: "crit", positive: "ok", neutral: "" })[store.voice?.analise_texto?.sentimento_dominante] ?? "");
    const sentimentName = computed(() => ({ negative: "negativo", positive: "positivo", neutral: "neutro", mixed: "misto" })[store.voice?.analise_texto?.sentimento_dominante] ?? "");

    return { busy, error, store, recorder, target, toggle, uploadConsulta, pickBase, severe, mild, sentimentTone, sentimentName, num };
  },
  template: `
    <section class="card" aria-labelledby="voice-title">
      <div class="card-head">
        <div>
          <h2 id="voice-title">Voz e fala</h2>
          <p class="small muted">Sem as chaves do Azure, nada é transcrito. Nenhum texto é inventado.</p>
        </div>
      </div>

      <div class="stack" style="gap: 12px">
        <div class="rec">
          <button class="btn" @click="toggle('base')" :disabled="busy || (recorder.recording.value && target !== 'base')" data-testid="rec-base">
            <Icon :name="recorder.recording.value && target === 'base' ? 'stop' : 'mic'" :size="16" />
            {{ recorder.recording.value && target === 'base' ? 'Parar' : 'Gravar repouso' }}
          </button>
          <button class="btn primary" @click="toggle('consulta')" :disabled="busy || (recorder.recording.value && target !== 'consulta')" data-testid="rec-consulta">
            <Icon :name="recorder.recording.value && target === 'consulta' ? 'stop' : 'mic'" :size="16" />
            {{ recorder.recording.value && target === 'consulta' ? 'Parar e analisar' : 'Gravar consulta' }}
          </button>
          <span class="small row center" v-if="recorder.recording.value" role="status"><span class="rec-dot"></span> Gravando... {{ recorder.seconds.value }} s</span>
          <span class="small muted" v-else-if="store.voiceBase" data-testid="base-ready">Repouso pronto: {{ store.voiceBase.name }}</span>
        </div>
        <p class="small muted" style="margin: 0">Grave primeiro o repouso (15 s falando normalmente, com pausas de 1 s entre as frases) para o sinal vocal somar no risco.</p>

        <details>
          <summary>Ou envie arquivos .wav</summary>
          <div class="stack" style="gap: 10px; margin-top: 10px">
            <div class="field">
              <label for="voice-base">Gravação em repouso (opcional, recomendada)</label>
              <input id="voice-base" class="file" type="file" accept=".wav" @change="pickBase" :disabled="busy" />
            </div>
            <div class="field">
              <label for="voice-file">Áudio da consulta (a análise começa ao escolher o arquivo)</label>
              <input id="voice-file" class="file" type="file" accept=".wav" @change="uploadConsulta" :disabled="busy" />
            </div>
          </div>
        </details>
      </div>

      <p class="small" v-if="busy" style="margin-top: 12px"><span class="spinner"></span> Analisando...</p>
      <p class="notice error" v-if="error || recorder.error.value" role="alert" style="margin-top: 12px">{{ error || recorder.error.value }}</p>

      <div v-if="store.voice" class="stack" style="margin-top: 16px; gap: 12px" data-testid="voice-result">
        <p class="small muted" style="margin: 0">{{ store.voice.acustica.arquivo }} · {{ store.voice.acustica.duracao_s }} s</p>
        <div>
          <h3>Transcrição</h3>
          <p class="transcript" style="margin-top: 6px">{{ store.voice.transcricao || 'indisponível' }}</p>
        </div>
        <div v-if="store.voice.analise_texto">
          <h3>Análise do texto (Azure Language)</h3>
          <div class="chips" style="margin-top: 6px">
            <span class="pill" :class="sentimentTone">Sentimento {{ sentimentName }}</span>
            <span v-for="term in severe" :key="'s' + term" class="pill crit">{{ term }}</span>
            <span v-for="term in mild" :key="'m' + term" class="pill warn">{{ term }}</span>
            <span v-if="!severe.length && !mild.length" class="pill">Nenhum sintoma da lista</span>
          </div>
          <p class="small muted" style="margin: 6px 0 0">Vermelho: sintoma grave (gera alerta). Amarelo: outra queixa (só informa). Termos negados não contam.</p>
        </div>
        <p class="small" v-else>{{ store.voice.descricao }}</p>
        <div>
          <h3>Sinal vocal</h3>
          <template v-if="store.voice.sinal_vocal.avaliado">
            <p style="margin: 6px 0 0" class="small">
              <span class="pill" :class="store.voice.sinal_vocal.presente ? 'warn' : 'ok'">{{ store.voice.sinal_vocal.presente ? 'presente' : 'ausente' }}</span>
              {{ store.voice.sinal_vocal.descricao }}: {{ num(store.voice.sinal_vocal.pontuacao) }}, limiar {{ num(store.voice.sinal_vocal.limiar) }}, em {{ store.voice.sinal_vocal.frases }} trechos de fala.
            </p>
            <p class="small muted">{{ store.voice.sinal_vocal.aviso }}</p>
          </template>
          <p v-else class="small" style="margin: 6px 0 0">{{ store.voice.sinal_vocal.motivo }}</p>
        </div>
      </div>
    </section>`,
};
