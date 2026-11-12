import { computed } from "../vue.js";
import { store } from "../store.js";

export default {
  name: "StatusStrip",
  setup() {
    const items = computed(() => {
      const s = store.status;
      if (!s) return [];
      return [
        { name: "Sinais vitais e prescrições", active: s.sinais_vitais_prescricoes.ativo },
        { name: "Marcha", active: s.marcha.ativo },
        { name: "Sinal vocal", active: s.sinal_vocal.ativo },
        { name: "Azure Speech", active: s.azure_speech.ativo },
        { name: "Azure Language", active: s.azure_language.ativo },
        { name: "Modelos de cirurgia", active: s.cirurgia.ativo },
      ];
    });
    return { items, store };
  },
  template: `
    <div class="status-strip" role="list" aria-label="Estado dos módulos">
      <span v-for="item in items" :key="item.name" role="listitem" class="pill" :class="item.active ? 'ok' : ''">
        <span class="dot"></span>{{ item.name }}: {{ item.active ? 'ativo' : 'não ligado' }}
      </span>
      <div v-if="!store.status && !store.loadError" class="skeleton" style="width: 320px"></div>
    </div>
    <p v-if="store.loadError" class="notice error" role="alert">{{ store.loadError }}</p>`,
};
