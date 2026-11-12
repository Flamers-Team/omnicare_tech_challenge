import { computed } from "../vue.js";
import { store } from "../store.js";
import Icon from "./Icon.js";

const TABS = [
  { id: "internacao", label: "Internação e reabilitação" },
  { id: "cirurgia", label: "Centro cirúrgico" },
  { id: "alertas", label: "Alertas" },
];

function currentTheme() {
  if (store.theme !== "auto") return store.theme;
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

export default {
  name: "AppHeader",
  components: { Icon },
  setup() {
    const openAlerts = computed(() => store.alerts.filter((alert) => !alert.confirmado_por).length);
    const dark = computed(() => currentTheme() === "dark");

    function toggleTheme() {
      store.theme = dark.value ? "light" : "dark";
      document.documentElement.dataset.theme = store.theme;
      try { localStorage.setItem("omnicare-theme", store.theme); } catch { /* sem armazenamento: segue sem lembrar */ }
    }
    return { TABS, store, openAlerts, dark, toggleTheme };
  },
  template: `
    <header class="topbar">
      <div class="topbar-inner">
        <div class="brand">
          <div class="brand-mark"><Icon name="plus" :size="20" /></div>
          <div>
            <div class="brand-name">OmniCare AI</div>
            <div class="brand-sub">Monitoramento hospitalar multimodal · protótipo sem validação clínica</div>
          </div>
        </div>
        <nav class="tabs" role="tablist" aria-label="Seções">
          <button v-for="tab in TABS" :key="tab.id" class="tab" role="tab" :id="'tab-' + tab.id"
                  :aria-selected="store.tab === tab.id" @click="store.tab = tab.id">
            {{ tab.label }}
            <span v-if="tab.id === 'alertas' && openAlerts" class="badge-count" :aria-label="openAlerts + ' alertas abertos'">{{ openAlerts }}</span>
          </button>
        </nav>
        <button class="icon-btn" @click="toggleTheme" :aria-label="dark ? 'Usar tema claro' : 'Usar tema escuro'" :title="dark ? 'Tema claro' : 'Tema escuro'">
          <Icon :name="dark ? 'sun' : 'moon'" />
        </button>
      </div>
    </header>`,
};
