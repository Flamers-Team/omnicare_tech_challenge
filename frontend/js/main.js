import { createApp } from "./vue.js";
import { boot, store } from "./store.js";
import AppHeader from "./components/AppHeader.js";
import StatusStrip from "./components/StatusStrip.js";
import InternacaoView from "./components/InternacaoView.js";
import SurgeryView from "./components/SurgeryView.js";
import AlertsView from "./components/AlertsView.js";
import ToastHost from "./components/ToastHost.js";

// Tema escolhido antes: aplica antes de desenhar para não piscar.
try {
  const saved = localStorage.getItem("omnicare-theme");
  if (saved === "light" || saved === "dark") {
    store.theme = saved;
    document.documentElement.dataset.theme = saved;
  }
} catch { /* sem armazenamento: usa o tema do sistema */ }

createApp({
  components: { AppHeader, StatusStrip, InternacaoView, SurgeryView, AlertsView, ToastHost },
  setup() {
    boot();
    return { store };
  },
  template: `
    <AppHeader />
    <main class="container page" :aria-labelledby="'tab-' + store.tab" role="tabpanel">
      <StatusStrip />
      <InternacaoView v-if="store.tab === 'internacao'" />
      <SurgeryView v-else-if="store.tab === 'cirurgia'" />
      <AlertsView v-else />
    </main>
    <ToastHost />`,
}).mount("#app");
