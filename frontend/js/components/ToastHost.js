import { dismissToast, store } from "../store.js";
import { levelName } from "../format.js";

export default {
  name: "ToastHost",
  setup() {
    return { store, dismissToast, levelName };
  },
  template: `
    <div class="toasts" role="status" aria-live="polite">
      <div v-for="toast in store.toasts" :key="toast.id" class="toast" :class="toast.nivel" data-testid="toast">
        <div class="row center" style="justify-content: space-between; flex-wrap: nowrap">
          <span class="toast-title">Novo alerta, risco {{ levelName(toast.nivel) }}: {{ toast.titulo }}</span>
          <button class="icon-btn" style="width: 28px; height: 28px" @click="dismissToast(toast.id)" aria-label="Fechar aviso">×</button>
        </div>
        <p class="small muted" style="margin: 4px 0 0">{{ toast.texto }}</p>
      </div>
    </div>`,
};
