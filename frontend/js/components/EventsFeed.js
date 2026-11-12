import { computed } from "../vue.js";
import { store } from "../store.js";
import { eventType } from "../format.js";

export default {
  name: "EventsFeed",
  setup() {
    const events = computed(() => [...store.vitalEvents, ...store.rxEvents].sort((a, b) => (a.inicio < b.inicio ? 1 : -1)));
    function hourIndex(iso) {
      const start = store.hours[0];
      if (!start) return "–";
      return Math.round((new Date(iso) - new Date(start)) / 3600000) + 1;
    }
    return { events, hourIndex, eventType };
  },
  template: `
    <section class="card" aria-labelledby="events-title">
      <div class="card-head"><h2 id="events-title">Eventos</h2><span class="pill info">{{ events.length }}</span></div>
      <div class="feed">
        <ul class="list">
          <li v-for="(event, index) in events" :key="index">
            <span class="tag">{{ eventType(event) }}</span> · hora {{ hourIndex(event.inicio) }}<br />{{ event.descricao }}
          </li>
        </ul>
        <p class="small muted" v-if="!events.length">Nenhum evento até aqui.</p>
      </div>
    </section>`,
};
