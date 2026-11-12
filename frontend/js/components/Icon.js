const PATHS = {
  plus: "M12 5v14M5 12h14",
  sun: "M12 3v2M12 19v2M5 12H3M21 12h-2M6.3 6.3 4.9 4.9M19.1 19.1l-1.4-1.4M6.3 17.7l-1.4 1.4M19.1 4.9l-1.4 1.4M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8Z",
  moon: "M20 14.5A8 8 0 1 1 9.5 4a6.5 6.5 0 0 0 10.5 10.5Z",
  check: "m5 12.5 4.5 4.5L19 7.5",
  warn: "M12 4 2.5 20h19L12 4ZM12 10v4.5M12 17.5h.01",
  alert: "M12 3 3 8v8l9 5 9-5V8l-9-5ZM12 8v5M12 16h.01",
  mic: "M12 3a3 3 0 0 0-3 3v6a3 3 0 0 0 6 0V6a3 3 0 0 0-3-3ZM6 11a6 6 0 0 0 12 0M12 17v4",
  stop: "M7 7h10v10H7z",
  play: "M8 5v14l11-7z",
  circle: "M12 4a8 8 0 1 0 0 16 8 8 0 0 0 0-16Z",
};

export default {
  name: "Icon",
  props: { name: { type: String, required: true }, size: { type: Number, default: 18 } },
  template: `
    <svg :width="size" :height="size" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
      <path :d="path" />
    </svg>`,
  computed: {
    path() { return PATHS[this.name] ?? PATHS.circle; },
  },
};
