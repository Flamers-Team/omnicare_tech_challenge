/* Funções de gráfico. O Chart.js vem de vendor/chart.umd.js e fica em `window.Chart`. */

export const cssColor = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

function baseOptions() {
  return {
    animation: false,
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: "index", intersect: false },
    scales: {
      x: { grid: { display: false }, border: { color: cssColor("--border-strong") }, ticks: { color: cssColor("--muted"), maxTicksLimit: 8, maxRotation: 0, font: { size: 11 } } },
      y: { grid: { color: cssColor("--grid") }, border: { display: false }, ticks: { color: cssColor("--muted"), maxTicksLimit: 3, font: { size: 11 } } },
    },
  };
}

export function createVitalChart(canvas, vital) {
  const options = baseOptions();
  options.plugins = {
    legend: { display: false },
    tooltip: { displayColors: false, callbacks: { title: (items) => `Hora ${items[0].label}`, label: (item) => `${item.formattedValue} ${vital.unit}` } },
  };
  return new window.Chart(canvas, {
    type: "line",
    data: { labels: [], datasets: [{ data: [], borderColor: cssColor("--line"), borderWidth: 2, spanGaps: 6, tension: 0.15, pointRadius: [], pointHoverRadius: 4 }] },
    options,
  });
}

/** Atualiza o gráfico de um sinal. `marks` são os instantes com evento, destacados em vermelho. */
export function updateVitalChart(chart, hours, values, marks) {
  const dataset = chart.data.datasets[0];
  chart.data.labels = hours.map((_, index) => index + 1);
  // Cópia simples: o Chart.js não funciona com os arrays reativos do Vue.
  dataset.data = [...values];
  // Sinais medidos poucas vezes, como a temperatura, precisam de um ponto para aparecer sem linha.
  dataset.pointRadius = hours.map((hour) => (marks.has(hour) ? 5 : 1.5));
  dataset.pointBackgroundColor = hours.map((hour) => cssColor(marks.has(hour) ? "--crit" : "--line"));
  dataset.pointBorderColor = cssColor("--surface");
  dataset.pointBorderWidth = hours.map((hour) => (marks.has(hour) ? 2 : 0));
  dataset.borderColor = cssColor("--line");
  chart.update("none");
}

export function createSurgeryChart(canvas, result, name, limit) {
  const options = baseOptions();
  options.scales.y.min = 0;
  options.scales.y.max = 1;
  options.plugins = { legend: { display: false }, tooltip: { displayColors: false, callbacks: { title: (items) => `${items[0].label} s` } } };
  return new window.Chart(canvas, {
    type: "line",
    data: {
      labels: [...result.tempo_s],
      datasets: [
        { label: "Pontuação", data: [...result.series[name]], borderColor: cssColor("--line"), borderWidth: 2, tension: 0.15, pointRadius: 0, pointHoverRadius: 4 },
        { label: "Limiar", data: result.tempo_s.map(() => limit), borderColor: cssColor("--muted"), borderWidth: 1, borderDash: [4, 4], pointRadius: 0, pointHoverRadius: 0 },
      ],
    },
    options,
  });
}
