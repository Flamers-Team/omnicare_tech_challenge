// Desenho fixo das estruturas que o detector procura. É um esquema próprio, não uma imagem da base Endoscapes.
// A ordem e os nomes seguem a tabela "Estruturas detectadas" (STRUCTURE_NAMES no backend).

// Cada estrutura: posição do número (bx, by), início e fim da linha indicadora (from, to) e onde o nome é escrito.
const STRUCTURES = [
  {
    n: 1, name: "Vesícula", bx: 668, by: 214, from: [654, 214], to: [528, 222], textX: 686,
    text: "O órgão que guarda a bile. É o que a cirurgia retira.",
  },
  {
    n: 2, name: "Instrumento", bx: 668, by: 290, from: [668, 304], to: [668, 350], textX: 686,
    text: "Pinças e outras ferramentas que entram por pequenos furos para segurar, afastar e cortar.",
  },
  {
    n: 3, name: "Triângulo de Calot", bx: 26, by: 262, from: [198, 262], to: [328, 262], textX: 44,
    text: "Área entre o ducto cístico, a via biliar principal e a borda do fígado. Precisa estar limpa para se ver cada estrutura (critério C2).",
  },
  {
    n: 4, name: "Ducto cístico", bx: 352, by: 336, from: [352, 322], to: [336, 281], textX: 370,
    text: "Canal que leva a bile da vesícula à via biliar principal. É uma das duas estruturas do critério C1 e, depois da Visão Crítica, recebe os clipes.",
  },
  {
    n: 5, name: "Artéria cística", bx: 26, by: 222, from: [164, 222], to: [345, 196], textX: 44,
    text: "Artéria que leva sangue à vesícula. É a outra estrutura do critério C1.",
  },
  {
    n: 6, name: "Placa cística", bx: 668, by: 128, from: [654, 128], to: [532, 148], textX: 686,
    text: "Parte do fígado onde a vesícula fica presa. Separar a base da vesícula dessa placa é o critério C3.",
  },
];

export default {
  name: "SurgeryAnatomy",
  setup() {
    return { STRUCTURES };
  },
  template: `
    <section class="card" aria-labelledby="anatomy-title" data-testid="anatomy">
      <div class="card-head">
        <div>
          <h2 id="anatomy-title">Onde fica cada estrutura</h2>
          <p class="small muted">Desenho para ajudar a ler a tabela “Estruturas detectadas”. Os números seguem a ordem da tabela.</p>
        </div>
      </div>
      <figure class="anatomy-figure">
        <svg class="anatomy-svg" viewBox="0 0 800 440" role="img" aria-labelledby="anatomy-svg-title anatomy-svg-desc">
          <title id="anatomy-svg-title">Esquema da região operada na retirada da vesícula</title>
          <desc id="anatomy-svg-desc">O fígado fica em cima. A vesícula fica presa embaixo dele, pela placa cística. O ducto cístico sai do colo da vesícula e se junta à via biliar principal. A artéria cística atravessa o triângulo de Calot até a vesícula. Uma pinça segura o fundo da vesícula.</desc>

          <!-- Fígado e placa cística -->
          <path class="anat-liver" d="M 170 64 C 172 30 220 18 300 16 L 560 18 C 625 22 640 52 632 92 C 624 132 590 148 540 150 C 480 153 430 146 380 150 C 345 153 325 162 300 166 C 255 174 215 176 190 162 C 168 150 168 100 170 64 Z" />
          <text class="anat-context" x="246" y="92">fígado</text>
          <path class="anat-plate-edge" d="M 392 152 Q 470 140 556 154" />
          <path class="anat-plate" d="M 392 152 Q 470 140 556 154" />

          <!-- Triângulo de Calot: entre a via biliar, o ducto cístico e a borda do fígado -->
          <path class="anat-zone" d="M 309 166 L 380 158 C 376 170 372 184 372 198 C 372 240 352 280 309 296 Z" />

          <!-- Artéria cística, passando por trás da via biliar -->
          <path class="anat-artery" d="M 208 244 C 270 232 330 200 390 178" />

          <!-- Via biliar principal -->
          <path class="anat-duct-edge" d="M 300 160 L 300 440" stroke-width="15" />
          <path class="anat-duct" d="M 300 160 L 300 440" stroke-width="10" />
          <text class="anat-context" x="290" y="424" text-anchor="end">via biliar principal</text>

          <!-- Vesícula e ducto cístico -->
          <path class="anat-gb" d="M 384 168 C 414 150 505 150 550 176 C 596 203 596 284 550 296 C 505 308 445 270 410 238 C 392 222 372 214 372 198 C 372 184 376 174 384 168 Z" />
          <path class="anat-duct-edge" d="M 374 200 C 372 240 352 280 306 298" stroke-width="11" />
          <path class="anat-duct" d="M 374 200 C 372 240 352 280 306 298" stroke-width="7" />

          <!-- Instrumento (pinça) segurando o fundo da vesícula -->
          <path class="anat-tool-edge" d="M 760 420 L 600 300" stroke-width="16" />
          <path class="anat-tool" d="M 760 420 L 600 300" stroke-width="12" />
          <path class="anat-tool-edge" d="M 600 300 L 574 286 M 600 300 L 584 272" stroke-width="6" />
          <circle class="anat-joint" cx="600" cy="300" r="6" />

          <!-- Números, linhas indicadoras e nomes -->
          <g v-for="item in STRUCTURES" :key="item.n">
            <line class="anat-leader" :x1="item.from[0]" :y1="item.from[1]" :x2="item.to[0]" :y2="item.to[1]" />
            <circle class="anat-dot" :cx="item.to[0]" :cy="item.to[1]" r="4" />
            <g class="anat-badge">
              <circle :cx="item.bx" :cy="item.by" r="12" />
              <text :x="item.bx" :y="item.by" text-anchor="middle" dominant-baseline="central">{{ item.n }}</text>
            </g>
            <text class="anat-label" :x="item.textX" :y="item.by" dominant-baseline="central">{{ item.name }}</text>
          </g>
        </svg>
        <figcaption class="small muted">Esquema simplificado e fora de escala, feito para este sistema. Não é uma imagem da base. Na cirurgia, a câmera vê a região por outro ângulo, com o fígado levantado pelos instrumentos.</figcaption>
      </figure>
      <ol class="anatomy-legend">
        <li v-for="item in STRUCTURES" :key="item.n">
          <span class="num-badge" aria-hidden="true">{{ item.n }}</span>
          <span><strong>{{ item.name }}.</strong> {{ item.text }}</span>
        </li>
      </ol>
    </section>`,
};
