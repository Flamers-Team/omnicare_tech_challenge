import { onBeforeUnmount, ref } from "./vue.js";

/** Converte o áudio gravado pelo navegador (webm/ogg) em WAV PCM 16 kHz mono, que é o que o servidor aceita. */
export async function blobToWav(blob) {
  const context = new AudioContext({ sampleRate: 16000 });
  let decoded;
  try {
    decoded = await context.decodeAudioData(await blob.arrayBuffer());
  } finally {
    await context.close();
  }
  const samples = decoded.getChannelData(0);
  const view = new DataView(new ArrayBuffer(44 + samples.length * 2));
  const text = (offset, value) => [...value].forEach((c, i) => view.setUint8(offset + i, c.charCodeAt(0)));
  text(0, "RIFF"); view.setUint32(4, 36 + samples.length * 2, true); text(8, "WAVE"); text(12, "fmt ");
  view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, 16000, true); view.setUint32(28, 32000, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  text(36, "data"); view.setUint32(40, samples.length * 2, true);
  samples.forEach((s, i) => view.setInt16(44 + i * 2, Math.max(-1, Math.min(1, s)) * 0x7fff, true));
  const stamp = new Date().toISOString().slice(11, 19).replace(/:/g, "");
  return new File([view], `gravacao_${stamp}.wav`, { type: "audio/wav" });
}

/** Gravador de microfone. `onFinish` recebe o WAV pronto. */
export function useRecorder(onFinish) {
  const recording = ref(false);
  const seconds = ref(0);
  const error = ref("");
  let recorder = null;
  let timer = null;
  let stream = null;

  function release() {
    clearInterval(timer);
    stream?.getTracks().forEach((track) => track.stop());
    stream = null;
    recording.value = false;
  }

  async function start() {
    error.value = "";
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false } });
    } catch (e) {
      error.value = `Não foi possível acessar o microfone: ${e.message}`;
      return;
    }
    const chunks = [];
    recorder = new MediaRecorder(stream);
    recorder.ondataavailable = (event) => chunks.push(event.data);
    recorder.onstop = async () => {
      const type = recorder.mimeType;
      release();
      try {
        await onFinish(await blobToWav(new Blob(chunks, { type })));
      } catch (e) {
        error.value = `Não foi possível processar a gravação: ${e.message}`;
      }
    };
    seconds.value = 0;
    timer = setInterval(() => { seconds.value += 1; }, 1000);
    recording.value = true;
    recorder.start();
  }

  function stop() {
    if (recorder && recorder.state !== "inactive") recorder.stop();
  }

  onBeforeUnmount(() => {
    if (recorder && recorder.state !== "inactive") recorder.onstop = null;
    release();
  });

  return { recording, seconds, error, start, stop };
}
