/** Chamada à API. Devolve o JSON ou lança Error com a mensagem do servidor. */
export async function api(url, options) {
  const response = await fetch(url, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(typeof body.detail === "string" ? body.detail : "O servidor não conseguiu concluir o pedido.");
  }
  return body;
}

export function postJson(url, data) {
  return api(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });
}

export function postFile(url, file, extra = {}) {
  const form = new FormData();
  form.append("arquivo", file);
  Object.entries(extra).forEach(([key, value]) => {
    if (value) form.append(key, value);
  });
  return api(url, { method: "POST", body: form });
}
