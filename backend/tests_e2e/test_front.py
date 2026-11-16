import json
import re

from playwright.sync_api import expect

from services.surgery_video_service import RULE_FILE, STRUCTURE_NAMES
from tests_e2e.conftest import BACKEND, make_wav


def open_alert_count(page) -> int:
    badge = page.locator(".badge-count")
    return int(badge.inner_text()) if badge.count() else 0


def alert_total(page) -> int:
    return page.evaluate("fetch('/api/v1/alertas').then(r => r.json()).then(d => d.alertas.length)")


def test_page_loads_without_errors(page):
    expect(page).to_have_title("OmniCare AI")
    expect(page.get_by_text("Marcha: ativo")).to_be_visible()
    expect(page.locator(".kpis .kpi")).to_have_count(4)
    expect(page.get_by_test_id("kpi-level")).not_to_have_text("–")  # aguarda a linha do tempo
    assert page.errors == []


def test_libraries_load_from_local_files(page):
    # Sem internet o front precisa continuar funcionando: nada pode vir de CDN.
    hosts = page.evaluate("() => performance.getEntriesByType('resource').map(e => new URL(e.name).host)")
    assert set(hosts) == {page.url.split("//")[1].split("/")[0]}


def test_five_vital_charts_are_drawn(page):
    expect(page.get_by_test_id("kpi-level")).not_to_have_text("–")
    canvases = page.locator(".chart-wrap canvas")
    expect(canvases).to_have_count(5)
    for index in range(5):
        box = canvases.nth(index).bounding_box()
        assert box["width"] > 100 and box["height"] > 40


def test_monitor_replay_updates_and_stops(page):
    page.select_option("#speed", "0.08")
    page.get_by_test_id("start-monitor").click()
    expect(page.get_by_test_id("stop-monitor")).to_be_visible()
    expect(page.get_by_text(re.compile(r"Hora [1-9]\d*"))).to_be_visible()
    page.get_by_test_id("stop-monitor").click()
    expect(page.get_by_test_id("start-monitor")).to_be_visible()
    assert page.errors == []


def test_gait_alert_flow_to_confirmation(page):
    expect(page.get_by_test_id("kpi-level")).not_to_have_text("–")
    before = open_alert_count(page)
    page.get_by_role("button", name=re.compile("Tronco 20")).click()
    expect(page.get_by_test_id("gait-result")).to_contain_text("Fora da referência")
    # O alerta chega pelo stream: aviso flutuante e contador na aba.
    expect(page.get_by_test_id("toast").first).to_be_visible()
    expect(page.locator(".badge-count")).to_have_text(str(before + 1))

    page.get_by_role("tab", name=re.compile("Alertas")).click()
    card = page.get_by_test_id("alert-card").filter(has_text="Marcha fora da referência").first
    expect(card).to_be_visible()
    button = card.get_by_test_id("confirm-alert")
    expect(button).to_be_disabled()  # sem nome, não confirma
    page.fill("#responsible", "Enfermeira Teste")
    button.click()
    expect(card).to_contain_text("Enfermeira Teste")
    if before:
        expect(page.locator(".badge-count")).to_have_text(str(before))
    else:
        expect(page.locator(".badge-count")).to_have_count(0)


def test_normal_gait_does_not_alert(page):
    expect(page.get_by_test_id("kpi-level")).not_to_have_text("–")
    before = alert_total(page)
    page.get_by_role("button", name="Exemplo normal").click()
    expect(page.get_by_test_id("gait-result")).to_contain_text("Dentro da referência")
    page.wait_for_timeout(1500)  # dá tempo de um alerta indevido chegar
    assert alert_total(page) == before


def test_voice_upload_without_azure_shows_clear_message(page, tmp_path):
    expect(page.get_by_test_id("kpi-level")).not_to_have_text("–")
    wav = make_wav(tmp_path / "consulta.wav")
    page.get_by_text("Ou envie arquivos .wav").click()
    page.set_input_files("#voice-file", str(wav))
    result = page.get_by_test_id("voice-result")
    expect(result).to_be_visible()
    expect(result).to_contain_text("indisponível")  # sem chave, nada é inventado
    assert page.errors == []


def test_microphone_recording_is_sent_as_wav(page):
    expect(page.get_by_test_id("kpi-level")).not_to_have_text("–")
    bodies = []

    def capture(route):
        bodies.append(route.request.post_data_buffer)  # lido aqui, antes de o pedido seguir
        route.continue_()

    page.route(re.compile(r"/voz$"), capture)
    page.get_by_test_id("rec-consulta").click()
    expect(page.get_by_text(re.compile(r"Gravando\.\.\. [0-9]+ s"))).to_be_visible()
    page.wait_for_timeout(2500)
    with page.expect_response(lambda r: r.url.endswith("/voz") and r.request.method == "POST", timeout=30000) as sent:
        page.get_by_test_id("rec-consulta").click()
    assert sent.value.status == 200
    assert bodies and bodies[0], "o corpo do envio não foi capturado"
    assert b'filename="gravacao_' in bodies[0] and b"RIFF" in bodies[0] and b"WAVE" in bodies[0]
    expect(page.get_by_test_id("voice-result")).to_contain_text("gravacao_")


def test_rest_recording_is_kept_for_the_consultation(page):
    page.get_by_test_id("rec-base").click()
    page.wait_for_timeout(1500)
    page.get_by_test_id("rec-base").click()
    expect(page.get_by_test_id("base-ready")).to_contain_text("gravacao_")


def test_theme_toggle_keeps_charts_and_remembers_choice(page):
    expect(page.get_by_test_id("kpi-level")).not_to_have_text("–")
    before = page.evaluate("document.documentElement.dataset.theme || 'auto'")
    page.get_by_role("button", name=re.compile("Usar tema")).click()
    after = page.evaluate("document.documentElement.dataset.theme")
    assert after in {"light", "dark"} and after != before
    expect(page.locator(".chart-wrap canvas")).to_have_count(5)
    page.reload()
    page.wait_for_selector(".status-strip .pill")
    assert page.evaluate("document.documentElement.dataset.theme") == after


def test_surgery_tab_renders(page):
    page.get_by_role("tab", name="Centro cirúrgico").click()
    expect(page.get_by_role("heading", name="Analisar uma cirurgia")).to_be_visible()
    expect(page.get_by_role("heading", name="Quanto confiar neste alerta")).to_be_visible()
    assert page.errors == []


def test_surgery_drawing_shows_every_detected_structure(page):
    # O desenho é fixo, aparece antes de qualquer análise e segue os nomes e a ordem da tabela "Estruturas detectadas".
    ap50 = json.loads((BACKEND / "models" / RULE_FILE).read_text(encoding="utf-8"))["detector_ap50_teste"]
    table_order = [STRUCTURE_NAMES[name] for name in sorted(STRUCTURE_NAMES, key=lambda item: -ap50[item])]
    page.get_by_role("tab", name="Centro cirúrgico").click()
    drawing = page.get_by_test_id("anatomy")
    expect(drawing.get_by_role("img", name=re.compile("retirada da vesícula"))).to_be_visible()
    expect(drawing.locator(".anat-label")).to_have_text(table_order)
    expect(drawing.locator(".anatomy-legend li")).to_have_count(len(table_order))
    assert page.errors == []


def test_no_horizontal_scroll_on_phone(browser, server):
    context = browser.new_context(viewport={"width": 390, "height": 844})
    page = context.new_page()
    page.goto(server)
    page.wait_for_selector(".status-strip .pill")
    page.wait_for_timeout(1000)
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1")
    context.close()


def make_test_alert(page):
    """Gera um alerta de teste pela marcha alterada e espera ele chegar à aba Alertas."""
    expect(page.get_by_test_id("kpi-level")).not_to_have_text("–")
    page.get_by_role("button", name=re.compile("Tronco 20")).click()
    expect(page.get_by_test_id("gait-result")).to_contain_text("Fora da referência")
    expect(page.locator(".badge-count")).to_be_visible()


def test_clear_alerts_is_visible_and_explains_itself(page):
    make_test_alert(page)
    page.get_by_role("tab", name=re.compile("Alertas")).click()
    # O usuário vê, sem procurar, que pode limpar e o que isso afeta.
    expect(page.get_by_role("heading", name="Vai gravar uma demonstração?")).to_be_visible()
    expect(page.get_by_text("não apaga")).to_be_visible()
    expect(page.get_by_test_id("open-clear")).to_be_visible()


def test_clear_alerts_cancel_keeps_everything(page):
    make_test_alert(page)
    page.get_by_role("tab", name=re.compile("Alertas")).click()
    total = alert_total(page)
    page.get_by_test_id("open-clear").click()
    dialog = page.get_by_test_id("clear-dialog")
    expect(dialog).to_be_visible()
    expect(dialog).to_contain_text("não pode ser desfeita")
    page.get_by_test_id("clear-cancel").click()
    expect(dialog).not_to_be_visible()
    assert alert_total(page) == total and total >= 1


def test_clear_alerts_removes_alerts_and_forgets_results(page):
    make_test_alert(page)
    page.get_by_role("tab", name=re.compile("Alertas")).click()
    page.get_by_test_id("open-clear").click()
    expect(page.get_by_test_id("clear-dialog")).to_contain_text("Esquecer também a marcha e a fala")
    page.get_by_test_id("clear-confirm").click()

    expect(page.get_by_test_id("clear-result")).to_contain_text("apagado")
    expect(page.get_by_test_id("alert-card")).to_have_count(0)
    expect(page.locator(".badge-count")).to_have_count(0)
    assert alert_total(page) == 0
    # O resultado da marcha também foi esquecido, e o risco voltou ao estado inicial.
    page.get_by_role("tab", name=re.compile("Internação")).click()
    expect(page.get_by_test_id("kpi-level")).not_to_have_text("–")
    expect(page.get_by_test_id("gait-result")).to_have_count(0)


def test_clear_alerts_can_keep_analysis_results(page):
    make_test_alert(page)
    page.get_by_role("tab", name=re.compile("Alertas")).click()
    page.get_by_test_id("open-clear").click()
    page.get_by_label("Esquecer também a marcha e a fala").uncheck()
    page.get_by_test_id("clear-confirm").click()
    expect(page.get_by_test_id("alert-card")).to_have_count(0)
    page.get_by_role("tab", name=re.compile("Internação")).click()
    expect(page.get_by_test_id("gait-result")).to_contain_text("Fora da referência")


def test_clear_alerts_is_hidden_when_the_server_disables_it(page):
    page.route("**/api/v1/status", lambda route: route.fulfill(
        json={**route.fetch().json(), "limpeza_permitida": False}))
    page.reload()
    page.wait_for_selector(".status-strip .pill")
    page.get_by_role("tab", name=re.compile("Alertas")).click()
    expect(page.get_by_role("heading", name="Histórico de alertas")).to_be_visible()
    expect(page.get_by_test_id("open-clear")).to_have_count(0)
