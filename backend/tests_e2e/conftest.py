"""Testes de ponta a ponta no navegador. Sobem o servidor numa porta livre, com banco de alertas temporário e sem chaves do Azure."""
import io
import os
import socket
import subprocess
import sys
import time
import urllib.request
import wave
from pathlib import Path

import numpy as np
import pytest
from playwright.sync_api import expect, sync_playwright

expect.set_options(timeout=10_000)
BACKEND = Path(__file__).resolve().parent.parent


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="session")
def server(tmp_path_factory):
    port = free_port()
    env = {
        **os.environ,
        "OMNICARE_ALERTS_DB": str(tmp_path_factory.mktemp("db") / "alertas.db"),
        # Variáveis vazias valem mais que o .env: os testes não chamam o Azure.
        "AZURE_SPEECH_KEY": "",
        "AZURE_LANGUAGE_KEY": "",
    }
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(port)],
        cwd=BACKEND, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}"
    for _ in range(120):
        try:
            urllib.request.urlopen(f"{url}/api/v1/status", timeout=1)
            break
        except OSError:
            time.sleep(0.5)
    else:
        process.kill()
        raise RuntimeError("O servidor não subiu a tempo.")
    # A primeira chamada carrega os modelos; aquece antes para os testes não estourarem o tempo.
    for path in ("/api/v1/status", "/api/v1/cirurgia", "/api/v1/pacientes"):
        urllib.request.urlopen(f"{url}{path}", timeout=60).read()
    yield url
    process.terminate()
    process.wait(timeout=15)


@pytest.fixture(scope="session")
def browser():
    with sync_playwright() as playwright:
        # Local (Windows): usa o Edge instalado, sem baixar outro navegador.
        # No GitHub Actions: E2E_BROWSER_CHANNEL="" usa o Chromium do Playwright. O microfone é simulado pelo Chromium.
        instance = playwright.chromium.launch(
            channel=os.getenv("E2E_BROWSER_CHANNEL", "msedge") or None,
            args=["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"],
        )
        yield instance
        instance.close()


@pytest.fixture()
def page(browser, server):
    # Cada teste começa sem alertas nem análises guardadas, porque o servidor é compartilhado entre eles.
    urllib.request.urlopen(urllib.request.Request(f"{server}/api/v1/alertas?incluir_resultados=true", method="DELETE"), timeout=30).read()
    context = browser.new_context(viewport={"width": 1366, "height": 900}, permissions=["microphone"])
    page = context.new_page()
    page.errors = []
    page.on("pageerror", lambda error: page.errors.append(str(error)))
    page.on("console", lambda message: page.errors.append(message.text) if message.type == "error" else None)
    page.goto(server)
    page.wait_for_selector(".status-strip .pill")
    yield page
    context.close()


def make_wav(path: Path, seconds: float = 3.0) -> Path:
    samples = (np.sin(np.arange(int(16000 * seconds)) * 0.1) * 8000).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(samples.tobytes())
    path.write_bytes(buffer.getvalue())
    return path
