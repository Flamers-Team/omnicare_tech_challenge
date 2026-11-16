from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import config
from app.routers import alertas, cirurgia, marcha, pacientes, status, voz

app = FastAPI(title="OmniCare AI - Tech Challenge Fase 4")

for router in (status.router, pacientes.router, marcha.router, voz.router, cirurgia.router, alertas.router):
    app.include_router(router)

# Os arquivos da tela não têm versão no nome. Sem este aviso, o navegador pode continuar usando
# uma cópia antiga depois de uma atualização. "no-cache" faz ele conferir antes de usar; se nada
# mudou, o servidor responde só "não mudou" (304), sem reenviar o arquivo.
NO_CACHE = {"Cache-Control": "no-cache"}


class FrontendFiles(StaticFiles):
    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers.update(NO_CACHE)
        return response


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(config.FRONTEND_DIR / "index.html", headers=NO_CACHE)


app.mount("/static", FrontendFiles(directory=config.FRONTEND_DIR), name="static")
