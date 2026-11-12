from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import config
from app.routers import alertas, cirurgia, marcha, pacientes, status, voz

app = FastAPI(title="OmniCare AI - Tech Challenge Fase 4")

for router in (status.router, pacientes.router, marcha.router, voz.router, cirurgia.router, alertas.router):
    app.include_router(router)


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(config.FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=config.FRONTEND_DIR), name="static")
