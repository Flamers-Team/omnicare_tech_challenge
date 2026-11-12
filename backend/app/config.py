"""Caminhos, limites e constantes da aplicação."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # pasta backend/
FRONTEND_DIR = BASE_DIR.parent / "frontend"
UPLOAD_DIR = BASE_DIR / "temp_uploads"
MODELS_DIR = BASE_DIR / "models"
SAMPLE_DIR = BASE_DIR / "sample_data"
ALERTS_DB = Path(os.getenv("OMNICARE_ALERTS_DB") or BASE_DIR / "dados_locais" / "alertas.db")

GAIT_EXAMPLES = {
    "normal": SAMPLE_DIR / "marcha" / "caminhada_normal.csv.gz",
    "alterada": SAMPLE_DIR / "marcha" / "caminhada_tronco_20_graus.csv.gz",
}

GAIT_CSV = {".csv"}
GAIT_VIDEO = {".mp4", ".avi", ".mov", ".mkv"}
AUDIO_TYPES = {".wav"}
MAX_CSV_BYTES = 100 * 1024 * 1024
MAX_VIDEO_BYTES = 500 * 1024 * 1024
MAX_AUDIO_BYTES = 50 * 1024 * 1024
ALERT_STREAM_SECONDS = 1.0

# A base tem licença restrita e fica fora do projeto. Em outro computador, informe o caminho em ENDOSCAPES_ZIP.
ENDOSCAPES_ZIP = os.getenv("ENDOSCAPES_ZIP") or Path.home() / "Downloads" / "endoscapes_omnicare.zip"
