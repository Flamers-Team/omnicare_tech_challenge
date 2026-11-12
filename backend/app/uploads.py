import uuid
from pathlib import Path

from fastapi import HTTPException, UploadFile

from app import config


def save_upload(upload: UploadFile, allowed: set[str], max_bytes: int) -> Path:
    """Salva o envio com um nome gerado aqui, sem usar o caminho que veio do navegador."""
    extension = Path(upload.filename or "").suffix.lower()
    if extension not in allowed:
        raise HTTPException(400, f"Tipo de arquivo não aceito. Use: {', '.join(sorted(allowed))}.")
    config.UPLOAD_DIR.mkdir(exist_ok=True)
    target = config.UPLOAD_DIR / f"{uuid.uuid4().hex}{extension}"
    size = 0
    try:
        with target.open("wb") as output:
            while chunk := upload.file.read(1024 * 1024):
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(413, f"Arquivo maior que o limite de {max_bytes // (1024 * 1024)} MB.")
                output.write(chunk)
    except HTTPException:
        target.unlink(missing_ok=True)
        raise
    return target
