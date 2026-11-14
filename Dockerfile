# Imagem do OmniCare AI. Build a partir da raiz do repositório: docker build -t omnicare .
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Bibliotecas de sistema: OpenCV/MediaPipe (libgl1, libglib2.0-0), áudio (libsndfile1),
# PyTorch (libgomp1) e o SDK do Azure Speech (libasound2, ca-certificates).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libsndfile1 libgomp1 libasound2 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/backend

# PyTorch só para CPU: a versão padrão traz CUDA e passa de 2 GB. Camada separada para aproveitar o cache.
RUN pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu

COPY backend/requirements.txt .
RUN pip install -r requirements.txt

# O config.py espera backend/ e frontend/ lado a lado.
COPY backend/ /app/backend/
COPY frontend/ /app/frontend/

# Alertas gravados fora da imagem, em /home (persistente no App Service). Sobrescreva com OMNICARE_ALERTS_DB.
ENV OMNICARE_ALERTS_DB=/home/data/alertas.db
RUN mkdir -p /home/data

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
