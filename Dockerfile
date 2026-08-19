FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.docker.txt .

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# GPU-enabled PyTorch runtime for deployed inference
RUN pip install --no-cache-dir \
    torch==2.6.0 \
    torchvision==0.21.0 \
    torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

RUN pip install --no-cache-dir -r requirements.docker.txt

COPY . .

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

CMD [
    "gunicorn",
    "-k",
    "uvicorn.workers.UvicornWorker",
    "app.api.voice_stream_server:app",
    "--bind",
    "0.0.0.0:8000",
    "--workers",
    "2"
]