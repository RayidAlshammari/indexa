# Indexa service: assessment engine + demo page. No torch, no vector DB, no auth provider.
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src INDEXA_DATA_DIR=/data
WORKDIR /app
RUN pip install --no-cache-dir "fastapi>=0.115" "uvicorn[standard]>=0.30" "httpx>=0.27" "pdfplumber>=0.11" "python-multipart>=0.0.9"
COPY services/ai-service/src/indexa /app/src/indexa
COPY services/ai-service/config /app/config
RUN useradd -m indexa && mkdir -p /data && chown indexa /data
USER indexa
EXPOSE 8080
CMD ["sh", "-c", "uvicorn indexa.server:app --host 0.0.0.0 --port ${PORT:-8080}"]
