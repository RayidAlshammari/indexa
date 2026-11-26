# Hugging Face Space (Docker SDK) for Indexa.
# The Space holds only this file and README.md; the code is pulled from GitHub at build time,
# so a "Factory rebuild" in the Space settings deploys the latest commit on main.
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src INDEXA_DATA_DIR=/data PORT=7860
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "fastapi>=0.115" "uvicorn[standard]>=0.30" "httpx>=0.27" "pdfplumber>=0.11" "python-multipart>=0.0.9"
# Changes whenever main moves, so the clone below is never served from a stale cache layer.
ADD https://api.github.com/repos/RayidAlshammari/indexa/git/refs/heads/main /tmp/indexa-ref.json
RUN git clone --depth 1 https://github.com/RayidAlshammari/indexa /tmp/indexa \
    && mkdir -p /app/src \
    && cp -r /tmp/indexa/services/ai-service/src/indexa /app/src/indexa \
    && cp -r /tmp/indexa/services/ai-service/config /app/config \
    && rm -rf /tmp/indexa
# Spaces run the container as uid 1000.
RUN useradd -m -u 1000 indexa && mkdir -p /data && chown -R indexa /data /app
USER indexa
WORKDIR /app
EXPOSE 7860
CMD ["sh", "-c", "uvicorn indexa.server:app --host 0.0.0.0 --port ${PORT:-7860}"]
