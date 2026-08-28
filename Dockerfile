FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ARG WITH_PYANNOTE=false

ARG WITH_DEV=false

COPY requirements.txt requirements-pyannote.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && if [ "$WITH_PYANNOTE" = "true" ]; then \
        pip install --no-cache-dir -r requirements-pyannote.txt; \
    fi \
    && if [ "$WITH_DEV" = "true" ]; then \
        pip install --no-cache-dir -r requirements-dev.txt; \
    fi

COPY . .

EXPOSE 8080

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
