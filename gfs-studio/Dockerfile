ARG CORE_IMAGE=gfs-core:7.0.0a1
FROM ${CORE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    GFS_PROJECT_ROOT=/app

USER root
WORKDIR /app
COPY pyproject.toml README.md requirements.txt requirements-linux-py312.lock ./
COPY src ./src
RUN python -m pip install --require-hashes -r requirements-linux-py312.lock \
    && python -m pip install --no-deps .
COPY data ./data
COPY evidence ./evidence
COPY supply-chain ./supply-chain
RUN mkdir -p data/persistence backups && chown -R gfs:gfs data/persistence backups
USER 10001:10001
EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import json,urllib.request; payload=json.load(urllib.request.urlopen('http://127.0.0.1:8765/health', timeout=3)); assert payload['status'] == 'ok' and payload['transport'] == 'loopback'"]
ENTRYPOINT ["gfs-studio"]
CMD ["--base-dir", "/app", "web"]
