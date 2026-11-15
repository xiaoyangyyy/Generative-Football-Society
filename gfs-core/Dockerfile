ARG PYTHON_IMAGE=python:3.12.11-slim-bookworm@sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7
FROM ${PYTHON_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    GFS_PROJECT_ROOT=/app

RUN groupadd --system --gid 10001 gfs \
    && useradd --system --uid 10001 --gid gfs --home-dir /nonexistent --shell /usr/sbin/nologin gfs
WORKDIR /app
COPY pyproject.toml requirements.txt ./
COPY src ./src
RUN python -m pip install -r requirements.txt \
    && python -m pip install --no-deps .
COPY data ./data
RUN mkdir -p data/persistence backups && chown -R gfs:gfs data/persistence backups
USER 10001:10001
EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/health', timeout=3).read()"]
ENTRYPOINT ["gfs-core"]
CMD ["status", "--top", "20"]
