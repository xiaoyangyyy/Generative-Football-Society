FROM gfs-core:latest

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    GFS_PROJECT_ROOT=/app

WORKDIR /app
COPY pyproject.toml requirements.txt ./
COPY src ./src
RUN python -m pip install -r requirements.txt \
    && python -m pip install --no-deps .
COPY data ./data
COPY evidence ./evidence
RUN mkdir -p data/persistence backups && chown -R gfs:gfs data/persistence backups
USER 10001:10001
EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/health', timeout=3).read()"]
ENTRYPOINT ["gfs-studio"]
CMD ["web"]
