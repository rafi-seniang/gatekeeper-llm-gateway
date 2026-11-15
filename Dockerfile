# syntax=docker/dockerfile:1.7
# Multi-stage: build the app's wheel once, ship a slim CPU image that runs as a non-root user.
FROM python:3.12-slim AS build
WORKDIR /src
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
COPY pyproject.toml ./
COPY main.py ./
COPY app ./app
COPY config ./config
RUN pip wheel --no-deps --wheel-dir /wheels .

FROM python:3.12-slim AS runtime
ARG INCLUDE_LOCAL_MODELS=false
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8020 DEVICE=cpu
# Install the CPU build of PyTorch and transformers only when a local DeBERTa guard is baked in
RUN if [ "$INCLUDE_LOCAL_MODELS" = "true" ]; then \
      pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu && \
      pip install --no-cache-dir "transformers>=5.0"; \
    fi
RUN useradd --create-home --uid 10001 app
RUN mkdir -p /home/app/.cache/huggingface && chown -R app:app /home/app/.cache
COPY --from=build /wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels
USER app
WORKDIR /home/app
EXPOSE 8020
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health', timeout=3)"
# One worker: the classifier is loaded once per process and every request uses it
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --no-access-log"]
