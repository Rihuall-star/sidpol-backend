# SIDPOL · imagen de producción
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Usuario sin privilegios: si alguien compromete la app, no es root dentro del contenedor
RUN useradd --create-home --uid 10001 sidpol

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY --chown=sidpol:sidpol app.py ./
COPY --chown=sidpol:sidpol sidpol ./sidpol
COPY --chown=sidpol:sidpol templates ./templates
COPY --chown=sidpol:sidpol static ./static

USER sidpol
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

# 1 proceso con hilos: el plan gratuito tiene 512 MB y los modelos (pandas, statsmodels) pesan
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8000", "--workers", "1", "--threads", "4", "--timeout", "120"]
