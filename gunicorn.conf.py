"""
Configuración de gunicorn. Se carga sola si gunicorn arranca desde esta carpeta
(Render con `gunicorn app:app` y el contenedor Docker).

El valor por defecto de gunicorn corta las peticiones a los 30 s; una consulta al chat con
herramientas puede hacer varias llamadas a Gemini, así que se da más margen.
"""
import os

bind = f"0.0.0.0:{os.getenv('PORT', '8000')}"
workers = int(os.getenv("WEB_CONCURRENCY", "1"))   # 512 MB en el plan gratuito: un proceso
threads = int(os.getenv("GUNICORN_THREADS", "4"))  # varias visitas a la vez sin duplicar memoria
timeout = 150
graceful_timeout = 30
keepalive = 5
accesslog = None  # el registro de acceso lo hace la aplicación (JSON con X-Request-ID)
