"""
Registro estructurado (JSON por línea) con un identificador por petición.

Cada línea se puede filtrar en los logs de Render o enviar a un SIEM sin parsear texto libre:
{"ts": "...", "nivel": "INFO", "msg": "peticion", "request_id": "...", "ruta": "/", "estado": 200, "ms": 42}
"""
import json
import logging
import sys
import time
import uuid
from datetime import datetime, timezone

from flask import g, has_request_context, request

CAMPOS_ESTANDAR = set(vars(logging.makeLogRecord({}))) | {"message", "asctime"}


class FormatoJSON(logging.Formatter):
    def format(self, record):
        datos = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"),
            "nivel": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if has_request_context():
            datos["request_id"] = g.get("request_id")
        datos.update({k: v for k, v in vars(record).items() if k not in CAMPOS_ESTANDAR})
        if record.exc_info:
            datos["excepcion"] = self.formatException(record.exc_info)
        return json.dumps(datos, ensure_ascii=False, default=str)


def configurar_logs(nivel=logging.INFO):
    raiz = logging.getLogger()
    if any(isinstance(h.formatter, FormatoJSON) for h in raiz.handlers):
        return
    manejador = logging.StreamHandler(sys.stdout)
    manejador.setFormatter(FormatoJSON())
    raiz.handlers[:] = [manejador]
    raiz.setLevel(nivel)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)  # evita duplicar el log de acceso


def registrar(app):
    log = logging.getLogger("sidpol.acceso")

    @app.before_request
    def inicio_peticion():
        entrante = request.headers.get("X-Request-ID", "")
        g.request_id = entrante[:64] if entrante.replace("-", "").isalnum() else uuid.uuid4().hex[:16]
        g.inicio = time.perf_counter()

    @app.after_request
    def fin_peticion(resp):
        resp.headers["X-Request-ID"] = g.get("request_id", "")
        if not request.path.startswith("/static/"):
            log.info("peticion", extra={
                "metodo": request.method, "ruta": request.path, "estado": resp.status_code,
                "ms": round((time.perf_counter() - g.get("inicio", time.perf_counter())) * 1000, 1),
            })
        return resp
