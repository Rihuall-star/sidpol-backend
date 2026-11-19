"""
SIDPOL: análisis y proyección delictiva del Perú con Machine Learning e IA generativa.

Arquitectura por capas:
    blueprints/  rutas HTTP (auth, secciones, ia, admin, salud)
    servicios/   lógica: analítica, modelos, resultados precalculados, IA y herramientas
    repositorio  único acceso a MongoDB
"""
import logging
from datetime import datetime

from dotenv import load_dotenv
from flask import Flask, session
from flask_login import UserMixin, current_user
from werkzeug.middleware.proxy_fix import ProxyFix

from sidpol import config as configuracion
from sidpol import registro, seguridad
from sidpol.extensiones import csrf, limiter, login_manager
from sidpol.repositorio import Repositorio

log = logging.getLogger(__name__)


class Usuario(UserMixin):
    def __init__(self, datos):
        self.id = datos["username"]          # Flask-Login usa 'id' como identificador
        self.username = datos["username"]
        self.rol = datos.get("rol", "invitado")


def _conectar(cfg):
    from pymongo import MongoClient
    cliente = MongoClient(cfg["MONGO_URI"], serverSelectionTimeoutMS=8000, appname="sidpol")
    return cliente[cfg["MONGO_DB"]]


def create_app(sobrescribir=None, db=None):
    """Crea la aplicación. `db` permite inyectar una base (p. ej. mongomock en las pruebas)."""
    load_dotenv()
    registro.configurar_logs()
    cfg = configuracion.cargar(sobrescribir)

    app = Flask(__name__, template_folder="../templates", static_folder="../static")
    app.config.update(cfg)
    # IP real del visitante detrás del proxy de Render (para auditoría y rate limiting)
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

    repo = Repositorio(db if db is not None else _conectar(cfg))
    app.extensions["sidpol_repo"] = repo

    csrf.init_app(app)
    limiter.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = "auth.login"

    @login_manager.user_loader
    def cargar_usuario(username):
        datos = repo.buscar_usuario(username)
        return Usuario(datos) if datos else None

    registro.registrar(app)
    seguridad.registrar(app)

    @app.before_request
    def marcar_actividad():
        """Radar de usuarios en línea. Escribe como máximo una vez por minuto por sesión."""
        if not current_user.is_authenticated:
            return
        ahora = datetime.now().timestamp()
        if ahora - session.get("visto", 0) > 60:
            try:
                repo.marcar_actividad(current_user.username)
                session["visto"] = ahora
            except Exception as e:  # noqa: BLE001
                log.warning("no se pudo actualizar last_seen: %s", e)

    from sidpol.blueprints import admin, auth, ia, salud, secciones
    for bp in (auth.bp, secciones.bp, ia.bp, admin.bp, salud.bp):
        app.register_blueprint(bp)

    from sidpol import precalculo
    precalculo.registrar_comando(app)
    return app


__all__ = ["create_app", "Usuario"]
