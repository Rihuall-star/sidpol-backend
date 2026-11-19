"""Extensiones de Flask compartidas por todos los blueprints (se inicializan en create_app)."""
from flask import current_app
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_login import LoginManager
from flask_wtf.csrf import CSRFProtect

csrf = CSRFProtect()
limiter = Limiter(get_remote_address, default_limits=[])
login_manager = LoginManager()


def repo():
    """Repositorio de datos de la aplicación activa."""
    return current_app.extensions["sidpol_repo"]
