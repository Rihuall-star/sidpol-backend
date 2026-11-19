"""Configuración de SIDPOL leída desde variables de entorno (nunca desde el código)."""
import os
from datetime import timedelta


def cargar(sobrescribir=None):
    """Devuelve la configuración de Flask. `sobrescribir` permite ajustar valores en las pruebas."""
    en_produccion = bool(os.getenv("RENDER"))  # Render define RENDER=true
    config = {
        "SECRET_KEY": os.environ.get("SECRET_KEY"),
        "MONGO_URI": os.getenv("MONGO_URI", "mongodb://localhost:27017/"),
        "MONGO_DB": os.getenv("MONGO_DB", "denuncias_db"),
        "EN_PRODUCCION": en_produccion,
        "VERSION_CODIGO": (os.getenv("RENDER_GIT_COMMIT") or "local")[:7],
        # Cookies de sesión endurecidas (OWASP A07)
        "SESSION_COOKIE_HTTPONLY": True,
        "SESSION_COOKIE_SAMESITE": "Lax",
        "SESSION_COOKIE_SECURE": en_produccion,
        "REMEMBER_COOKIE_HTTPONLY": True,
        "REMEMBER_COOKIE_SECURE": en_produccion,
        "PERMANENT_SESSION_LIFETIME": timedelta(hours=8),
        "WTF_CSRF_TIME_LIMIT": None,  # el token dura lo que dure la sesión
        "RATELIMIT_STORAGE_URI": os.getenv("RATELIMIT_STORAGE_URI", "memory://"),
    }
    config.update(sobrescribir or {})
    if not config["SECRET_KEY"]:
        raise RuntimeError("Falta la variable de entorno SECRET_KEY: la aplicación no arranca sin ella.")
    return config
