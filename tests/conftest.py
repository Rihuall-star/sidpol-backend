"""
Configuración de pruebas de SIDPOL.

- Reemplaza MongoDB por mongomock (base en memoria): las pruebas no tocan Atlas.
- No llama a Gemini: las funciones de IA se reemplazan por respuestas fijas.
"""
import os
import re
import sys

import mongomock
import pymongo
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Variables mínimas para que la app arranque (antes de importarla)
os.environ["SECRET_KEY"] = "clave-solo-para-pruebas"
os.environ["MONGO_URI"] = "mongodb://localhost:27017/"
os.environ.pop("GEMINI_API_KEY", None)
os.environ.pop("RENDER", None)

# Toda conexión a MongoDB usa la base en memoria
pymongo.MongoClient = mongomock.MongoClient

from werkzeug.security import generate_password_hash  # noqa: E402

import app as sidpol  # noqa: E402

USUARIOS = {
    "admin_test": ("Admin-Prueba-2026", "admin"),
    "demo": ("SidpolDemo2026", "invitado"),
}


@pytest.fixture(autouse=True)
def base_limpia():
    """Cada prueba empieza con usuarios conocidos, auditoría vacía y límites reiniciados."""
    db = sidpol.db
    for nombre in ("usuarios", "auditoria", "estadisticas"):
        db[nombre].delete_many({})
    for usuario, (clave, rol) in USUARIOS.items():
        db["usuarios"].insert_one(
            {"username": usuario, "password": generate_password_hash(clave), "rol": rol}
        )
    sidpol.limiter.reset()
    import analitica
    analitica.limpiar_cache()
    sidpol.app.config["TESTING"] = True
    yield db


@pytest.fixture
def client():
    return sidpol.app.test_client()


def csrf_de(html):
    """Extrae el token CSRF del formulario o de la etiqueta <meta>."""
    m = re.search(r'name="csrf_token" value="([^"]+)"', html) or re.search(
        r'name="csrf-token" content="([^"]+)"', html
    )
    return m.group(1) if m else None


def login(client, usuario, clave):
    token = csrf_de(client.get("/login").get_data(as_text=True))
    return client.post(
        "/login", data={"username": usuario, "password": clave, "csrf_token": token}
    )
