"""
Configuración de pruebas de SIDPOL.

- La aplicación se crea con create_app() y una base en memoria (mongomock): no toca Atlas.
- No llama a Gemini: el cliente se reemplaza por uno falso en las pruebas que lo necesitan.
"""
import os
import re
import sys

import mongomock
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Variables mínimas para que la app arranque (antes de importarla)
os.environ["SECRET_KEY"] = "clave-solo-para-pruebas"
# Vacía (no borrada): así load_dotenv() no la vuelve a cargar desde .env y ninguna prueba llama a Gemini
os.environ["GEMINI_API_KEY"] = ""
os.environ.pop("RENDER", None)

from werkzeug.security import generate_password_hash  # noqa: E402

from sidpol import create_app  # noqa: E402
from sidpol.extensiones import limiter  # noqa: E402
from sidpol.servicios import ia, resultados  # noqa: E402

USUARIOS = {
    "admin_test": ("Admin-Prueba-2026", "admin"),
    "demo": ("SidpolDemo2026", "invitado"),
}

DB = mongomock.MongoClient()["sidpol_pruebas"]
APP = create_app({"TESTING": True}, db=DB)
REPO = APP.extensions["sidpol_repo"]


@pytest.fixture(autouse=True)
def base_limpia():
    """Cada prueba empieza con usuarios conocidos, sin resultados guardados y con los límites reiniciados."""
    for nombre in DB.list_collection_names():
        DB[nombre].delete_many({})
    for usuario, (clave, rol) in USUARIOS.items():
        DB["usuarios"].insert_one({"username": usuario, "password": generate_password_hash(clave), "rol": rol})
    limiter.reset()
    resultados.olvidar()
    REPO.olvidar_memoria()
    ia._cliente = None
    yield DB


@pytest.fixture
def repo():
    return REPO


@pytest.fixture
def client():
    return APP.test_client()


def csrf_de(html):
    """Extrae el token CSRF del formulario o de la etiqueta <meta>."""
    m = re.search(r'name="csrf_token" value="([^"]+)"', html) or re.search(
        r'name="csrf-token" content="([^"]+)"', html
    )
    return m.group(1) if m else None


def login(client, usuario, clave):
    token = csrf_de(client.get("/login").get_data(as_text=True))
    return client.post("/login", data={"username": usuario, "password": clave, "csrf_token": token})


# ---------- Datos de ejemplo ----------
MODALIDADES = ["Extorsión", "Homicidio", "Estafa", "Hurto", "Robo",
               "Violencia contra la mujer e integrantes", "Otros"]
DISTRITOS = [("TRUJILLO", "LA LIBERTAD"), ("SAN JUAN DE LURIGANCHO", "LIMA METROPOLITANA"),
             ("SULLANA", "PIURA"), ("AREQUIPA", "AREQUIPA"), ("CUSCO", "CUSCO")]


def documentos_completos():
    """Dataset pequeño con todos los campos reales: 2018-2025 (hasta octubre)."""
    docs = []
    for anio in range(2018, 2026):
        for mes in range(1, 13):
            if anio == 2025 and mes > 10:
                break
            for i, mod in enumerate(MODALIDADES):
                for j, (dist, dpto) in enumerate(DISTRITOS):
                    base = 50 + 10 * i + 5 * j
                    if mod == "Extorsión":
                        base = (5 + (anio - 2018) * 12 * (5 - j)) * (1.5 if anio == 2025 else 1)  # crece y se concentra
                    docs.append({"ANIO": anio, "MES": mes, "trimestre": f"T{(mes - 1) // 3 + 1}",
                                 "anio_trimestre": f"{anio}-T{(mes - 1) // 3 + 1}",
                                 "DPTO_HECHO_NEW": dpto, "PROV_HECHO": dist, "DIST_HECHO": dist,
                                 "P_MODALIDADES": mod, "cantidad": int(base)})
    return docs


@pytest.fixture
def datos_completos(base_limpia):
    col = base_limpia["denuncias"]
    col.insert_many(documentos_completos())
    REPO.olvidar_memoria()
    yield col


class RespuestaFalsa:
    def __init__(self, texto=None, llamadas=None):
        from google.genai import types
        self.text = texto
        self.function_calls = [types.FunctionCall(name=n, args=a) for n, a in (llamadas or [])]
        partes = [types.Part.from_function_call(name=n, args=a) for n, a in (llamadas or [])] or \
                 [types.Part.from_text(text=texto or "")]
        self.candidates = [type("C", (), {"content": types.Content(role="model", parts=partes)})()]


class ClienteFalso:
    """Simula el cliente de Gemini. `guion` es la lista de respuestas que devolverá en orden;
    los modelos de `fallan` responden 429 (cuota agotada)."""
    def __init__(self, guion=None, fallan=()):
        self.guion, self.fallan = list(guion or []), set(fallan)
        self.llamados, self.configs, self.contenidos = [], [], []
        self.models = self

    def generate_content(self, model, contents, config=None):
        self.llamados.append(model)
        self.configs.append(config)
        self.contenidos.append(contents)
        if model in self.fallan:
            raise RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded")
        return self.guion.pop(0) if self.guion else RespuestaFalsa(f"ok desde {model}")


@pytest.fixture
def gemini_falso(monkeypatch):
    def crear(guion=None, fallan=()):
        falso = ClienteFalso(guion, fallan)
        monkeypatch.setattr(ia, "_cliente", falso)
        return falso
    return crear


@pytest.fixture(autouse=True)
def sin_red(monkeypatch):
    """Red de seguridad: si algún código intenta salir a internet durante las pruebas, falla."""
    import socket

    def bloqueado(*args, **kwargs):
        raise RuntimeError("Las pruebas no deben conectarse a internet")
    monkeypatch.setattr(socket.socket, "connect", bloqueado)
