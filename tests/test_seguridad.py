"""Pruebas de los controles de seguridad (OWASP Top 10) de SIDPOL."""
import re

from conftest import USUARIOS, csrf_de, login

from sidpol.seguridad import formato_ia
from sidpol.servicios import ia


# ---------- A05: cabeceras de seguridad y CSP con nonce ----------
def test_cabeceras_de_seguridad(client):
    r = client.get("/login")
    assert r.status_code == 200
    csp = r.headers["Content-Security-Policy"]
    assert "frame-ancestors 'none'" in csp and "object-src 'none'" in csp
    assert r.headers["X-Frame-Options"] == "DENY"
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    assert "Referrer-Policy" in r.headers
    assert "Permissions-Policy" in r.headers


def test_csp_sin_unsafe_inline_en_scripts(client):
    r = client.get("/login")
    script_src = next(d for d in r.headers["Content-Security-Policy"].split(";") if "script-src" in d)
    assert "'unsafe-inline'" not in script_src
    nonce = re.search(r"'nonce-([^']+)'", script_src).group(1)
    html = r.get_data(as_text=True)
    # Todo <script> en línea de la página lleva el nonce de esta respuesta
    en_linea = re.findall(r"<script(?![^>]*\bsrc=)([^>]*)>", html)
    assert en_linea and all(f'nonce="{nonce}"' in attrs for attrs in en_linea)


def test_nonce_cambia_en_cada_peticion(client):
    a = client.get("/login").headers["Content-Security-Policy"]
    b = client.get("/login").headers["Content-Security-Policy"]
    assert a != b


def test_plantillas_sin_manejadores_en_linea():
    import pathlib
    for p in pathlib.Path("templates").glob("*.html"):
        assert not re.search(r"\son[a-z]+=", p.read_text(encoding="utf-8")), p.name


def test_cookie_de_sesion_endurecida(client):
    r = login(client, "demo", USUARIOS["demo"][0])
    cookie = r.headers.get("Set-Cookie", "")
    assert "HttpOnly" in cookie
    assert "SameSite=Lax" in cookie


# ---------- A01: CSRF ----------
def test_login_sin_token_csrf_es_rechazado(client):
    r = client.post("/login", data={"username": "demo", "password": USUARIOS["demo"][0]})
    assert r.status_code == 302 and r.headers["Location"].endswith("/login")
    with client.session_transaction() as s:
        assert "user" not in s


def test_chat_sin_token_csrf_es_rechazado(client):
    login(client, "demo", USUARIOS["demo"][0])
    r = client.post("/chat-ia", data={"mensaje": "hola"})
    assert r.status_code == 400


def test_csrf_no_redirige_a_sitios_externos(client):
    """Si el token CSRF falla, solo se vuelve a una página del mismo sitio (CodeQL: open redirect)."""
    r = client.post("/login", data={"username": "x", "password": "y"},
                    headers={"Referer": "https://sitio-malicioso.com/phishing"})
    assert r.status_code == 302
    assert "sitio-malicioso" not in r.headers["Location"]
    r = client.post("/login", data={"username": "x", "password": "y"},
                    headers={"Referer": "http://localhost/delitos"})
    assert r.headers["Location"].endswith("/delitos")


# ---------- A07: autenticación ----------
def test_login_correcto(client):
    r = login(client, "demo", USUARIOS["demo"][0])
    assert r.status_code == 302 and r.headers["Location"].endswith("/")
    with client.session_transaction() as s:
        assert s["rol"] == "invitado"


def test_login_incorrecto_queda_auditado(client, base_limpia):
    r = login(client, "demo", "clave-equivocada")
    assert r.headers["Location"].endswith("/login")
    evento = base_limpia["auditoria"].find_one({"evento": "Intento fallido"})
    assert evento is not None and evento["usuario"] == "demo"


def test_fuerza_bruta_bloqueada_tras_5_intentos(client, base_limpia):
    for _ in range(5):
        login(client, "demo", "mal")
    login(client, "demo", "mal")  # el 6.º intento debe ser bloqueado por el rate limiter
    assert base_limpia["auditoria"].count_documents({"evento": "Intento fallido"}) == 5
    assert "Demasiados intentos" in client.get("/login").get_data(as_text=True)


def test_sesion_nueva_en_cada_login(client):
    with client.session_transaction() as s:
        s["dato_previo"] = "atacante"
    login(client, "demo", USUARIOS["demo"][0])
    with client.session_transaction() as s:
        assert "dato_previo" not in s  # protección contra fijación de sesión


# ---------- A01: control de acceso por rol ----------
def test_invitado_no_accede_al_monitor(client):
    login(client, "demo", USUARIOS["demo"][0])
    assert client.get("/usuarios-activos").status_code == 302


def test_invitado_no_ve_botones_de_admin(client):
    login(client, "demo", USUARIOS["demo"][0])
    html = client.get("/").get_data(as_text=True)
    assert "Crear Usuario" not in html
    assert "/usuarios-activos" not in html


def test_admin_accede_al_monitor_sin_ver_hashes(client):
    login(client, "admin_test", USUARIOS["admin_test"][0])
    r = client.get("/usuarios-activos")
    assert r.status_code == 200
    html = r.get_data(as_text=True)
    assert "pbkdf2" not in html and "scrypt" not in html


def test_admin_no_crea_usuarios_con_clave_corta(client, base_limpia):
    login(client, "admin_test", USUARIOS["admin_test"][0])
    token = csrf_de(client.get("/crear_usuario").get_data(as_text=True))
    client.post("/crear_usuario", data={"username": "nuevo", "password": "corta", "rol": "admin",
                                        "csrf_token": token})
    assert base_limpia["usuarios"].find_one({"username": "nuevo"}) is None


def test_rutas_protegidas_requieren_login(client):
    for ruta in ["/", "/cluster-departamentos", "/prediccion-2026", "/crear_usuario", "/riesgo-modalidad"]:
        r = client.get(ruta)
        assert r.status_code == 302 and "/login" in r.headers["Location"], ruta


# ---------- A04: límite diario de IA para la cuenta demo ----------
def test_limite_diario_de_ia_para_invitado(repo, monkeypatch):
    monkeypatch.setattr(ia, "LIMITE_IA_INVITADO", 3)
    usos = [ia.cuota_ia_agotada(repo, "demo", "invitado") for _ in range(5)]
    assert usos == [False, False, False, True, True]
    assert not any(ia.cuota_ia_agotada(repo, "admin_test", "admin") for _ in range(5))  # admin sin límite


def test_formato_ia_escapa_html_y_da_formato():
    salida = str(formato_ia("**Alerta**\n* uno\n<script>alert(1)</script>"))
    assert "<strong>Alerta</strong>" in salida
    assert "• uno" in salida
    assert "<script>" not in salida and "&lt;script&gt;" in salida


def test_chat_no_inserta_html_en_el_navegador():
    js = open("static/js/sidpol-chat.js", encoding="utf-8").read()
    assert "innerHTML" not in js and "insertAdjacentHTML" not in js   # todo el texto va por textContent
    assert "document.createTextNode(msg)" in js


# ---------- Operación ----------
def test_health_responde_sin_sesion_y_sin_secretos(client):
    r = client.get("/health")
    assert r.status_code == 200
    datos = r.get_json()
    assert datos["estado"] == "ok" and datos["base_de_datos"] == "ok"
    assert not any(k in r.get_data(as_text=True).lower() for k in ("mongodb://", "secret", "api_key"))


def test_cada_respuesta_lleva_request_id(client):
    r = client.get("/login")
    assert len(r.headers["X-Request-ID"]) >= 8
    r = client.get("/login", headers={"X-Request-ID": "abc123-def"})
    assert r.headers["X-Request-ID"] == "abc123-def"
    r = client.get("/login", headers={"X-Request-ID": "<script>"})
    assert r.headers["X-Request-ID"] != "<script>"  # no se reflejan valores con caracteres extraños
