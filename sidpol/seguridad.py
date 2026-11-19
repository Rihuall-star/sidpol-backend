"""
Controles de seguridad transversales (OWASP Top 10).

- CSP con *nonce* por petición: solo se ejecutan los scripts propios de la página y los de
  los CDN permitidos. Sin 'unsafe-inline' en script-src, un XSS inyectado no se ejecuta.
- Cabeceras de seguridad, HSTS en producción.
- Formato seguro del texto de la IA y validación de redirecciones.
"""
import re
import secrets
from functools import wraps
from urllib.parse import urlparse

from flask import current_app, flash, g, jsonify, redirect, request, session, url_for
from flask_login import current_user
from flask_wtf.csrf import CSRFError
from markupsafe import Markup, escape
from werkzeug.exceptions import HTTPException
from werkzeug.routing import RequestRedirect

CDN_SCRIPTS = "https://cdn.jsdelivr.net https://cdnjs.cloudflare.com https://code.jquery.com"


def politica_csp(nonce):
    return "; ".join([
        "default-src 'self'",
        f"script-src 'self' 'nonce-{nonce}' {CDN_SCRIPTS}",
        # Los estilos en línea siguen permitidos: Bootstrap, Highcharts y Chart.js los usan
        # y no ejecutan código. El riesgo de XSS lo cubre script-src.
        "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com https://fonts.googleapis.com",
        "font-src 'self' https://fonts.gstatic.com https://cdnjs.cloudflare.com https://cdn.jsdelivr.net data:",
        "img-src 'self' data: blob:",
        "connect-src 'self'",
        "object-src 'none'",
        "base-uri 'self'",
        "form-action 'self'",
        "frame-ancestors 'none'",
    ])


def formato_ia(texto):
    """Da formato básico (negritas, viñetas, saltos de línea) al texto de la IA.
    Primero escapa todo el HTML, así una respuesta maliciosa no puede inyectar código (XSS)."""
    t = str(escape(texto or ""))
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"(?m)^\s*[\*\-]\s+", "• ", t)
    return Markup(t.replace("\n", "<br>"))


def url_segura(url):
    """Convierte una URL de origen (p. ej. el Referer) en una ruta de ESTA aplicación, o None.

    No se reutiliza el texto recibido: se busca a qué vista de la app corresponde la ruta y se
    reconstruye la URL con url_for. Así es imposible redirigir a otro sitio (open redirect)."""
    if not url:
        return None
    destino = urlparse(url)
    if destino.netloc and destino.netloc != request.host:
        return None
    try:
        endpoint, argumentos = current_app.url_map.bind(request.host).match(destino.path or "/", method="GET")
    except (HTTPException, RequestRedirect):
        return None
    return url_for(endpoint, **argumentos)


def admin_required(f):
    @wraps(f)
    def envoltura(*args, **kwargs):
        rol = session.get("rol") or (current_user.rol if current_user.is_authenticated else None)
        if rol != "admin":
            flash("⛔ Acceso denegado: se requieren privilegios de administrador.", "error")
            return redirect(url_for("secciones.index"))
        return f(*args, **kwargs)
    return envoltura


def registrar(app):
    en_produccion = app.config["EN_PRODUCCION"]

    @app.before_request
    def generar_nonce():
        g.csp_nonce = secrets.token_urlsafe(16)

    @app.context_processor
    def nonce_en_plantillas():
        return {"csp_nonce": lambda: g.get("csp_nonce", "")}

    @app.after_request
    def cabeceras_seguridad(resp):
        resp.headers["Content-Security-Policy"] = politica_csp(g.get("csp_nonce", secrets.token_urlsafe(16)))
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        resp.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        if en_produccion:
            resp.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return resp

    app.add_template_filter(formato_ia, "formato_ia")

    @app.errorhandler(CSRFError)
    def error_csrf(e):
        if request.path == "/chat-ia":
            return jsonify({"respuesta": "Tu sesión expiró. Recarga la página e intenta de nuevo."}), 400
        flash("La sesión del formulario expiró. Intenta de nuevo.", "error")
        return redirect(url_segura(request.referrer) or url_for("auth.login"))

    @app.errorhandler(429)
    def demasiadas_peticiones(e):
        if request.path == "/chat-ia":
            return jsonify({"respuesta": "Demasiadas consultas seguidas. Espera un minuto."}), 429
        flash("Demasiados intentos. Espera un minuto antes de volver a intentar.", "error")
        return redirect(url_for("auth.login"))
