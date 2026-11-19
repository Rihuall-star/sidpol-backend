"""Inicio y cierre de sesión, con auditoría y límite de intentos."""
import logging

from flask import Blueprint, flash, redirect, render_template, request, session, url_for
from flask_login import login_required, login_user, logout_user
from werkzeug.security import check_password_hash

from sidpol.extensiones import limiter, repo

bp = Blueprint("auth", __name__)
log = logging.getLogger(__name__)


def _auditar(usuario, evento):
    try:
        repo().registrar_evento(usuario, evento, request.remote_addr)
    except Exception as e:  # noqa: BLE001
        log.error("no se pudo guardar la auditoría: %s", e)


@bp.route("/login", methods=["GET", "POST"])
@limiter.limit("5 per minute; 30 per hour", methods=["POST"])
def login():
    if request.method == "POST":
        from sidpol import Usuario
        username = str(request.form.get("username", "")).strip()
        password = str(request.form.get("password", ""))
        datos = repo().buscar_usuario(username)

        if datos and check_password_hash(datos["password"], password):
            session.clear()  # sesión nueva: evita la fijación de sesión
            session.permanent = True
            login_user(Usuario(datos))
            session["user"] = username
            session["rol"] = datos.get("rol", "invitado")
            _auditar(username, "Inicio de Sesión")
            return redirect(url_for("secciones.index"))

        _auditar(username, "Intento fallido")  # queda en la auditoría (solo admin); útil contra fuerza bruta
        # En el log de la aplicación no se escriben datos de cuenta, solo el evento y el origen
        log.warning("intento de login fallido", extra={"ip": request.remote_addr})
        flash("Usuario o contraseña incorrectos", "error")
        return redirect(url_for("auth.login"))

    return render_template("login.html")


@bp.route("/logout")
@login_required
def logout():
    logout_user()
    session.clear()
    flash("Has cerrado sesión correctamente.", "success")
    return redirect(url_for("auth.login"))
