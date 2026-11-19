"""Administración: monitor de usuarios en línea, bitácora de accesos y alta de usuarios."""
import logging
from datetime import datetime, timedelta

from flask import Blueprint, flash, redirect, render_template, request, url_for
from flask_login import login_required
from werkzeug.security import generate_password_hash

from sidpol.extensiones import repo
from sidpol.seguridad import admin_required

bp = Blueprint("admin", __name__)
log = logging.getLogger(__name__)
ROLES = ("admin", "analista", "invitado")


@bp.route("/usuarios-activos")
@login_required
@admin_required
def usuarios_activos():
    try:
        online = repo().usuarios_activos(datetime.now() - timedelta(minutes=5))
        historial = repo().historial(50)
    except Exception as e:  # noqa: BLE001
        log.error("error en el monitor de usuarios: %s", e)
        online, historial = [], []
    return render_template("usuarios_activos.html", online=online, historial=historial)


@bp.route("/crear_usuario", methods=["GET", "POST"])
@login_required
@admin_required
def crear_usuario():
    if request.method == "POST":
        username = request.form.get("username", "").strip()[:50]
        password = request.form.get("password", "")
        rol = request.form.get("rol", "")
        if not username or len(password) < 12 or rol not in ROLES:
            flash("Datos no válidos: la contraseña debe tener al menos 12 caracteres y el rol debe existir.", "error")
        elif repo().crear_usuario(username, generate_password_hash(password), rol):
            flash(f"Usuario {username} creado.", "success")
            return redirect(url_for("admin.crear_usuario"))
        else:
            flash("Usuario ya existe.", "error")
    return render_template("crear_usuario.html")
