"""Chequeo de salud para Render, Docker y monitoreo externo. No requiere sesión ni expone secretos."""
from flask import Blueprint, current_app, jsonify

from sidpol.extensiones import limiter, repo

bp = Blueprint("salud", __name__)


@bp.route("/health")
@limiter.limit("60 per minute")
def health():
    estado = {"estado": "ok", "version": current_app.config["VERSION_CODIGO"]}
    try:
        repo().ping()
        estado["base_de_datos"] = "ok"
        estado["dataset"] = repo().version_dataset()
    except Exception:  # noqa: BLE001
        estado.update(estado="degradado", base_de_datos="sin conexión")
        return jsonify(estado), 503
    return jsonify(estado)
