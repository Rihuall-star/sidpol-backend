"""Agente estratégico y asistente de chat con herramientas de datos."""
import logging
from datetime import datetime

from flask import Blueprint, jsonify, render_template, request, session
from flask_login import login_required

from sidpol.extensiones import limiter, repo
from sidpol.servicios import analitica, ia, modelos, resultados

bp = Blueprint("ia", __name__)
log = logging.getLogger(__name__)


def contexto_base(r):
    """Resumen corto que siempre acompaña a la pregunta. Lo específico, el modelo lo pide con herramientas."""
    pan = resultados.obtener(r, "panorama")
    partes = ["Dataset SIDPOL de denuncias policiales del Perú, 2018-2025."]
    if pan:
        anual = ", ".join(f"{a}: {v:,}" for a, v in zip(pan["anios"], pan["serie_anual"]))
        partes.append(f"Denuncias por año (nacional): {anual}. {pan['anio_fin']} tiene datos solo hasta "
                      f"{pan['periodo'].split('–')[1]} (año incompleto).")
    partes.append(analitica.texto_evolucion(resultados.obtener(r, "modalidades") or [], top=7))
    partes.append("PROYECCIÓN: " + modelos.texto_proyeccion(resultados.obtener(r, "proyeccion")))
    return "\n".join(partes)


@bp.route("/agente-estrategico")
@login_required
def agente_estrategico():
    r = repo()
    p = resultados.obtener(r, "proyeccion")
    if not p:
        return render_template("agente_estrategico.html", total="No disponible",
                               analisis="No se pudo calcular la proyección en este momento.")

    # El análisis se genera una vez al día y se guarda: así las visitas no gastan cuota de Gemini
    clave = f"estratega-{datetime.now():%Y-%m-%d}-{r.version_dataset()}-{p['total_2026']}"
    analisis = r.leer_cache_ia(clave)
    if not analisis:
        contexto = (modelos.texto_proyeccion(p) + "\n" +
                    analitica.texto_evolucion(resultados.obtener(r, "modalidades") or [], top=7))
        analisis = ia.consultar_estratega(p["total_2026"], contexto, "Extorsión y delitos patrimoniales")
        if analisis not in ia.MENSAJES_DE_FALLA:
            r.guardar_cache_ia(clave, analisis, tipo="estratega")
    return render_template("agente_estrategico.html", total="{:,}".format(p["total_2026"]), analisis=analisis)


@bp.route("/chat-ia", methods=["POST"])
@login_required
@limiter.limit("10 per minute")
def chat_ia():
    mensaje = request.form.get("mensaje", "").strip()[:500]  # limita el tamaño del prompt
    if not mensaje:
        return jsonify({"respuesta": "No entendí.", "fuentes": []})
    try:
        r = repo()
        return jsonify(ia.responder_chat(r, mensaje, contexto_base(r),
                                         usuario=session.get("user"), rol=session.get("rol")))
    except Exception as e:  # noqa: BLE001
        log.error("error en el chat: %s", e)
        return jsonify({"respuesta": "Ocurrió un error consultando la base de datos.", "fuentes": []})
