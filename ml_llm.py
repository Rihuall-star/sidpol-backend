import google.generativeai as genai
import os
import sys
from datetime import date

from flask import session, has_request_context
from pymongo import ReturnDocument

# Límite diario de consultas a Gemini para usuarios con rol "invitado" (cuenta demo).
# Se puede ajustar con la variable de entorno LIMITE_IA_INVITADO.
LIMITE_IA_INVITADO = int(os.getenv("LIMITE_IA_INVITADO", "30"))

# Modelo de Gemini. Se puede cambiar sin tocar el código con la variable GEMINI_MODEL.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
MENSAJE_LIMITE = (
    "La cuenta demo alcanzó su límite diario de consultas a la IA. "
    "Vuelve mañana o explora el resto de los módulos del dashboard."
)


def cuota_ia_agotada():
    """Cuenta una consulta y devuelve True si el usuario invitado superó su límite del día.

    El contador se guarda en MongoDB (por usuario y por día), así que no se
    reinicia al cerrar sesión. Los roles admin y analista no tienen límite.
    """
    if not has_request_context() or session.get("rol") != "invitado":
        return False
    try:
        from ml_utils import db  # import diferido para evitar dependencias circulares
        hoy = date.today().isoformat()
        doc = db["usuarios"].find_one_and_update(
            {"username": session.get("user")},
            [{"$set": {
                "ia_fecha": hoy,
                "ia_usadas": {"$cond": [
                    {"$eq": ["$ia_fecha", hoy]},
                    {"$add": [{"$ifNull": ["$ia_usadas", 0]}, 1]},
                    1,
                ]},
            }}],
            return_document=ReturnDocument.AFTER,
        )
        return doc is None or doc.get("ia_usadas", 0) > LIMITE_IA_INVITADO
    except Exception as e:
        print(f"Error verificando cuota IA: {e}", file=sys.stderr)
        return True  # ante la duda, no gastar cuota

def configurar_gemini():
    # Usará la llave nueva que configuraste en Render
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ ERROR: No se encontró GEMINI_API_KEY.", file=sys.stderr)
        return False
    try:
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        print(f"❌ Error Configuración IA: {e}", file=sys.stderr)
        return False

def generar_respuesta(prompt):
    """
    Función maestra para Gemini (modelo configurable con GEMINI_MODEL)
    """
    if cuota_ia_agotada():
        return MENSAJE_LIMITE
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt, request_options={"timeout": 60})
        return response.text
    except Exception as e:
        print(f"❌ Error Gemini: {str(e)}", file=sys.stderr)
        return "El sistema de inteligencia está reiniciando servicios. Intente nuevamente."

# --- FUNCIONES PARA LOS AGENTES ---

def consultar_chat_general(mensaje_usuario, contexto_datos=""):
    if not configurar_gemini(): return "Error de conexión."
    prompt = f"""
    Eres el Asistente Oficial SIDPOL.
    CONTEXTO: {contexto_datos}
    USUARIO: "{mensaje_usuario}"
    Responde de forma breve, táctica y profesional.
    """
    return generar_respuesta(prompt)

def consultar_estratega_ia(total, contexto, riesgo):
    if not configurar_gemini(): return "Error Config."
    prompt = f"""
    Actúa como Comandante General.
    SITUACIÓN 2026: {total} delitos proyectados.
    TENDENCIA: {contexto}.
    FOCO DE RIESGO: {riesgo}.
    Genera: 1 Diagnóstico breve y 3 Acciones Estratégicas (viñetas).
    """
    return generar_respuesta(prompt)

def analizar_riesgo_ia(pred, mod, dpto, trim, anio):
    if not configurar_gemini(): return "Análisis no disponible."
    prompt = f"""
    Actúa como Analista de Inteligencia Táctica.
    ALERTA: {mod} en {dpto} ({trim}-{anio}).
    PROYECCIÓN: {pred} incidentes.
    Orden: Dame UNA recomendación operativa específica para patrullaje en esa zona.
    """
    return generar_respuesta(prompt)