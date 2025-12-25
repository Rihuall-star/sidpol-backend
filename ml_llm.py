import google.generativeai as genai
import os
import sys

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
    Función maestra para Gemini 2.5 Flash
    """
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
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