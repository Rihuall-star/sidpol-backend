import google.generativeai as genai
import os
import sys

def configurar_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ ERROR: No se encontró GEMINI_API_KEY.", file=sys.stderr)
        return False
    try:
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        print(f"❌ Error al configurar Gemini: {e}", file=sys.stderr)
        return False

def obtener_modelo():
    # USAMOS LA VERSIÓN QUE CONFIRMASTE QUE FUNCIONA 100%
    return genai.GenerativeModel('gemini-2.5-flash')

def consultar_chat_general(mensaje_usuario, contexto_datos=""):
    """
    Función para el Chatbot Flotante.
    """
    if not configurar_gemini(): return "Error: API Key no configurada."

    prompt = f"""
    Eres el Asistente IA de SIDPOL (Sistema de Inteligencia Policial).
    
    INFORMACIÓN DEL SISTEMA (Contexto Real):
    {contexto_datos}
    
    PREGUNTA DEL OFICIAL: "{mensaje_usuario}"
    
    INSTRUCCIONES:
    - Responde de forma breve, útil y con tono de autoridad policial.
    - Usa el contexto numérico si es relevante para la pregunta.
    - Si no sabes, sugiere consultar los módulos especializados.
    """
    try:
        model = obtener_modelo()
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"❌ Error Chat General: {e}", file=sys.stderr)
        return "El sistema de comunicaciones IA está inestable. Reintente."

# --- Funciones para los otros módulos (Riesgo y Estratega) ---
def consultar_estratega_ia(total, contexto, riesgo):
    if not configurar_gemini(): return "Error Configuración."
    prompt = f"Comandante SIDPOL. Proyección 2026: {total}. Tendencia: {contexto}. Foco: {riesgo}. Dame 1 diagnóstico y 3 acciones tácticas."
    try:
        model = obtener_modelo()
        return model.generate_content(prompt).text
    except: return "Servicio no disponible."

def analizar_riesgo_ia(pred, mod, dpto, trim, anio):
    if not configurar_gemini(): return "Error Configuración."
    prompt = f"Analista Seguridad. Alerta: {mod} en {dpto}, {trim}-{anio}. Proyección: {pred}. Dame 1 recomendación operativa urgente."
    try:
        model = obtener_modelo()
        return model.generate_content(prompt).text
    except: return "Análisis no disponible."