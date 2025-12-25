import google.generativeai as genai
import os
import sys

def configurar_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ ERROR: Falta GEMINI_API_KEY.", file=sys.stderr)
        return False
    try:
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        print(f"❌ Error Config: {e}", file=sys.stderr)
        return False

def obtener_modelo():
    # USAMOS LA VERSIÓN SOLICITADA
    return genai.GenerativeModel('gemini-2.5-flash')

def consultar_chat_general(mensaje_usuario, contexto_datos=""):
    """ Cerebro del Chatbot Flotante """
    if not configurar_gemini(): return "Error de conexión con la IA."

    prompt = f"""
    Eres el Asistente IA de SIDPOL.
    
    DATOS DEL SISTEMA:
    {contexto_datos}
    
    PREGUNTA: "{mensaje_usuario}"
    
    Responde brevemente como un oficial de inteligencia.
    """
    try:
        model = obtener_modelo()
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"❌ Error Chat: {e}", file=sys.stderr)
        return "Servicio de inteligencia temporalmente no disponible."

# Funciones auxiliares para los otros módulos
def consultar_estratega_ia(total, contexto, riesgo):
    if not configurar_gemini(): return "Error Config."
    try:
        model = obtener_modelo()
        prompt = f"Comandante SIDPOL. Proyección 2026: {total}. Tendencia: {contexto}. Riesgo: {riesgo}. Dame diagnóstico y acciones."
        return model.generate_content(prompt).text
    except: return "Sin servicio."

def analizar_riesgo_ia(pred, mod, dpto, trim, anio):
    if not configurar_gemini(): return "Error Config."
    try:
        model = obtener_modelo()
        prompt = f"Analista. Alerta: {mod} en {dpto}, {trim}-{anio}. Proyección: {pred}. Dame recomendación táctica."
        return model.generate_content(prompt).text
    except: return "Sin servicio."