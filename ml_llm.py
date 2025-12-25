import google.generativeai as genai
import os
import sys

def configurar_gemini():
    # Asegúrate de que esta variable exista en Render
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

def generar_respuesta(prompt):
    """
    Usa Gemini 1.5 Flash (Gratis y Rápido)
    """
    try:
        # Este modelo es el equilibrio perfecto: Gratis y sin límites absurdos
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error Gemini: {error_msg}", file=sys.stderr)
        
        if "429" in error_msg:
            return "⚠️ El sistema está saturado. Intenta en 1 minuto."
        return "El servicio de inteligencia no está disponible."

# --- FUNCIONES DEL SISTEMA ---

def consultar_chat_general(mensaje_usuario, contexto_datos=""):
    if not configurar_gemini(): return "Error de conexión IA."
    
    prompt = f"""
    Eres el Asistente IA de SIDPOL.
    Contexto del Sistema: {contexto_datos}
    Usuario: "{mensaje_usuario}"
    Responde breve y profesionalmente.
    """
    return generar_respuesta(prompt)

def consultar_estratega_ia(total, contexto, riesgo):
    if not configurar_gemini(): return "Error Config."
    prompt = f"Comandante SIDPOL. Proyección 2026: {total}. Tendencia: {contexto}. Riesgo: {riesgo}. Dame diagnóstico y acciones."
    return generar_respuesta(prompt)

def analizar_riesgo_ia(pred, mod, dpto, trim, anio):
    if not configurar_gemini(): return "Error Config."
    prompt = f"Analista. Alerta: {mod} en {dpto}, {trim}-{anio}. Proyección: {pred}. Recomendación táctica."
    return generar_respuesta(prompt)