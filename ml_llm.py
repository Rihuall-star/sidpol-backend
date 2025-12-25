import google.generativeai as genai
import os
import sys

def configurar_gemini():
    # Asegúrate de haber actualizado la variable en Render con la NUEVA Key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ ERROR: No hay API KEY configurada.", file=sys.stderr)
        return False
    try:
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        print(f"❌ Error Config: {e}", file=sys.stderr)
        return False

def generar_respuesta(prompt):
    """
    Configuración estricta para Gemini 2.5 Flash.
    """
    try:
        # Usamos el modelo experimental que querías
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        # Log del error para depuración
        print(f"❌ Error Gemini 2.5: {str(e)}", file=sys.stderr)
        return "El sistema de IA no está disponible en este momento."

# --- FUNCIONES DEL SISTEMA ---

def consultar_chat_general(mensaje_usuario, contexto_datos=""):
    if not configurar_gemini(): return "Error de conexión."
    prompt = f"Asistente SIDPOL. Contexto: {contexto_datos}. Usuario: '{mensaje_usuario}'. Responde brevemente."
    return generar_respuesta(prompt)

def consultar_estratega_ia(total, contexto, riesgo):
    if not configurar_gemini(): return "Error Config."
    prompt = f"Comandante SIDPOL. Proyección: {total}. Tendencia: {contexto}. Riesgo: {riesgo}. Diagnóstico y acciones."
    return generar_respuesta(prompt)

def analizar_riesgo_ia(pred, mod, dpto, trim, anio):
    if not configurar_gemini(): return "Error Config."
    prompt = f"Analista Seguridad. Alerta: {mod} en {dpto}, {trim}-{anio}. Proyección: {pred}. Recomendación."
    return generar_respuesta(prompt)