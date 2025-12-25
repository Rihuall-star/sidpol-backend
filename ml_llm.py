import google.generativeai as genai
import os
import sys

def configurar_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ ERROR: No se encontró GEMINI_API_KEY", file=sys.stderr)
        return False
    genai.configure(api_key=api_key)
    return True

def obtener_modelo_seguro():
    # Usamos 'gemini-pro' porque es el más estable y evita el error 404
    return genai.GenerativeModel('gemini-pro')

def consultar_chat_general(mensaje_usuario, contexto_datos=""):
    """
    Cerebro del Chatbot Flotante
    """
    if not configurar_gemini(): return "Error: API Key no configurada."

    prompt = f"""
    Eres el Asistente IA de SIDPOL (Sistema de Inteligencia Policial).
    
    INFORMACIÓN CLAVE DEL SISTEMA:
    {contexto_datos}
    
    USUARIO: "{mensaje_usuario}"
    
    Instrucciones:
    - Responde de forma breve y profesional (estilo policial).
    - Usa la información clave si es relevante para la pregunta.
    - Si no sabes, sugiere ir al módulo 'Agente Estratega'.
    """
    try:
        model = obtener_modelo_seguro()
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"❌ Error Gemini Chat: {e}", file=sys.stderr)
        return "El sistema de IA está reiniciando servicios. Intente en unos segundos."

def consultar_estratega_ia(total_proyectado, contexto, top_riesgo):
    if not configurar_gemini(): return "Error Configuración."
    prompt = f"Actúa como Comandante SIDPOL. Proyección 2026: {total_proyectado} delitos. Tendencia: {contexto}. Riesgo: {top_riesgo}. Dame 1 diagnóstico y 3 acciones tácticas."
    try:
        model = obtener_modelo_seguro()
        response = model.generate_content(prompt)
        return response.text
    except: return "Servicio no disponible."

def analizar_riesgo_ia(prediccion, modalidad, dpto, trim, anio):
    if not configurar_gemini(): return "Error Configuración."
    prompt = f"Analista de seguridad. Alerta: {modalidad} en {dpto}, {trim}-{anio}. Proyección: {prediccion}. Recomienda 1 medida operativa urgente."
    try:
        model = obtener_modelo_seguro()
        response = model.generate_content(prompt)
        return response.text
    except: return "Análisis no disponible."