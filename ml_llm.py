import google.generativeai as genai
import os

def configurar_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key: return False
    genai.configure(api_key=api_key)
    return True

# Función existente (Agente Estratega)
def consultar_estratega_ia(total_proyectado, contexto_historico, top_riesgos_txt):
    if not configurar_gemini(): return "Error: Falta API Key."
    if total_proyectado == 0: return "Datos insuficientes."

    prompt = f"""
    Actúa como Comandante de Inteligencia (SIDPOL).
    SITUACIÓN 2026: Se proyectan {total_proyectado} incidentes.
    TENDENCIA: {contexto_historico}
    FOCO: {top_riesgos_txt}
    
    Redacta un informe (máx 100 palabras) con:
    1. Diagnóstico breve.
    2. Tres acciones tácticas urgentes.
    """
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except: return "Error de conexión con el Agente Cognitivo."

# --- NUEVA FUNCIÓN PARA RIESGO POR MODALIDAD ---
def analizar_riesgo_ia(prediccion, modalidad, departamento, trimestre, anio):
    """
    Analiza una predicción específica de riesgo.
    """
    if not configurar_gemini(): return "Error de configuración IA."

    prompt = f"""
    Eres un analista de riesgos de seguridad ciudadana.
    
    ALERTA DETECTADA:
    - Modalidad: {modalidad}
    - Ubicación: {departamento}
    - Periodo: {trimestre} del {anio}
    - Predicción del Modelo Matemático: {prediccion} incidentes estimados.

    TAREA:
    Provee un análisis táctico operativo (máximo 50 palabras) sobre este escenario específico.
    ¿Qué tipo de patrullaje o medida preventiva recomiendas para {departamento} considerando esta cifra?
    Sé directo y autoritario.
    """
    
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Análisis no disponible por el momento."
    
# --- AGREGAR AL FINAL DE ml_llm.py ---

def consultar_chat_general(mensaje_usuario):
    """
    Función para el Chatbot Flotante (Estilo Cisco).
    Responde preguntas generales sobre seguridad o el sistema.
    """
    if not configurar_gemini():
        return "Error de configuración: API Key no detectada."

    prompt = f"""
    Eres el Asistente Virtual Inteligente de SIDPOL (Sistema de Inteligencia Policial).
    Tu objetivo es asistir a oficiales y analistas.
    
    PREGUNTA DEL USUARIO: "{mensaje_usuario}"
    
    INSTRUCCIONES:
    - Responde de forma concisa, profesional y útil.
    - Si te preguntan quién eres, di que eres el Asistente IA de SIDPOL basado en Gemini 2.5.
    - Si te piden datos exactos, indica que pueden consultarlos en los módulos de 'Predicción' o 'Agente Logístico'.
    - Mantén un tono de servicio policial.
    """

    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return "Lo siento, mi enlace con la central de IA está inestable. Intente de nuevo."