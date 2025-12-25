import google.generativeai as genai
import os
import sys

def configurar_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ ERROR: No se encontró la variable GEMINI_API_KEY.", file=sys.stderr)
        return False
    
    try:
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        print(f"❌ Error configurando Gemini: {e}", file=sys.stderr)
        return False

def obtener_modelo():
    # Intentamos usar el modelo más compatible.
    # Si tienes acceso a gemini-2.0, úsalo. Si no, gemini-1.5-flash es el más seguro.
    return genai.GenerativeModel('gemini-1.5-flash')

def consultar_estratega_ia(total_proyectado, contexto_historico, top_riesgos_txt):
    if not configurar_gemini(): return "Error: Falta API Key en configuración."

    prompt = f"""
    Actúa como Comandante de Inteligencia (SIDPOL).
    SITUACIÓN 2026: Proyección de {total_proyectado} delitos.
    TENDENCIA: {contexto_historico}
    FOCO: {top_riesgos_txt}
    
    Misión: Informe ejecutivo breve (3 líneas) y 3 acciones tácticas con viñetas.
    """
    try:
        model = obtener_modelo()
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"❌ Error en Estratega IA: {e}", file=sys.stderr)
        return "El Agente Estratégico no pudo conectar con la central de IA."

def analizar_riesgo_ia(prediccion, modalidad, departamento, trimestre, anio):
    if not configurar_gemini(): return "Error Configuración."
    
    prompt = f"""
    Eres analista de seguridad.
    ALERTA: {modalidad} en {departamento}, {trimestre}-{anio}.
    Proyección: {prediccion} casos.
    Recomienda 1 medida de patrullaje específica. Sé breve y directivo.
    """
    try:
        model = obtener_modelo()
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"❌ Error en Riesgo IA: {e}", file=sys.stderr)
        return "Análisis táctico no disponible."

def consultar_chat_general(mensaje_usuario, contexto_datos=""):
    """
    Chatbot General que conoce los datos de la base de datos.
    """
    if not configurar_gemini(): return "Error: API Key no configurada."

    prompt = f"""
    Eres el Asistente IA de SIDPOL.
    
    CONTEXTO DE DATOS EN TIEMPO REAL:
    {contexto_datos}
    
    PREGUNTA DEL USUARIO: "{mensaje_usuario}"
    
    Responde de forma útil y profesional. Si te preguntan por datos que tienes en el contexto, úsalos.
    Si no sabes, sugiere consultar los módulos 'Predicción' o 'Agente Logístico'.
    """
    try:
        model = obtener_modelo()
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"❌ Error en Chat General: {e}", file=sys.stderr)
        return "Lo siento, hubo una interferencia en la comunicación. Intente de nuevo."