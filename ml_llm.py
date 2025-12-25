import google.generativeai as genai
import os

def configurar_gemini():
    """
    Configura la API key desde las variables de entorno de Render.
    Retorna True si fue exitoso, False si falta la key.
    """
    # 1. SEGURIDAD: Leemos del entorno, NUNCA hardcodeado
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("❌ ERROR CRÍTICO: No se encontró la variable GEMINI_API_KEY en el entorno.")
        return False

    genai.configure(api_key=api_key)
    return True

def consultar_estratega_ia(total_proyectado, contexto_historico, top_riesgos_txt):
    """
    Genera el análisis estratégico usando Gemini.
    """
    if not configurar_gemini():
        return "Error de Configuración: Falta API Key en el servidor."

    # Si los datos vienen vacíos o son 0, evitamos alucinaciones
    if total_proyectado == 0:
        return "El sistema no dispone de datos históricos suficientes para generar un perfil estratégico confiable."

    # 2. INGENIERÍA DE PROMPT (Rol: Estratega Policial)
    prompt = f"""
    Actúa como un Comandante Estratégico de Inteligencia Criminal (SIDPOL).
    
    SITUACIÓN ACTUAL (Datos en tiempo real):
    - Proyección de Incidencia Delictiva para 2026: {total_proyectado} casos estimados.
    - Contexto reciente (tendencia mensual): {contexto_historico}
    - Zonas/Modalidades de mayor riesgo: {top_riesgos_txt}

    MISIÓN:
    Redacta un informe ejecutivo de inteligencia (máximo 150 palabras).
    Estructura:
    1. DIAGNÓSTICO: Una frase contundente sobre la gravedad de la proyección.
    2. ACCIÓN TÁCTICA: Recomienda 3 acciones operativas concretas (usar viñetas) para mitigar este escenario.
    
    Tono: Formal, Policial, Autoritario y basado en datos.
    """

    try:
        # 3. MODELO: Usamos la versión Flash por velocidad y eficiencia
        # Si tu versión específica es 'gemini-2.5', cámbialo aquí.
        # Por defecto usamos 'gemini-1.5-flash' que es el estándar actual rápido.
        model = genai.GenerativeModel('gemini-1.5-flash') 
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error en Gemini: {e}")
        return "El Agente Estratégico está temporalmente fuera de servicio (Error de conexión IA)."