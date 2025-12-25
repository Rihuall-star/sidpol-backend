import os
import sys
from openai import OpenAI

# Configuración del Cliente DeepSeek
def obtener_cliente():
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("❌ ERROR: Falta DEEPSEEK_API_KEY.", file=sys.stderr)
        return None
    
    # DeepSeek usa la estructura compatible con OpenAI
    return OpenAI(
        api_key=api_key, 
        base_url="https://api.deepseek.com"
    )

def generar_respuesta_deepseek(system_prompt, user_prompt):
    """
    Función genérica para consultar a DeepSeek-V3.
    """
    client = obtener_cliente()
    if not client:
        return "Error de configuración: Falta API Key de DeepSeek."

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",  # Este es el modelo V3 (rápido y potente)
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=False,
            temperature=0.7 # Creatividad balanceada
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"❌ Error DeepSeek: {e}", file=sys.stderr)
        return "El sistema de inteligencia está temporalmente fuera de servicio."

# --- FUNCIONES ESPECÍFICAS PARA TUS MÓDULOS ---

def consultar_chat_general(mensaje_usuario, contexto_datos=""):
    system = f"""
    Eres el Asistente IA de SIDPOL (Sistema de Inteligencia Policial).
    Tu misión es asistir a oficiales con respuestas breves, tácticas y profesionales.
    
    DATOS ACTUALES DEL SISTEMA:
    {contexto_datos}
    """
    return generar_respuesta_deepseek(system, mensaje_usuario)

def consultar_estratega_ia(total, contexto, riesgo):
    system = "Actúa como un Comandante Estratégico de la Policía (SIDPOL). Sé directo y autoritario."
    prompt = f"""
    Analiza esta situación para el 2026:
    - Proyección Total: {total} delitos.
    - Tendencia Reciente: {contexto}.
    - Foco de Riesgo: {riesgo}.
    
    Genera:
    1. Un diagnóstico de una frase.
    2. Tres acciones tácticas operativas (con viñetas).
    """
    return generar_respuesta_deepseek(system, prompt)

def analizar_riesgo_ia(pred, mod, dpto, trim, anio):
    system = "Eres un Analista de Inteligencia Criminal experto en prevención del delito."
    prompt = f"""
    ALERTA DETECTADA:
    - Modalidad: {mod}
    - Zona: {dpto}
    - Periodo: {trim}-{anio}
    - Proyección Matemática: {pred} incidentes.
    
    Dame UNA recomendación operativa urgente y específica para mitigar este riesgo.
    """
    return generar_respuesta_deepseek(system, prompt)