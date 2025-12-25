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

def obtener_modelo_2_5():
    """
    Intenta obtener el modelo Gemini 2.5 Flash solicitado.
    """
    # Usamos la cadena exacta que solicitaste.
    # Si este falla por cuota (429), el sistema avisará.
    return genai.GenerativeModel('gemini-2.5-flash')

def generar_respuesta(prompt):
    try:
        model = obtener_modelo_2_5()
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error Gemini 2.5: {error_msg}", file=sys.stderr)
        
        # Manejo de errores específicos
        if "404" in error_msg:
            return "⚠️ Error Técnico: La librería del servidor necesita actualizarse (Clear Cache)."
        if "429" in error_msg or "quota" in error_msg.lower():
            return "⚠️ Cuota Excedida: El modelo 2.5 ha alcanzado su límite diario gratuito."
            
        return "Servicio de inteligencia temporalmente no disponible."

# --- FUNCIONES DEL SISTEMA ---

def consultar_chat_general(mensaje_usuario, contexto_datos=""):
    if not configurar_gemini(): return "Error de conexión."
    
    prompt = f"""
    Eres el Asistente IA de SIDPOL.
    Contexto: {contexto_datos}
    Usuario: "{mensaje_usuario}"
    Responde breve y con autoridad policial.
    """
    return generar_respuesta(prompt)

def consultar_estratega_ia(total, contexto, riesgo):
    if not configurar_gemini(): return "Error Config."
    prompt = f"Comandante SIDPOL. Proyección: {total}. Tendencia: {contexto}. Riesgo: {riesgo}. Dame diagnóstico y acciones."
    return generar_respuesta(prompt)

def analizar_riesgo_ia(pred, mod, dpto, trim, anio):
    if not configurar_gemini(): return "Error Config."
    prompt = f"Analista. Alerta: {mod} en {dpto}, {trim}-{anio}. Proyección: {pred}. Recomendación táctica."
    return generar_respuesta(prompt)