import os
import google.generativeai as genai

# Configurar Gemini con la API KEY del .env
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Modelo correcto según tu cuenta → ESTE FUNCIONA
MODEL_NAME = "gemini-2.5-flash"

def preguntar_gemini(mensaje, contexto=None):
    """
    Envía una pregunta a Gemini.
    Soporta opcionalmente un contexto (si deseas incluir datos desde MongoDB).
    """
    model = genai.GenerativeModel(MODEL_NAME)

    # Construcción del prompt
    if contexto:
        prompt = (
            "Eres un asistente especializado en denuncias policiales del Perú "
            "y también puedes responder preguntas generales.\n\n"
            f"Contexto:\n{contexto}\n\n"
            f"Pregunta del usuario:\n{mensaje}\n\n"
            "Responde de forma clara y en español."
        )
    else:
        prompt = (
            "Eres un asistente en español. Responde de manera clara y concisa.\n\n"
            f"Pregunta del usuario:\n{mensaje}"
        )

    # Manejo de errores para evitar que Flask se caiga
    try:
        response = model.generate_content(prompt)
        return response.text if response and response.text else "No hay respuesta del modelo."
    except Exception as e:
        print("❌ Error al consultar Gemini:", e)
        return "⚠ Ocurrió un error al conectarse con la IA. Revisa tu API Key o el modelo."
