# nlp_consulta.py
import re

DEPARTAMENTOS = [
    "AMAZONAS","ANCASH","APURIMAC","AREQUIPA","AYACUCHO","CAJAMARCA","CUSCO",
    "HUANCAVELICA","HUANUCO","ICA","JUNIN","LA LIBERTAD","LAMBAYEQUE",
    "LIMA METROPOLITANA","LORETO","MADRE DE DIOS","MOQUEGUA","PASCO","PIURA",
    "PROV. CONST. DEL CALLAO","PUNO","REGION LIMA","SAN MARTIN","TACNA",
    "TUMBES","UCAYALI"
]

def extraer_anio(texto):
    match = re.search(r"(2018|2019|2020|2021|2022|2023|2024|2025|2026)", texto)
    return int(match.group()) if match else None

def extraer_modalidad(texto):
    texto = texto.lower()
    if "extors" in texto: return "Extorsión"
    if "homic" in texto: return "Homicidio"
    if "hurto" in texto: return "Hurto"
    if "estaf" in texto: return "Estafa"
    if "robo" in texto: return "Robo"
    if "violencia" in texto: return "Violencia contra la mujer e integrantes"
    return None

def extraer_departamento(texto):
    texto_up = texto.upper()
    for d in DEPARTAMENTOS:
        if d in texto_up:
            return d
    return None

def detectar_intencion(texto):
    t = texto.lower()

    if "más frecuente" in t or "modalidad más" in t or "principal" in t:
        return "top_modalidad"

    if "top" in t or "ranking" in t:
        return "ranking"

    if "compar" in t and ("entre" in t or "vs" in t):
        return "comparacion"

    if "crec" in t or "aument" in t or "variación" in t:
        return "crecimiento"

    if "tendencia" in t or "evolución" in t or "cómo ha cambiado" in t:
        return "tendencia"

    if "per cápita" in t or "por habitante" in t:
        return "percapita"

    if "trimestre" in t:
        return "trimestres"

    if "total" in t or "cuántas" in t or "cuantos" in t:
        return "total"

    return "desconocida"

def analizar_pregunta(texto):
    return {
        "anio": extraer_anio(texto),
        "modalidad": extraer_modalidad(texto),
        "departamento": extraer_departamento(texto),
        "intencion": detectar_intencion(texto),
    }
