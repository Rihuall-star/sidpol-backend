"""
Herramientas de datos que el asistente de IA puede usar (function calling de Gemini).

El modelo NO escribe consultas: solo elige una de estas funciones y sus parámetros. Cada
función es de solo lectura, valida sus argumentos contra listas cerradas y limita el tamaño
de la respuesta. Así se mitiga el riesgo de "agencia excesiva" (OWASP Top 10 para LLM, LLM06)
y la inyección de consultas a través del prompt.
"""
import unicodedata

from sidpol.servicios import analitica, resultados

MODALIDADES = analitica.MODALIDADES_ORDEN
DEPARTAMENTOS = analitica.DEPARTAMENTOS
ANIO_MIN, ANIO_MAX = 2018, 2025
MAX_FILAS = 15

ALIAS_DEPARTAMENTO = {
    "LIMA": "LIMA METROPOLITANA", "LIMA METROPOLITANA": "LIMA METROPOLITANA",
    "LIMA PROVINCIAS": "REGION LIMA", "LIMA REGION": "REGION LIMA", "REGION LIMA": "REGION LIMA",
    "CALLAO": "PROV. CONST. DEL CALLAO",
}


class ArgumentoInvalido(ValueError):
    pass


def _sin_tildes(texto):
    return "".join(c for c in unicodedata.normalize("NFD", str(texto)) if unicodedata.category(c) != "Mn")


def _departamento(valor):
    if valor in (None, ""):
        return None
    limpio = _sin_tildes(valor).upper().strip()
    limpio = ALIAS_DEPARTAMENTO.get(limpio, limpio)
    if limpio not in DEPARTAMENTOS:
        raise ArgumentoInvalido(f"Departamento desconocido: {valor}")
    return limpio


def _modalidad(valor):
    if valor in (None, ""):
        return None
    buscado = _sin_tildes(valor).lower().strip()
    for m in MODALIDADES:
        if _sin_tildes(m).lower() == buscado:
            return m
    raise ArgumentoInvalido(f"Modalidad desconocida: {valor}")


def _anio(valor):
    if valor in (None, ""):
        return None
    anio = int(valor)
    if not ANIO_MIN <= anio <= ANIO_MAX:
        raise ArgumentoInvalido(f"Año fuera de rango ({ANIO_MIN}-{ANIO_MAX}): {valor}")
    return anio


def _n(valor, defecto):
    return max(1, min(int(valor or defecto), MAX_FILAS))


def _filtro(modalidad=None, departamento=None, anio=None):
    f = {}
    if modalidad:
        f["P_MODALIDADES"] = modalidad
    if departamento:
        f["DPTO_HECHO_NEW"] = departamento
    if anio:
        f["ANIO"] = anio
    return f


def _corte(repo):
    pan = resultados.obtener(repo, "panorama") or {}
    return pan.get("anio_fin"), pan.get("mes_fin")


# ---------------------------------------------------------------------
# Herramientas
# ---------------------------------------------------------------------
def denuncias_por_anio(repo, modalidad=None, departamento=None):
    modalidad, departamento = _modalidad(modalidad), _departamento(departamento)
    filas = repo.agregada().aggregate([
        {"$match": _filtro(modalidad, departamento)},
        {"$group": {"_id": "$ANIO", "total": {"$sum": "$cantidad"}}}, {"$sort": {"_id": 1}}])
    anio_fin, mes_fin = _corte(repo)
    return {"modalidad": modalidad or "todas", "departamento": departamento or "nacional",
            "por_anio": {str(d["_id"]): d["total"] for d in filas if d["_id"]},
            "nota": f"{anio_fin} solo tiene datos hasta el mes {mes_fin}." if anio_fin else ""}


def ranking_departamentos(repo, anio=None, modalidad=None, n=10):
    anio, modalidad, n = _anio(anio), _modalidad(modalidad), _n(n, 10)
    filas = repo.agregada().aggregate([
        {"$match": {**_filtro(modalidad, None, anio), "DPTO_HECHO_NEW": {"$nin": [None, ""]}}},
        {"$group": {"_id": "$DPTO_HECHO_NEW", "total": {"$sum": "$cantidad"}}},
        {"$sort": {"total": -1}}, {"$limit": n}])
    return {"anio": anio or "2018-2025", "modalidad": modalidad or "todas",
            "ranking": [{"departamento": analitica.nombre(d["_id"]), "total": d["total"]} for d in filas]}


def principales_delitos(repo, departamento=None, anio=None, n=5):
    departamento, anio, n = _departamento(departamento), _anio(anio), _n(n, 5)
    filas = repo.agregada().aggregate([
        {"$match": _filtro(None, departamento, anio)},
        {"$group": {"_id": "$P_MODALIDADES", "total": {"$sum": "$cantidad"}}},
        {"$sort": {"total": -1}}, {"$limit": n}])
    return {"departamento": departamento or "nacional", "anio": anio or "2018-2025",
            "delitos": [{"modalidad": d["_id"], "total": d["total"]} for d in filas if d["_id"]]}


def comparar_anios(repo, anio_base, anio_comparado, modalidad=None, departamento=None):
    """Compara dos años usando los mismos meses (si uno está incompleto, se recorta el otro)."""
    anio_base, anio_comparado = _anio(anio_base), _anio(anio_comparado)
    if not anio_base or not anio_comparado:
        raise ArgumentoInvalido("Se necesitan anio_base y anio_comparado")
    modalidad, departamento = _modalidad(modalidad), _departamento(departamento)
    anio_fin, mes_fin = _corte(repo)
    meses = mes_fin if anio_fin in (anio_base, anio_comparado) and mes_fin else 12
    filas = repo.agregada().aggregate([
        {"$match": {**_filtro(modalidad, departamento), "ANIO": {"$in": [anio_base, anio_comparado]},
                    "MES": {"$lte": meses}}},
        {"$group": {"_id": "$ANIO", "total": {"$sum": "$cantidad"}}}])
    totales = {d["_id"]: d["total"] for d in filas}
    a, b = totales.get(anio_base, 0), totales.get(anio_comparado, 0)
    return {"modalidad": modalidad or "todas", "departamento": departamento or "nacional",
            "meses_comparados": f"enero a mes {meses}",
            f"total_{anio_base}": a, f"total_{anio_comparado}": b,
            "variacion_pct": analitica.variacion(b, a)}


def focos_distritales(repo, modalidad="Extorsión", departamento=None, anio=None, n=10):
    modalidad = _modalidad(modalidad) or "Extorsión"
    departamento, n = _departamento(departamento), _n(n, 10)
    anio = _anio(anio) or (resultados.obtener(repo, "territorio") or {}).get("ultimo_completo")
    filas = repo.denuncias.aggregate([   # el detalle por distrito solo está en la colección cruda
        {"$match": {**_filtro(modalidad, departamento, anio), "DIST_HECHO": {"$nin": [None, ""]}}},
        {"$group": {"_id": {"d": "$DIST_HECHO", "p": "$DPTO_HECHO_NEW"}, "total": {"$sum": "$cantidad"}}},
        {"$sort": {"total": -1}}, {"$limit": n}])
    return {"modalidad": modalidad, "anio": anio, "departamento": departamento or "nacional",
            "distritos": [{"distrito": analitica.nombre_distrito(d["_id"]["d"], d["_id"]["p"]),
                           "departamento": analitica.nombre(d["_id"]["p"]), "total": d["total"]} for d in filas]}


def proyeccion_2026(repo):
    p = resultados.obtener(repo, "proyeccion")
    if not p:
        return {"error": "Proyección no disponible"}
    return {k: p[k] for k in ("modelo", "total_2026", "total_2026_min", "total_2026_max",
                              "total_2025_real", "ultimo_mes", "mape_sarima", "mape_lineal")}


# ---------------------------------------------------------------------
# Catálogo para Gemini (JSON Schema con listas cerradas)
# ---------------------------------------------------------------------
_MOD = {"type": "string", "enum": MODALIDADES, "description": "Modalidad del delito"}
_DPTO = {"type": "string", "enum": DEPARTAMENTOS,
         "description": "Departamento. Lima ciudad = LIMA METROPOLITANA; Callao = PROV. CONST. DEL CALLAO"}
_ANIO = {"type": "integer", "minimum": ANIO_MIN, "maximum": ANIO_MAX}
_N = {"type": "integer", "minimum": 1, "maximum": MAX_FILAS}

HERRAMIENTAS = {
    "denuncias_por_anio": (denuncias_por_anio,
        "Total de denuncias por año (2018-2025), nacional o de un departamento, de todas o de una modalidad.",
        {"modalidad": _MOD, "departamento": _DPTO}, []),
    "ranking_departamentos": (ranking_departamentos,
        "Departamentos con más denuncias, opcionalmente de un año o una modalidad.",
        {"anio": _ANIO, "modalidad": _MOD, "n": _N}, []),
    "principales_delitos": (principales_delitos,
        "Modalidades más denunciadas, nacional o de un departamento, opcionalmente de un año.",
        {"departamento": _DPTO, "anio": _ANIO, "n": _N}, []),
    "comparar_anios": (comparar_anios,
        "Compara el total de dos años con los mismos meses y da la variación porcentual.",
        {"anio_base": _ANIO, "anio_comparado": _ANIO, "modalidad": _MOD, "departamento": _DPTO},
        ["anio_base", "anio_comparado"]),
    "focos_distritales": (focos_distritales,
        "Distritos con más denuncias de una modalidad (por defecto extorsión) en un año.",
        {"modalidad": _MOD, "departamento": _DPTO, "anio": _ANIO, "n": _N}, []),
    "proyeccion_2026": (proyeccion_2026,
        "Proyección de denuncias para 2026 (SARIMA) con su intervalo de confianza y error de validación.",
        {}, []),
}


def declaraciones():
    """Declaraciones de funciones en el formato del SDK google-genai."""
    from google.genai import types
    return [types.FunctionDeclaration(
        name=nombre, description=descripcion,
        parameters_json_schema={"type": "object", "properties": props, "required": requeridos})
        for nombre, (_, descripcion, props, requeridos) in HERRAMIENTAS.items()]


def ejecutar(repo, nombre, argumentos):
    """Ejecuta una herramienta pedida por el modelo. Nunca lanza: los errores vuelven como datos."""
    if nombre not in HERRAMIENTAS:
        return {"error": f"Herramienta no permitida: {nombre}"}
    funcion, _, props, _ = HERRAMIENTAS[nombre]
    argumentos = {k: v for k, v in (argumentos or {}).items() if k in props}  # ignora parámetros extra
    try:
        return funcion(repo, **argumentos)
    except (ArgumentoInvalido, ValueError, TypeError) as e:
        return {"error": str(e)}
