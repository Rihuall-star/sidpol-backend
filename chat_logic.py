# chat_logic.py
import re
from nlp_consulta import analizar_pregunta
from mongo_queries import (
    total_denuncias,
    modalidad_mas_frecuente,
    top_modalidades,
    ranking_departamentos,
    tendencia_modalidad,
    comparar_dos_anios,
)

def construir_contexto(col, mensaje):
    info = analizar_pregunta(mensaje)

    anio = info["anio"]
    modalidad = info["modalidad"]
    departamento = info["departamento"]
    intencion = info["intencion"]

    contexto = ""

    # 1) Total de denuncias
    if intencion == "total":
        total = total_denuncias(col, anio=anio, departamento=departamento, modalidad=modalidad)
        contexto = (
            "Datos calculados desde la base de datos de denuncias policiales del Perú:\n"
            f"Año: {anio or 'todos'}, "
            f"Departamento: {departamento or 'todos'}, "
            f"Modalidad: {modalidad or 'todas'}.\n"
            f"Total de denuncias: {total}.\n"
        )

    # 2) Modalidad más frecuente
    elif intencion == "top_modalidad":
        mod_top, total_top = modalidad_mas_frecuente(col, anio=anio, departamento=departamento)
        if mod_top:
            contexto = (
                "Cálculo de modalidad más frecuente a partir de la base de datos:\n"
                f"Año: {anio or 'todos'}, Departamento: {departamento or 'todos'}.\n"
                f"La modalidad con mayor número de denuncias es '{mod_top}' "
                f"con {total_top} casos.\n"
            )
        else:
            contexto = "No se encontraron denuncias para los filtros detectados.\n"

    # 3) Ranking de departamentos
    elif intencion == "ranking":
        ranking = ranking_departamentos(col, anio=anio, modalidad=modalidad, n=5)
        contexto = "Ranking de departamentos según la base de datos:\n"
        for r in ranking:
            contexto += f"- {r['_id']}: {r['total']} denuncias.\n"

    # 4) Tendencia histórica
    elif intencion == "tendencia" and departamento and modalidad:
        serie = tendencia_modalidad(col, departamento, modalidad)
        contexto = (
            "Evolución histórica de denuncias según la base de datos:\n"
            f"Departamento: {departamento}, Modalidad: {modalidad}.\n"
        )
        for p in serie:
            contexto += f"- Año {p['_id']}: {p['total']} denuncias.\n"

    # 5) Comparación entre dos años (cuando existan dos años en el texto)
    elif intencion == "comparacion" and departamento and modalidad:
        anios = re.findall(r"(2018|2019|2020|2021|2022|2023|2024|2025|2026)", mensaje)
        if len(anios) >= 2:
            a1, a2 = int(anios[0]), int(anios[1])
            t1, t2 = comparar_dos_anios(col, departamento, modalidad, a1, a2)
            contexto = (
                "Comparación de denuncias según la base de datos:\n"
                f"Departamento: {departamento}, Modalidad: {modalidad}.\n"
                f"- {a1}: {t1} denuncias.\n"
                f"- {a2}: {t2} denuncias.\n"
            )

    # Si no se reconoce la intención o faltan datos:
    if not contexto:
        contexto = (
            "No se detectó una intención específica o no hay suficientes datos para "
            "realizar un cálculo exacto. Responde la pregunta del usuario de forma general, "
            "y si habla de denuncias en Perú puedes usar tu conocimiento general.\n"
        )

    return contexto
