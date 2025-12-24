# mongo_queries.py

def total_denuncias(col, anio=None, departamento=None, modalidad=None):
    filtros = {}
    if anio:
        filtros["ANIO"] = anio
    if departamento:
        filtros["DPTO_HECHO_NEW"] = departamento
    if modalidad:
        filtros["P_MODALIDADES"] = modalidad

    pipeline = [
        {"$match": filtros},
        {"$group": {"_id": None, "total": {"$sum": "$cantidad"}}}
    ]

    res = list(col.aggregate(pipeline))
    return res[0]["total"] if res else 0


def modalidad_mas_frecuente(col, anio=None, departamento=None):
    filtros = {}
    if anio:
        filtros["ANIO"] = anio
    if departamento:
        filtros["DPTO_HECHO_NEW"] = departamento

    pipeline = [
        {"$match": filtros},
        {"$group": {"_id": "$P_MODALIDADES", "total": {"$sum": "$cantidad"}}},
        {"$sort": {"total": -1}},
        {"$limit": 1}
    ]

    res = list(col.aggregate(pipeline))
    if res:
        return res[0]["_id"], res[0]["total"]
    return None, 0


def top_modalidades(col, anio=None, departamento=None, n=5):
    filtros = {}
    if anio:
        filtros["ANIO"] = anio
    if departamento:
        filtros["DPTO_HECHO_NEW"] = departamento

    pipeline = [
        {"$match": filtros},
        {"$group": {"_id": "$P_MODALIDADES", "total": {"$sum": "$cantidad"}}},
        {"$sort": {"total": -1}},
        {"$limit": n}
    ]

    return list(col.aggregate(pipeline))


def ranking_departamentos(col, anio=None, modalidad=None, n=10):
    filtros = {}
    if anio:
        filtros["ANIO"] = anio
    if modalidad:
        filtros["P_MODALIDADES"] = modalidad

    pipeline = [
        {"$match": filtros},
        {"$group": {"_id": "$DPTO_HECHO_NEW", "total": {"$sum": "$cantidad"}}},
        {"$sort": {"total": -1}},
        {"$limit": n}
    ]

    return list(col.aggregate(pipeline))


def tendencia_modalidad(col, departamento, modalidad):
    pipeline = [
        {"$match": {
            "DPTO_HECHO_NEW": departamento,
            "P_MODALIDADES": modalidad
        }},
        {"$group": {"_id": "$ANIO", "total": {"$sum": "$cantidad"}}},
        {"$sort": {"_id": 1}}
    ]
    return list(col.aggregate(pipeline))


def comparar_dos_anios(col, departamento, modalidad, anio1, anio2):
    pipeline = [
        {"$match": {
            "DPTO_HECHO_NEW": departamento,
            "P_MODALIDADES": modalidad,
            "ANIO": {"$in": [anio1, anio2]}
        }},
        {"$group": {"_id": "$ANIO", "total": {"$sum": "$cantidad"}}}
    ]

    res = {r["_id"]: r["total"] for r in col.aggregate(pipeline)}
    return res.get(anio1, 0), res.get(anio2, 0)
