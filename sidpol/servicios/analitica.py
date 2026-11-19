"""
Cálculos de las secciones de SIDPOL (Panorama, Delitos y Territorio).

Cada función devuelve datos listos para graficar y "hallazgos" redactados a partir
de los propios datos, de modo que el texto de la web nunca queda desactualizado.

Son funciones puras: reciben una colección (la gold o la cruda, que tienen los mismos campos)
y devuelven diccionarios serializables. No guardan caché: de eso se encarga `resultados.py`.
"""
MODALIDADES_ORDEN = [
    "Extorsión", "Homicidio", "Estafa", "Hurto", "Robo",
    "Violencia contra la mujer e integrantes", "Otros",
]

# Población estimada por departamento (aprox.). Se usa solo para tasas por 100 mil habitantes.
POBLACION = {
    "MADRE DE DIOS": 184083, "AREQUIPA": 1523839, "LAMBAYEQUE": 1367029, "MOQUEGUA": 200973,
    "ICA": 1004829, "TUMBES": 280723, "TACNA": 397737, "APURIMAC": 436820, "JUNIN": 1418738,
    "CUSCO": 1428028, "ANCASH": 1202828, "HUANUCO": 782039, "AMAZONAS": 458022,
    "LA LIBERTAD": 2078028, "PIURA": 2138730, "AYACUCHO": 671182, "UCAYALI": 568028,
    "SAN MARTIN": 924292, "PASCO": 278028, "CAJAMARCA": 1503836, "PUNO": 1268093,
    "LORETO": 1138637, "HUANCAVELICA": 371038, "LIMA METROPOLITANA": 11810722,
    "REGION LIMA": 1092827, "PROV. CONST. DEL CALLAO": 1147628,
}

REGION_NATURAL = {
    **{d: "Costa" for d in ["TUMBES", "PIURA", "LAMBAYEQUE", "LA LIBERTAD", "ANCASH", "LIMA METROPOLITANA",
                            "REGION LIMA", "PROV. CONST. DEL CALLAO", "ICA", "AREQUIPA", "MOQUEGUA", "TACNA"]},
    **{d: "Sierra" for d in ["CAJAMARCA", "HUANUCO", "PASCO", "JUNIN", "HUANCAVELICA", "AYACUCHO",
                             "APURIMAC", "CUSCO", "PUNO"]},
    **{d: "Selva" for d in ["AMAZONAS", "LORETO", "SAN MARTIN", "UCAYALI", "MADRE DE DIOS"]},
}

NOMBRE_VISIBLE = {
    "LIMA METROPOLITANA": "Lima Metropolitana",
    "REGION LIMA": "Lima Provincias",
    "PROV. CONST. DEL CALLAO": "Callao",
}

DEPARTAMENTOS = sorted(POBLACION)

PARTICULAS = {"de", "del", "la", "las", "los", "el", "y"}


def titulo(texto):
    """'SAN JUAN DE LURIGANCHO' -> 'San Juan de Lurigancho'."""
    palabras = str(texto).lower().split()
    return " ".join(w if (i and w in PARTICULAS) else w.capitalize() for i, w in enumerate(palabras))


def nombre(dpto):
    return NOMBRE_VISIBLE.get(dpto, titulo(dpto))


def nombre_distrito(distrito, dpto):
    if distrito == "LIMA" and dpto == "LIMA METROPOLITANA":
        return "Cercado de Lima"
    return titulo(distrito)


def trimestre_completo(etiqueta, anio_fin, mes_fin):
    """Descarta el trimestre en curso si no tiene sus 3 meses (evita una caída falsa al final)."""
    try:
        anio, t = etiqueta.split("-T")
        anio, t = int(anio), int(t)
    except (ValueError, AttributeError):
        return False
    return not (anio == anio_fin and t * 3 > mes_fin)


def variacion(nuevo, viejo):
    return round((nuevo - viejo) / viejo * 100) if viejo else None


def _corte(col):
    """Último año y mes con datos (p. ej. 2025, 10) para comparar periodos equivalentes."""
    ultimo = list(col.aggregate([
        {"$group": {"_id": {"a": "$ANIO", "m": "$MES"}}},
        {"$sort": {"_id.a": -1, "_id.m": -1}}, {"$limit": 1},
    ]))
    if not ultimo:
        return None, None
    return int(ultimo[0]["_id"]["a"]), int(ultimo[0]["_id"]["m"])


MESES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]


# =====================================================================
# 1. PANORAMA
# =====================================================================
def panorama(col):
    anio_fin, mes_fin = _corte(col)
    if not anio_fin:
        return None
    anual = {int(d["_id"]): d["total"] for d in col.aggregate([
        {"$group": {"_id": "$ANIO", "total": {"$sum": "$cantidad"}}}]) if str(d["_id"]).isdigit()}
    trimestral = [d for d in col.aggregate([
        {"$group": {"_id": "$anio_trimestre", "total": {"$sum": "$cantidad"}}}, {"$sort": {"_id": 1}}])
        if d["_id"] and trimestre_completo(d["_id"], anio_fin, mes_fin)]
    parcial = {int(d["_id"]): d["total"] for d in col.aggregate([
        {"$match": {"ANIO": {"$in": [anio_fin - 1, anio_fin]}, "MES": {"$lte": mes_fin}}},
        {"$group": {"_id": "$ANIO", "total": {"$sum": "$cantidad"}}}])}

    anios = sorted(anual)
    completos = [a for a in anios if a != anio_fin] if mes_fin < 12 else anios
    pico = max(completos, key=lambda a: anual[a]) if completos else None
    var_parcial = variacion(parcial.get(anio_fin, 0), parcial.get(anio_fin - 1, 0))
    periodo = f"ene–{MESES[mes_fin - 1]}"

    return {
        "anio_fin": anio_fin, "mes_fin": mes_fin, "periodo": periodo,
        "anios": anios, "serie_anual": [anual[a] for a in anios],
        "trim_etiquetas": [d["_id"] for d in trimestral], "trim_valores": [d["total"] for d in trimestral],
        "total": sum(anual.values()), "pico": pico, "pico_total": anual.get(pico, 0),
        "parcial_actual": parcial.get(anio_fin, 0), "parcial_anterior": parcial.get(anio_fin - 1, 0),
        "var_parcial": var_parcial,
        "covid_caida": variacion(anual.get(2020, 0), anual.get(2019, 0)),
    }


# =====================================================================
# 2. DELITOS
# =====================================================================
def delitos(col):
    anio_fin, mes_fin = _corte(col)
    if not anio_fin:
        return None
    matriz = {}
    for d in col.aggregate([{"$group": {"_id": {"m": "$P_MODALIDADES", "a": "$ANIO"}, "t": {"$sum": "$cantidad"}}}]):
        m, a = d["_id"].get("m"), d["_id"].get("a")
        if m and a:
            matriz.setdefault(m, {})[int(a)] = d["t"]
    parcial = {}
    for d in col.aggregate([
        {"$match": {"ANIO": {"$in": [anio_fin - 1, anio_fin]}, "MES": {"$lte": mes_fin}}},
        {"$group": {"_id": {"m": "$P_MODALIDADES", "a": "$ANIO"}, "t": {"$sum": "$cantidad"}}}]):
        m, a = d["_id"].get("m"), d["_id"].get("a")
        if m:
            parcial.setdefault(m, {})[int(a)] = d["t"]

    anios = sorted({a for v in matriz.values() for a in v})
    ultimo_completo = anio_fin - 1 if mes_fin < 12 else anio_fin
    filas = []
    for m in [x for x in MODALIDADES_ORDEN if x in matriz] + [x for x in matriz if x not in MODALIDADES_ORDEN]:
        v = matriz[m]
        filas.append({
            "modalidad": m,
            "por_anio": [v.get(a, 0) for a in anios],
            "total": sum(v.values()),
            "var_2021": variacion(v.get(ultimo_completo, 0), v.get(2021, 0)),
            "var_2018": variacion(v.get(ultimo_completo, 0), v.get(2018, 0)),
            "parcial_actual": parcial.get(m, {}).get(anio_fin, 0),
            "parcial_anterior": parcial.get(m, {}).get(anio_fin - 1, 0),
            "var_parcial": variacion(parcial.get(m, {}).get(anio_fin, 0), parcial.get(m, {}).get(anio_fin - 1, 0)),
            "record": parcial.get(m, {}).get(anio_fin, 0) > max([v.get(a, 0) for a in anios if a < anio_fin] or [0]),
        })

    # Extorsión vs homicidio por trimestre (nacional)
    trim = {}
    for d in col.aggregate([
        {"$match": {"P_MODALIDADES": {"$in": ["Extorsión", "Homicidio"]}}},
        {"$group": {"_id": {"t": "$anio_trimestre", "m": "$P_MODALIDADES"}, "total": {"$sum": "$cantidad"}}}]):
        t, m = d["_id"].get("t"), d["_id"].get("m")
        if t and trimestre_completo(t, anio_fin, mes_fin):
            trim.setdefault(t, {})[m] = d["total"]
    trim_etq = sorted(trim)

    ext = next((f for f in filas if f["modalidad"] == "Extorsión"), None)
    hom = next((f for f in filas if f["modalidad"] == "Homicidio"), None)
    est = next((f for f in filas if f["modalidad"] == "Estafa"), None)
    caen = sorted([f for f in filas if f["var_parcial"] is not None and f["var_parcial"] < 0],
                  key=lambda f: f["var_parcial"])

    return {
        "anio_fin": anio_fin, "mes_fin": mes_fin, "periodo": f"ene–{MESES[mes_fin - 1]}",
        "ultimo_completo": ultimo_completo, "anios": anios, "filas": filas,
        "extorsion": ext, "homicidio": hom, "estafa": est, "caen": caen,
        "trim_etiquetas": trim_etq,
        "trim_extorsion": [trim[t].get("Extorsión", 0) for t in trim_etq],
        "trim_homicidio": [trim[t].get("Homicidio", 0) for t in trim_etq],
    }


# =====================================================================
# 3. TERRITORIO
# =====================================================================
def territorio(col, col_distritos=None):
    """`col_distritos` es la colección cruda: la única que tiene el detalle por distrito."""
    col_distritos = col if col_distritos is None else col_distritos
    anio_fin, mes_fin = _corte(col)
    if not anio_fin:
        return None
    ultimo_completo = anio_fin - 1 if mes_fin < 12 else anio_fin

    # --- Departamentos: total histórico, último año completo y tasa por 100 mil ---
    hist, anual = {}, {}
    for d in col.aggregate([{"$group": {"_id": {"p": "$DPTO_HECHO_NEW", "a": "$ANIO"}, "t": {"$sum": "$cantidad"}}}]):
        p, a = d["_id"].get("p"), d["_id"].get("a")
        if not p:
            continue
        hist[p] = hist.get(p, 0) + d["t"]
        if a == ultimo_completo:
            anual[p] = d["t"]
    total = sum(hist.values())
    departamentos = sorted(
        [{"codigo": p, "nombre": nombre(p), "total": t, "pct": round(t / total * 100, 1),
          "anual": anual.get(p, 0), "poblacion": POBLACION.get(p),
          "tasa": round(anual.get(p, 0) / POBLACION[p] * 100000) if POBLACION.get(p) else None,
          "region": REGION_NATURAL.get(p, "—")} for p, t in hist.items()],
        key=lambda x: -x["total"])

    regiones = {}
    for d in departamentos:
        regiones[d["region"]] = regiones.get(d["region"], 0) + d["total"]
    regiones = [{"region": r, "total": t, "pct": round(t / total * 100, 1)}
                for r, t in sorted(regiones.items(), key=lambda kv: -kv[1])]

    # --- Extorsión por distrito: concentración y focos ---
    dist = {}
    for d in col_distritos.aggregate([
        {"$match": {"P_MODALIDADES": "Extorsión", "ANIO": {"$in": [2021, ultimo_completo]}}},
        {"$group": {"_id": {"d": "$DIST_HECHO", "p": "$DPTO_HECHO_NEW", "a": "$ANIO"}, "t": {"$sum": "$cantidad"}}}]):
        k = (d["_id"].get("d"), d["_id"].get("p"))
        if k[0]:
            dist.setdefault(k, {})[d["_id"].get("a")] = d["t"]
    tot_ext = sum(v.get(ultimo_completo, 0) for v in dist.values())
    orden = sorted(dist.items(), key=lambda kv: -kv[1].get(ultimo_completo, 0))
    con_casos = sum(1 for _, v in orden if v.get(ultimo_completo, 0) > 0)
    acum, distritos_50 = 0, 0
    for _, v in orden:
        acum += v.get(ultimo_completo, 0)
        distritos_50 += 1
        if acum >= tot_ext * 0.5:
            break
    focos = [{"distrito": nombre_distrito(d, p), "departamento": nombre(p), "anterior": v.get(2021, 0),
              "actual": v.get(ultimo_completo, 0), "var": variacion(v.get(ultimo_completo, 0), v.get(2021, 0))}
             for (d, p), v in orden[:15]]

    # --- Delito principal por departamento (sin "Otros") ---
    mezcla = {}
    for d in col.aggregate([
        {"$match": {"ANIO": ultimo_completo, "P_MODALIDADES": {"$ne": "Otros"}}},
        {"$group": {"_id": {"p": "$DPTO_HECHO_NEW", "m": "$P_MODALIDADES"}, "t": {"$sum": "$cantidad"}}}]):
        p = d["_id"].get("p")
        if p:
            mezcla.setdefault(p, []).append((d["t"], d["_id"].get("m")))
    principal = {p: max(v)[1] for p, v in mezcla.items()}
    conteo_principal = {}
    for m in principal.values():
        conteo_principal[m] = conteo_principal.get(m, 0) + 1
    vcm_lider = conteo_principal.get("Violencia contra la mujer e integrantes", 0)

    return {
        "anio_fin": anio_fin, "ultimo_completo": ultimo_completo, "total": total,
        "departamentos": departamentos, "regiones": regiones,
        "focos": focos, "distritos_50": distritos_50, "distritos_con_casos": con_casos,
        "extorsion_total": tot_ext, "principal": principal, "conteo_principal": conteo_principal,
        "vcm_lider": vcm_lider, "n_departamentos": len(principal),
        "tasa_top": sorted([d for d in departamentos if d["tasa"]], key=lambda x: -x["tasa"])[:1],
    }


# =====================================================================
# 4. EVOLUCIÓN POR MODALIDAD (contexto para la IA)
# =====================================================================
def evolucion_modalidades(col):
    """Cómo cambió cada modalidad: 2021 frente al último año completo, y el año en curso frente al
    mismo periodo del año anterior. Ordenadas de mayor a menor volumen en el último año completo."""
    anio_fin, mes_fin = _corte(col)
    if not anio_fin:
        return []
    ultimo_completo = anio_fin - 1 if mes_fin < 12 else anio_fin

    def por_modalidad(match):
        res = {}
        for f in col.aggregate([
                {"$match": match},
                {"$group": {"_id": {"m": "$P_MODALIDADES", "a": "$ANIO"}, "total": {"$sum": "$cantidad"}}}]):
            m, a = f["_id"].get("m"), f["_id"].get("a")
            if m:
                res.setdefault(m, {})[int(a)] = f["total"]
        return res

    completos = por_modalidad({"ANIO": {"$in": [2021, ultimo_completo]}})
    parciales = por_modalidad({"ANIO": {"$in": [anio_fin - 1, anio_fin]}, "MES": {"$lte": mes_fin}})
    filas = []
    for m in sorted(completos, key=lambda m: completos[m].get(ultimo_completo, 0), reverse=True):
        c, p = completos.get(m, {}), parciales.get(m, {})
        filas.append({
            "modalidad": m, "anio_base": 2021, "total_base": c.get(2021, 0),
            "anio_completo": ultimo_completo, "total_completo": c.get(ultimo_completo, 0),
            "var_completo": variacion(c.get(ultimo_completo, 0), c.get(2021, 0)),
            "periodo": f"ene–{MESES[mes_fin - 1]}", "anio_actual": anio_fin,
            "var_periodo": variacion(p.get(anio_fin, 0), p.get(anio_fin - 1, 0)),
        })
    return filas


def texto_evolucion(filas, top=10):
    """Resumen en texto de `evolucion_modalidades` para enviarlo como contexto a la IA."""
    def pct(v):
        return f"{v:+d}%" if v is not None else "n/d"

    lineas = [
        f"{f['modalidad']}: {f['anio_base']}={f['total_base']:,} → {f['anio_completo']}={f['total_completo']:,} "
        f"({pct(f['var_completo'])}); {f['periodo']} {f['anio_actual']} vs {f['anio_actual'] - 1}: {pct(f['var_periodo'])}"
        for f in filas[:top]
    ]
    return "EVOLUCIÓN POR MODALIDAD:\n" + "\n".join(lineas) + "\n"
