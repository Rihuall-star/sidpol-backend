"""
Modelos de SIDPOL: proyección 2026 (SARIMA), niveles de riesgo (K-Means), datos del mapa y
simulador táctico (Random Forest).

Funciones puras sobre una colección de MongoDB. El resultado de las más pesadas se guarda con
`resultados.py`, así no se vuelven a ajustar en cada visita ni después de que el servidor despierta.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

from sidpol.servicios import analitica

MESES_TXT = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]


# =====================================================================
# 1. PROYECCIÓN 2026 CON SARIMA
# =====================================================================
SARIMA_ORDEN = (0, 1, 1)
SARIMA_ESTACIONAL = (0, 1, 1, 12)   # "airline model" de Box-Jenkins
# Pandemia: caída y rebote atípicos. Se modela con un indicador por mes (intervención tipo pulso),
# así esos meses no inflan la varianza del modelo ni distorsionan la tendencia.
COVID_INICIO, COVID_FIN = "2020-03", "2021-02"
HORIZONTE_VALIDACION = 12            # meses que se reservan para medir el error


def preparar_mensual(col, modalidad=None):
    """Serie mensual nacional (suma del campo `cantidad`), opcionalmente de una modalidad."""
    filtro = {"P_MODALIDADES": modalidad} if modalidad else {}
    filas = col.aggregate([
        {"$match": filtro},
        {"$group": {"_id": {"anio": "$ANIO", "mes": "$MES"}, "total": {"$sum": "$cantidad"}}},
        {"$sort": {"_id.anio": 1, "_id.mes": 1}},
    ])
    datos = []
    for d in filas:
        try:
            anio, mes = int(d["_id"].get("anio") or 0), int(d["_id"].get("mes") or 0)
        except (TypeError, ValueError):
            continue
        if anio > 2000 and 1 <= mes <= 12:   # descarta registros incoherentes
            datos.append({"anio": anio, "mes": mes, "total": d.get("total", 0)})
    return pd.DataFrame(datos)


def _mape(reales, predichos):
    reales, predichos = np.asarray(reales, float), np.asarray(predichos, float)
    return float(np.mean(np.abs((reales - predichos) / reales)) * 100)


def _ajustar_sarima(serie, exog):
    from statsmodels.tsa.statespace.sarimax import SARIMAX  # import diferido (carga pesada)
    return SARIMAX(serie, exog=exog, order=SARIMA_ORDEN, seasonal_order=SARIMA_ESTACIONAL,
                   enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)


def proyeccion(col):
    """
    Proyecta las denuncias del año siguiente al último dato (y el cierre del año en curso) con
    SARIMA(0,1,1)(0,1,1)12 e indicadores de intervención para la pandemia (mar-2020 a feb-2021).
    Valida el modelo con los últimos 12 meses reales (MAPE) y lo compara con una regresión lineal.

    Ojo con el registro tardío: si los últimos meses del dataset todavía se están completando,
    la serie muestra una caída falsa al final. Por eso se recomienda usar versiones con meses cerrados.
    """
    df = preparar_mensual(col)
    if df.empty or len(df) < 36:
        return None

    serie = pd.Series(
        df["total"].astype(float).values,
        index=pd.period_range(f"{int(df.iloc[0]['anio'])}-{int(df.iloc[0]['mes']):02d}", periods=len(df), freq="M"),
    )
    meses_covid = pd.period_range(COVID_INICIO, COVID_FIN, freq="M")

    def covid(indice):
        return pd.DataFrame({str(p): (indice == p).astype(float) for p in meses_covid}, index=indice)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # --- 1) Validación: entrenar sin los últimos 12 meses y predecirlos ---
        H = HORIZONTE_VALIDACION
        entreno, prueba = serie[:-H], serie[-H:]
        val = _ajustar_sarima(entreno, covid(entreno.index))
        pred_val = val.get_forecast(H, exog=covid(prueba.index)).predicted_mean
        mape_sarima = _mape(prueba.values, pred_val.values)

        t = np.arange(len(serie)).reshape(-1, 1)
        lineal = LinearRegression().fit(t[:-H], entreno.values)
        mape_lineal = _mape(prueba.values, lineal.predict(t[-H:]))

        # --- 2) Modelo final con toda la serie y pronóstico hasta diciembre del año objetivo ---
        modelo = _ajustar_sarima(serie, covid(serie.index))
        ultimo = serie.index[-1]
        anio_actual, anio_objetivo = ultimo.year, ultimo.year + 1
        pasos = 12 + (12 - ultimo.month)
        futuro_idx = pd.period_range(ultimo + 1, periods=pasos, freq="M")
        fc = modelo.get_forecast(pasos, exog=covid(futuro_idx))
        media = fc.predicted_mean.clip(lower=0)
        ic = fc.conf_int(alpha=0.05).clip(lower=0)   # banda mensual (95%)

        # IC del TOTAL anual: no se pueden sumar los límites mensuales (los errores están
        # correlacionados). Se simulan trayectorias del modelo y se toman percentiles del total.
        es_objetivo = np.array([p.year == anio_objetivo for p in futuro_idx])
        sims = modelo.simulate(nsimulations=pasos, repetitions=1000, anchor="end",
                               exog=covid(futuro_idx).values, rng=np.random.default_rng(42))
        sims = np.asarray(sims).reshape(pasos, -1).clip(min=0)
        totales_sim = sims[es_objetivo].sum(axis=0)
        total_min, total_max = np.percentile(totales_sim, [2.5, 97.5])

    pred_objetivo = media[es_objetivo]
    real_actual = int(serie[[p.year == anio_actual for p in serie.index]].sum())
    resto_actual = int(media[~es_objetivo].sum())   # meses que faltan del año en curso
    hist = serie[-24:]
    return {
        "modelo": "SARIMA(0,1,1)(0,1,1)₁₂ con intervención COVID-19",
        "anio_objetivo": anio_objetivo,
        "total_objetivo": int(pred_objetivo.sum()),
        "total_objetivo_min": int(total_min),
        "total_objetivo_max": int(total_max),
        "anio_actual": anio_actual,
        "real_actual": real_actual,
        "cierre_actual": real_actual + resto_actual if ultimo.month < 12 else real_actual,
        "anio_completo": ultimo.month == 12,
        "ultimo_mes": f"{MESES_TXT[ultimo.month - 1]} {ultimo.year}",
        "mape_sarima": round(mape_sarima, 1),
        "mape_lineal": round(mape_lineal, 1),
        "meses_objetivo": MESES_TXT,
        "pred_objetivo": [int(v) for v in pred_objetivo.values],
        # Para el gráfico: últimos 24 meses reales + pronóstico completo
        "graf_etiquetas": [f"{MESES_TXT[p.month - 1]} {str(p.year)[2:]}" for p in list(hist.index) + list(futuro_idx)],
        "graf_real": [int(v) for v in hist.values] + [None] * pasos,
        "graf_pred": [None] * (len(hist) - 1) + [int(hist.iloc[-1])] + [int(v) for v in media.values],
        "graf_min": [None] * len(hist) + [int(v) for v in ic.iloc[:, 0].values],
        "graf_max": [None] * len(hist) + [int(v) for v in ic.iloc[:, 1].values],
        "historico_valores": [int(v) for v in serie[-12:].values],
        "historico_etiquetas": [f"{MESES_TXT[p.month - 1]} {p.year}" for p in serie.index[-12:]],
    }


def texto_proyeccion(r):
    """Resumen de la proyección para el contexto de la IA."""
    if not r:
        return "Proyección no disponible (datos insuficientes)."
    ultimos = ", ".join(f"{e}: {v:,}" for e, v in zip(r["historico_etiquetas"][-6:], r["historico_valores"][-6:]))
    cierre = ("" if r["anio_completo"] else
              f"Cierre estimado de {r['anio_actual']} (real + pronóstico de los meses que faltan): {r['cierre_actual']:,}. ")
    return (
        f"Últimos 6 meses reales: {ultimos}. "
        f"Total real {r['anio_actual']} (hasta {r['ultimo_mes']}): {r['real_actual']:,}. {cierre}"
        f"Proyección {r['anio_objetivo']} ({r['modelo']}): {r['total_objetivo']:,} "
        f"(IC 95%: {r['total_objetivo_min']:,} – {r['total_objetivo_max']:,}); "
        f"error de validación (MAPE) {r['mape_sarima']}%."
    )


# =====================================================================
# 2. NIVELES DE RIESGO (K-MEANS) Y MAPA
# =====================================================================
def _totales_departamento(col):
    filas = col.aggregate([{"$group": {"_id": "$DPTO_HECHO_NEW", "total": {"$sum": "$cantidad"}}},
                           {"$sort": {"total": -1}}])
    return [d for d in filas if d["_id"]]   # sin departamento no se puede ubicar


def clusters_departamentos(col):
    """Clasifica los departamentos en 3 niveles de riesgo con K-Means sobre su volumen de denuncias."""
    datos = sorted(_totales_departamento(col), key=lambda d: d["total"])
    if len(datos) < 3:
        return {"clusters": [], "stats": {"bajo": 0, "medio": 0, "alto": 0}}

    valores = np.array([d["total"] for d in datos], dtype=float).reshape(-1, 1)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(valores)
    # Se ordenan los grupos por su centroide: 0 = bajo, 1 = medio, 2 = alto
    orden = {viejo: nuevo for nuevo, viejo in enumerate(np.argsort(kmeans.cluster_centers_.flatten()))}
    colores, etiquetas = ["#10b981", "#f59e0b", "#f43f5e"], ["Bajo", "Medio", "Alto"]

    clusters = []
    for d, etiqueta in zip(datos, kmeans.labels_):
        idx = orden[etiqueta]
        clusters.append({"departamento": d["_id"], "total": int(d["total"]), "cluster": idx,
                         "color": colores[idx], "etiqueta": etiquetas[idx]})
    stats = {nivel: sum(1 for c in clusters if c["cluster"] == i) for i, nivel in enumerate(("bajo", "medio", "alto"))}
    return {"clusters": clusters, "stats": stats}


# Códigos verificados en @highcharts/map-collection 2.3.0 (countries/pe/pe-all)
CODIGOS_MAPA = {
    "LIMA METROPOLITANA": "pe-145",      # 'Lima Province' (distrito capital)
    "REGION LIMA": "pe-lr",              # Lima Provincias (alrededor de la capital)
    "LIMA": "pe-145",
    # En pe-all, 'pe-cl' se llama "Callao" pero sus coordenadas (-15.2, -70.0) son de PUNO.
    # El Callao real es 'pe-3341' (-12.09, -77.22).
    "PROV. CONST. DEL CALLAO": "pe-3341", "CALLAO": "pe-3341", "PUNO": "pe-cl",
    "AMAZONAS": "pe-am", "ANCASH": "pe-an", "APURIMAC": "pe-ap", "AREQUIPA": "pe-ar",
    "AYACUCHO": "pe-ay", "CAJAMARCA": "pe-cj", "CUSCO": "pe-cs", "HUANCAVELICA": "pe-hv",
    "HUANUCO": "pe-hc", "ICA": "pe-ic", "JUNIN": "pe-ju", "LA LIBERTAD": "pe-ll",
    "LAMBAYEQUE": "pe-lb", "LORETO": "pe-lo", "MADRE DE DIOS": "pe-md", "MOQUEGUA": "pe-mq",
    "PASCO": "pe-pa", "PIURA": "pe-pi", "SAN MARTIN": "pe-sm", "TACNA": "pe-ta",
    "TUMBES": "pe-tu", "UCAYALI": "pe-uc",
}


def datos_mapa(col):
    """Puntos del mapa de calor por departamento, con el nombre corregido para mostrar."""
    puntos = []
    for d in _totales_departamento(col):
        nombre_bd = str(d["_id"]).strip().upper()
        codigo = CODIGOS_MAPA.get(nombre_bd)
        if codigo:
            puntos.append({"hc-key": codigo, "value": d["total"], "name": analitica.nombre(nombre_bd)})
    return puntos


# =====================================================================
# 3. SIMULADOR TÁCTICO (RANDOM FOREST)
# =====================================================================
TRIMESTRES = ["T1", "T2", "T3", "T4"]


def datos_simulador(col, modalidad="Extorsión"):
    """Denuncias de una modalidad por departamento y trimestre (base del simulador y de su gráfico)."""
    filas = col.aggregate([
        {"$match": {"P_MODALIDADES": modalidad}},
        {"$group": {"_id": {"a": "$ANIO", "t": "$trimestre", "p": "$DPTO_HECHO_NEW"},
                    "total": {"$sum": "$cantidad"}}},
    ])
    registros = [{"anio": int(d["_id"]["a"]), "trimestre": d["_id"]["t"], "departamento": d["_id"]["p"],
                  "total": int(d["total"])}
                 for d in filas if d["_id"].get("p") and d["_id"].get("t") in TRIMESTRES and d["_id"].get("a")]
    if not registros:
        return None
    df = pd.DataFrame(registros)
    nacional = (df.groupby(["anio", "trimestre"])["total"].sum().reset_index()
                .sort_values(["anio", "trimestre"]))
    return {
        "modalidad": modalidad,
        "registros": registros,
        "labels": [f"{a}-{t}" for a, t in zip(nacional["anio"], nacional["trimestre"])],
        "valores": [int(v) for v in nacional["total"]],
        "departamentos": sorted(df["departamento"].unique().tolist()),
        "anios": sorted({int(a) for a in df["anio"]} | {2026}),
    }


def entrenar_simulador(datos):
    """Random Forest que estima las denuncias de un departamento en un trimestre."""
    df = pd.DataFrame(datos["registros"])
    codificador = LabelEncoder().fit(df["departamento"])
    X = np.column_stack([df["anio"], df["trimestre"].map(TRIMESTRES.index) + 1,
                         codificador.transform(df["departamento"])])
    modelo = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, df["total"])
    return modelo, codificador


def predecir_simulador(modelo, codificador, anio, trimestre, departamento):
    if departamento not in codificador.classes_ or trimestre not in TRIMESTRES:
        raise ValueError("Departamento o trimestre no válido")
    x = [[int(anio), TRIMESTRES.index(trimestre) + 1, codificador.transform([departamento])[0]]]
    return int(max(modelo.predict(x)[0], 0))
