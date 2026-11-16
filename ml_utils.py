import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from pymongo import MongoClient
from dotenv import load_dotenv

# ==========================================
# 1. CONFIGURACIÓN DE BASE DE DATOS (CRUCIAL)
# ==========================================
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

try:
    client = MongoClient(MONGO_URI)
    # Asegúrate que este sea el nombre real de tu BD en Atlas
    db = client['denuncias_db'] 
    print("✅ Conexión a MongoDB exitosa en ml_utils")
except Exception as e:
    print(f"❌ Error conectando a MongoDB: {e}")
    db = None

# ==========================================
# 2. FUNCIONES DE PREDICCIÓN 2026 (TU CÓDIGO)
# ==========================================

def preparar_mensual(col, modalidad=None):
    """
    Función base: Extrae datos reales sumando la columna 'cantidad'.
    Soporta nombres de campos mixtos (ANIO/anio, MES/mes) por seguridad.
    """
    # 1. Filtro de modalidad (si se especifica)
    match_filter = {}
    if modalidad:
        match_filter["$or"] = [
            {"P_MODALIDADES": modalidad}, 
            {"modalidad": modalidad}
        ]

    # 2. Pipeline de Agregación
    pipeline = [
        { "$match": match_filter },
        {
            "$group": {
                "_id": { "anio": "$ANIO", "mes": "$MES" },
                "total": { "$sum": "$cantidad" } 
            }
        },
        { "$sort": { "_id.anio": 1, "_id.mes": 1 } }
    ]

    resultados = list(col.aggregate(pipeline))
    
    # Fallback: Si no trajo nada, probamos agrupar por minúsculas
    if not resultados:
        pipeline[1]["$group"]["_id"] = { "anio": "$anio", "mes": "$mes" }
        pipeline[1]["$group"]["total"] = { "$sum": "$cantidad" }
        resultados = list(col.aggregate(pipeline))

    # 3. Limpieza y estructuración
    datos = []
    for d in resultados:
        id_doc = d.get("_id", {})
        try:
            val_anio = int(id_doc.get("anio") or 0)
            val_mes = int(id_doc.get("mes") or 0)
            
            # Filtramos datos incoherentes
            if val_anio > 2000 and 1 <= val_mes <= 12:
                datos.append({
                    "anio": val_anio,
                    "mes": val_mes,
                    "total": d.get("total", 0)
                })
        except:
            continue

    return pd.DataFrame(datos)

# ==========================================
# 2. PROYECCIÓN 2026 CON SARIMA
# ==========================================
MESES_TXT = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
SARIMA_ORDEN = (0, 1, 1)
SARIMA_ESTACIONAL = (0, 1, 1, 12)   # "airline model" de Box-Jenkins
# Pandemia: caída y rebote atípicos. Se modela con un indicador por mes (intervención tipo pulso),
# así esos meses no inflan la varianza del modelo ni distorsionan la tendencia.
COVID_INICIO, COVID_FIN = "2020-03", "2021-02"
HORIZONTE_VALIDACION = 12            # meses que se reservan para medir el error

_cache_proyeccion = {"clave": None, "resultado": None}


def _mape(reales, predichos):
    reales, predichos = np.asarray(reales, float), np.asarray(predichos, float)
    return float(np.mean(np.abs((reales - predichos) / reales)) * 100)


def _ajustar_sarima(serie, exog):
    from statsmodels.tsa.statespace.sarimax import SARIMAX  # import diferido (carga pesada)
    return SARIMAX(serie, exog=exog, order=SARIMA_ORDEN, seasonal_order=SARIMA_ESTACIONAL,
                   enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)


def proyeccion_2026(col):
    """
    Proyecta las denuncias mensuales de 2026 con SARIMA(0,1,1)(0,1,1)12 e indicadores de
    intervención para los meses de la pandemia (mar-2020 a feb-2021). Valida el modelo con los
    últimos 12 meses reales (MAPE) y lo compara con la regresión lineal que se usaba antes.
    """
    import warnings

    df = preparar_mensual(col)
    if df.empty or len(df) < 36:
        return None

    serie = pd.Series(
        df["total"].astype(float).values,
        index=pd.period_range(f"{int(df.iloc[0]['anio'])}-{int(df.iloc[0]['mes']):02d}", periods=len(df), freq="M"),
    )
    clave = (len(serie), float(serie.iloc[-1]))
    if _cache_proyeccion["clave"] == clave:
        return _cache_proyeccion["resultado"]

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

        # --- 2) Modelo final con toda la serie y pronóstico hasta dic-2026 ---
        modelo = _ajustar_sarima(serie, covid(serie.index))
        ultimo = serie.index[-1]
        pasos = (2026 - ultimo.year) * 12 + (12 - ultimo.month)
        futuro_idx = pd.period_range(ultimo + 1, periods=pasos, freq="M")
        fc = modelo.get_forecast(pasos, exog=covid(futuro_idx))
        media = fc.predicted_mean.clip(lower=0)
        ic = fc.conf_int(alpha=0.05).clip(lower=0)   # banda mensual (95%)

        # IC del TOTAL anual: no se pueden sumar los límites mensuales (los errores están
        # correlacionados). Se simulan trayectorias del modelo y se toman percentiles del total.
        es_2026 = np.array([p.year == 2026 for p in futuro_idx])
        sims = modelo.simulate(nsimulations=pasos, repetitions=1000, anchor="end",
                               exog=covid(futuro_idx).values,
                               rng=np.random.default_rng(42))
        sims = np.asarray(sims).reshape(pasos, -1).clip(min=0)
        totales_sim = sims[es_2026].sum(axis=0)
        total_min, total_max = np.percentile(totales_sim, [2.5, 97.5])

    pred_2026 = media[es_2026]
    hist = serie[-24:]

    resultado = {
        "modelo": "SARIMA(0,1,1)(0,1,1)₁₂ con intervención COVID-19",
        "total_2026": int(pred_2026.sum()),
        "total_2026_min": int(total_min),
        "total_2026_max": int(total_max),
        "total_2025_real": int(serie[[p.year == 2025 for p in serie.index]].sum()),
        "ultimo_mes": f"{MESES_TXT[ultimo.month - 1]} {ultimo.year}",
        "mape_sarima": round(mape_sarima, 1),
        "mape_lineal": round(mape_lineal, 1),
        "meses_2026": MESES_TXT,
        "pred_2026": [int(v) for v in pred_2026.values],
        # Para el gráfico: últimos 24 meses reales + pronóstico completo
        "graf_etiquetas": [f"{MESES_TXT[p.month - 1]} {str(p.year)[2:]}" for p in list(hist.index) + list(futuro_idx)],
        "graf_real": [int(v) for v in hist.values] + [None] * pasos,
        "graf_pred": [None] * (len(hist) - 1) + [int(hist.iloc[-1])] + [int(v) for v in media.values],
        "graf_min": [None] * len(hist) + [int(v) for v in ic.iloc[:, 0].values],
        "graf_max": [None] * len(hist) + [int(v) for v in ic.iloc[:, 1].values],
        "historico_valores": [int(v) for v in serie[-12:].values],
        "historico_etiquetas": [f"{MESES_TXT[p.month - 1]} {p.year}" for p in serie.index[-12:]],
    }
    _cache_proyeccion.update(clave=clave, resultado=resultado)
    return resultado


def predecir_total_2026(col):
    """
    Compatibilidad: retorna total, etiquetas, predicciones, historico_valores, historico_etiquetas.
    """
    r = proyeccion_2026(col)
    if not r:
        return 0, [], [], [], []
    return r["total_2026"], r["meses_2026"], r["pred_2026"], r["historico_valores"], r["historico_etiquetas"]

def obtener_contexto_ia(col):
    r = proyeccion_2026(col)
    if not r:
        return 0, "Datos insuficientes"

    ultimos = ", ".join(f"{e}: {v:,}" for e, v in zip(r["historico_etiquetas"][-6:], r["historico_valores"][-6:]))
    texto_contexto = (
        f"Últimos 6 meses reales: {ultimos}. "
        f"Total real 2025 (hasta {r['ultimo_mes']}): {r['total_2025_real']:,}. "
        f"Proyección 2026 ({r['modelo']}): {r['total_2026']:,} "
        f"(IC 95%: {r['total_2026_min']:,} – {r['total_2026_max']:,}); "
        f"error de validación (MAPE) {r['mape_sarima']}%."
    )
    return r["total_2026"], texto_contexto

# ==========================================
# 3. FUNCIONES DEL SIMULADOR DE RIESGO (FALTABAN ESTAS)
# ==========================================

def entrenar_modelo_riesgo(col, modalidad_objetivo="Extorsión"):
    """
    Prepara el modelo Random Forest para el Agente Logístico
    """
    data = list(col.find(
        {"P_MODALIDADES": modalidad_objetivo},
        {"ANIO": 1, "trimestre": 1, "DPTO_HECHO_NEW": 1, "cantidad": 1, "_id": 0}
    ))
    
    if not data:
        return None, pd.DataFrame(), None

    df = pd.DataFrame(data)
    
    # Normalización de nombres (por si acaso vienen distintos)
    df.rename(columns={
        "ANIO": "anio", 
        "DPTO_HECHO_NEW": "departamento", 
        "cantidad": "total"
    }, inplace=True)

    # Limpieza básica
    df = df.dropna()
    df['anio'] = pd.to_numeric(df['anio'], errors='coerce').fillna(0).astype(int)
    
    # Mapeo de Trimestres
    trim_map = {"T1": 1, "T2": 2, "T3": 3, "T4": 4}
    df['trimestre_num'] = df['trimestre'].map(trim_map).fillna(1).astype(int)

    # Codificación de Departamentos
    le_dpto = LabelEncoder()
    df['dpto_code'] = le_dpto.fit_transform(df['departamento'].astype(str))

    # Entrenar Random Forest
    X = df[['anio', 'trimestre_num', 'dpto_code']]
    y = df['total']
    
    rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
    rf_model.fit(X, y)
    
    # Agregamos una columna 'periodo' para gráficos
    df['periodo'] = df['anio'].astype(str) + "-" + df['trimestre']
    
    return rf_model, df, le_dpto

def predecir_valor_especifico(modelo, le_dpto, anio, trimestre_str, departamento):
    """
    Usa el modelo entrenado para predecir un valor puntual
    """
    if not modelo: return 0
    
    try:
        trim_map = {"T1": 1, "T2": 2, "T3": 3, "T4": 4}
        trim_num = trim_map.get(trimestre_str, 1)
        
        # Manejo de error si el departamento no existe en el entrenamiento
        try:
            dpto_code = le_dpto.transform([departamento])[0]
        except:
            # Si es un dpto nuevo, usamos el código 0 o promedio (aquí simplificamos)
            dpto_code = 0 
            
        prediccion = modelo.predict([[anio, trim_num, dpto_code]])
        return int(max(prediccion[0], 0))
    except Exception as e:
        print(f"Error predicción específica: {e}")
        return 0