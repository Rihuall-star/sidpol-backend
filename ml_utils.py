# ml_utils.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def preparar_mensual(col, modalidad=None):
    # --- PASO 1: DEFINIMOS LA VARIABLE PRIMERO ---
    match_filter = {}
    if modalidad:
        match_filter["P_MODALIDADES"] = modalidad

    # --- PASO 2: AHORA SÍ LA USAMOS EN EL PIPELINE ---
    pipeline = [
        {
            "$match": match_filter  # <--- Python ya la conoce porque la definimos arriba
        },
        {
            "$group": {
                "_id": { "anio": "$ANIO", "mes": "$MES" },
                "total": { "$sum": 1 }
            }
        },
        {
            "$sort": { "_id.anio": 1, "_id.mes": 1 }
        }
    ]

    # --- PASO 3: EJECUTAMOS ---
    resultados = list(col.aggregate(pipeline))
    
    datos_procesados = []
    for d in resultados:
        datos_procesados.append({
            "anio": d["_id"]["anio"],
            "mes": d["_id"]["mes"],
            "total": d["total"]
        })

    import pandas as pd
    df = pd.DataFrame(datos_procesados)
    return df

    # --- PASO 3: EJECUTAR ---
    resultados = list(col.aggregate(pipeline))
    
    datos_procesados = []
    for d in resultados:
        datos_procesados.append({
            "anio": d["_id"]["anio"],
            "mes": d["_id"]["mes"],
            "total": d["total"]
        })

    import pandas as pd
    df = pd.DataFrame(datos_procesados)
    return df


def predecir_total_2026(col):
    """
    Regresión lineal sobre la serie mensual total (t vs total).
    Devuelve:
    - total_pred_2026: suma de las predicciones de julio a diciembre 2026
    - hist_labels: etiquetas de la serie histórica (ej. '2018-01', '2018-02', ...)
    - hist_values: valores históricos
    - pred_labels: etiquetas de los meses predichos (ej. '2026-07' ... '2026-12')
    - pred_values: valores predichos por mes
    """
    df = preparar_mensual(col, modalidad=None)
    if df.empty:
        return 0, [], [], [], []

    # Etiquetas históricas tipo 'YYYY-MM'
    df["label"] = df["anio"].astype(str) + "-" + df["mes"].astype(str).str.zfill(2)

    X = df[["t"]].values
    y = df["total"].values

    modelo = LinearRegression()
    modelo.fit(X, y)

    t_last = int(df["t"].max())

    # Futuro: 6 puntos (julio a diciembre 2026)
    t_future = np.arange(t_last + 1, t_last + 7).reshape(-1, 1)
    y_pred = modelo.predict(t_future)

    # Etiquetas futuras fijas: 2026-07 a 2026-12
    pred_labels = [f"2026-{mes:02d}" for mes in range(7, 13)]
    pred_values = y_pred.tolist()

    total_pred_2026 = float(sum(pred_values))

    hist_labels = df["label"].tolist()
    hist_values = df["total"].tolist()

    return total_pred_2026, hist_labels, hist_values, pred_labels, pred_values



def predecir_2026_por_modalidad(col, modalidades):
    """
    Devuelve una lista de diccionarios:
    - modalidad
    - total_pred_2026  (suma de julio–diciembre 2026)
    """
    resultados = []

    for mod in modalidades:
        df = preparar_mensual(col, modalidad=mod)
        if df.empty:
            resultados.append({"modalidad": mod, "total_pred_2026": 0})
            continue

        X = df[["t"]].values
        y = df["total"].values

        modelo = LinearRegression()
        modelo.fit(X, y)

        t_last = df["t"].max()
        t_future = np.arange(t_last + 1, t_last + 7).reshape(-1, 1)
        y_pred = modelo.predict(t_future)
        total_pred_2026 = float(y_pred.sum())

        resultados.append({
            "modalidad": mod,
            "total_pred_2026": round(total_pred_2026)
        })

    return resultados
