import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def preparar_mensual(col, modalidad=None):
    match_filter = {}
    if modalidad:
        match_filter["P_MODALIDADES"] = modalidad

    pipeline = [
        { "$match": match_filter },
        {
            "$group": {
                "_id": { "anio": "$ANIO", "mes": "$MES" },
                "total": { "$sum": 1 }
            }
        },
        { "$sort": { "_id.anio": 1, "_id.mes": 1 } }
    ]
    resultados = list(col.aggregate(pipeline))
    datos = []
    for d in resultados:
        id_doc = d.get("_id", {})
        datos.append({
            "anio": id_doc.get("anio") or 0,
            "mes": id_doc.get("mes") or 0,
            "total": d.get("total", 0)
        })
    return pd.DataFrame(datos)

def predecir_total_2026(col):
    df = preparar_mensual(col)
    if df.empty or len(df) < 6: # Bajamos el requisito a 6 meses para que no de error si hay pocos datos
        return 0, [], [], [], []

    df['time_index'] = np.arange(len(df))
    X = df[['time_index']]
    y = df['total']
    
    model = LinearRegression()
    model.fit(X, y)
    
    ultimo_idx = df['time_index'].max()
    proximos_indices = np.array(range(ultimo_idx + 1, ultimo_idx + 13)).reshape(-1, 1)
    predicciones = model.predict(proximos_indices)
    
    total_2026 = int(max(0, predicciones.sum())) # max(0) evita números negativos
    meses_nombres = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
    
    return total_2026, meses_nombres, predicciones.tolist(), df['total'].tolist()[-12:], df['anio'].tolist()[-12:]