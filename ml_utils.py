import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def preparar_mensual(col, modalidad=None):
    # Filtro dinámico
    match_filter = {}
    if modalidad:
        # Buscamos en el campo real de la nube
        match_filter["P_MODALIDADES"] = modalidad

    pipeline = [
        { "$match": match_filter },
        {
            "$group": {
                # Mapeamos a minúsculas para el DataFrame
                "_id": { "anio": "$ANIO", "mes": "$MES" },
                "total": { "$sum": 1 }
            }
        },
        { "$sort": { "_id.anio": 1, "_id.mes": 1 } }
    ]

    resultados = list(col.aggregate(pipeline))
    
    datos = []
    for d in resultados:
        # Acceso seguro con .get()
        id_data = d.get("_id", {})
        datos.append({
            "anio": id_data.get("anio") or 0,
            "mes": id_data.get("mes") or 0,
            "total": d.get("total") or 0
        })

    return pd.DataFrame(datos)

def predecir_total_2026(col):
    df = preparar_mensual(col)
    if df.empty or len(df) < 12:
        return 0, [], [], [], []

    # Crear índice de tiempo (1, 2, 3...)
    df['time_index'] = np.arange(len(df))
    
    X = df[['time_index']]
    y = df['total']
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Predecir los 12 meses de 2026
    # Asumiendo que los datos llegan hasta 2024 o 2025
    ultimo_idx = df['time_index'].max()
    proximos_indices = np.array(range(ultimo_idx + 1, ultimo_idx + 13)).reshape(-1, 1)
    predicciones = model.predict(proximos_indices)
    
    total_2026 = int(predicciones.sum())
    
    # Retornamos datos para las gráficas
    meses_nombres = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
    return total_2026, meses_nombres, predicciones.tolist(), df['total'].tolist()[-12:], df['anio'].tolist()[-12:]