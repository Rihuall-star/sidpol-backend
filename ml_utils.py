import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def preparar_mensual(col, modalidad=None):
    # 1. Filtro: Usamos el nombre real 'modalidad' (minúscula)
    match_filter = {}
    if modalidad:
        match_filter["modalidad"] = modalidad

    # 2. Pipeline: Agrupamos por Año y Trimestre
    pipeline = [
        { "$match": match_filter },
        {
            "$group": {
                # Agrupamos por los campos REALES que vimos en el espía
                "_id": { 
                    "anio": "$anio", 
                    "trimestre": "$trimestre" 
                }, 
                # Sumamos el campo 'prediccion' (que vale 30, 40, etc.)
                "total": { "$sum": "$prediccion" } 
            }
        },
        { "$sort": { "_id.anio": 1, "_id.trimestre": 1 } }
    ]

    resultados = list(col.aggregate(pipeline))
    
    datos = []
    # Mapa para convertir T1 -> 1, T2 -> 2, etc.
    trimestre_map = {"T1": 1, "T2": 2, "T3": 3, "T4": 4}
    
    for d in resultados:
        id_doc = d.get("_id", {})
        trim_str = id_doc.get("trimestre", "T1")
        
        datos.append({
            "anio": id_doc.get("anio", 0),
            "trimestre_str": trim_str,
            "trimestre_num": trimestre_map.get(trim_str, 1),
            "total": d.get("total", 0)
        })

    return pd.DataFrame(datos)

def predecir_total_2026(col):
    df = preparar_mensual(col)
    
    # Si no hay datos, devolvemos 0
    if df.empty:
        return 0, [], [], [], []

    # Crear índice de tiempo continuo (Año + Trimestre fraccional)
    # 2018 T1 = 2018.00 | 2018 T2 = 2018.25 | ...
    df['time_index'] = df['anio'] + (df['trimestre_num'] - 1) * 0.25
    
    X = df[['time_index']]
    y = df['total']
    
    # Entrenar modelo (incluso si hay pocos datos)
    model = LinearRegression()
    model.fit(X, y)
    
    # Proyectar los 4 trimestres de 2026
    # 2026.0 (T1), 2026.25 (T2), 2026.50 (T3), 2026.75 (T4)
    futuro = np.array([2026.0, 2026.25, 2026.50, 2026.75]).reshape(-1, 1)
    
    predicciones = model.predict(futuro)
    predicciones = np.maximum(predicciones, 0) # Evitar negativos
    
    total_2026 = int(predicciones.sum())
    
    # Etiquetas y valores para el gráfico
    etiquetas_grafico = ["2026-T1", "2026-T2", "2026-T3", "2026-T4"]
    valores_grafico = [int(x) for x in predicciones]
    
    # Datos históricos para contexto (últimos 4 registros reales)
    historico = df['total'].tail(4).tolist()
    anios_hist = df['anio'].tail(4).tolist()
    
    return total_2026, etiquetas_grafico, valores_grafico, historico, anios_hist