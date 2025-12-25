import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def preparar_mensual(col, modalidad=None):
    # 1. Filtro
    match_filter = {}
    if modalidad:
        match_filter["modalidad"] = modalidad  # Nombre real: 'modalidad'

    # 2. Pipeline: Agrupamos por Año y Trimestre
    pipeline = [
        { "$match": match_filter },
        {
            "$group": {
                "_id": { "anio": "$anio", "trimestre": "$trimestre" }, # Nombres reales
                # Sumamos el campo 'prediccion' porque los datos ya vienen agrupados
                "total": { "$sum": "$prediccion" } 
            }
        },
        { "$sort": { "_id.anio": 1, "_id.trimestre": 1 } }
    ]

    resultados = list(col.aggregate(pipeline))
    
    datos = []
    trimestre_map = {"T1": 1, "T2": 2, "T3": 3, "T4": 4}
    
    for d in resultados:
        id_doc = d.get("_id", {})
        trim = id_doc.get("trimestre", "T1")
        
        datos.append({
            "anio": id_doc.get("anio", 0),
            "trimestre_num": trimestre_map.get(trim, 1), # Convertimos T1 a 1
            "total": d.get("total", 0)
        })

    return pd.DataFrame(datos)

def predecir_total_2026(col):
    df = preparar_mensual(col)
    
    if df.empty or len(df) < 2:
        return 0, [], [], [], []

    # Crear índice de tiempo continuo (Año + Trimestre)
    # Ejemplo: 2018 T1 -> 2018.0, 2018 T2 -> 2018.25
    df['time_index'] = df['anio'] + (df['trimestre_num'] - 1) * 0.25
    
    X = df[['time_index']]
    y = df['total']
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Proyectar para 2026 (4 trimestres)
    # Generamos los índices de tiempo para 2026: 2026.0, 2026.25, 2026.5, 2026.75
    futuro_indices = np.array([2026.0, 2026.25, 2026.50, 2026.75]).reshape(-1, 1)
    
    predicciones = model.predict(futuro_indices)
    predicciones = np.maximum(predicciones, 0) # Evitar negativos
    
    total_2026 = int(predicciones.sum())
    
    # Etiquetas para el gráfico (Trimestres 2026)
    etiquetas = ["2026-T1", "2026-T2", "2026-T3", "2026-T4"]
    
    # Devolvemos datos históricos recientes para contexto (últimos 4 registros)
    return total_2026, etiquetas, predicciones.tolist(), df['total'].tolist()[-4:], df['anio'].tolist()[-4:]