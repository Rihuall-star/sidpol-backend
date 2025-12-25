import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def preparar_mensual(col, modalidad=None):
    # 1. Filtro (Intentamos el nombre más común en bases grandes)
    match_filter = {}
    if modalidad:
        match_filter["P_MODALIDADES"] = modalidad # Probamos mayúscula primero

    # 2. Pipeline Híbrido: Intenta agrupar por campos comunes
    # NOTA: En 'denuncias' solemos contar documentos ($sum: 1)
    pipeline = [
        { "$match": match_filter },
        {
            "$group": {
                "_id": { 
                    "anio": "$ANIO", # Intentamos mayúsculas (común en datasets peruanos)
                    "mes": "$MES"
                }, 
                "total": { "$sum": 1 } # ¡AQUÍ ESTÁ LA CLAVE! Contamos 1 por 1.
            }
        },
        { "$sort": { "_id.anio": 1, "_id.mes": 1 } }
    ]

    resultados = list(col.aggregate(pipeline))
    
    # Si la lista está vacía, es posible que los campos sean minúsculas.
    # Intentamos el "Plan B" (minúsculas) si falló el primero.
    if not resultados:
        pipeline[1]["$group"]["_id"] = { "anio": "$anio", "mes": "$mes" }
        resultados = list(col.aggregate(pipeline))

    datos = []
    for d in resultados:
        id_doc = d.get("_id", {})
        # Limpieza de datos
        try:
            val_anio = int(id_doc.get("anio") or id_doc.get("ANIO") or 0)
            val_mes = int(id_doc.get("mes") or id_doc.get("MES") or 0)
        except:
            continue # Saltamos basura
            
        if val_anio > 2000: # Filtro básico de calidad
            datos.append({
                "anio": val_anio,
                "mes": val_mes,
                "total": d.get("total", 0)
            })

    return pd.DataFrame(datos)

def predecir_total_2026(col):
    df = preparar_mensual(col)
    
    if df.empty or len(df) < 6:
        return 0, [], [], [], []

    # Crear índice de tiempo (2018.0, 2018.08, etc.)
    df['time_index'] = df['anio'] + (df['mes'] - 1) / 12.0
    
    X = df[['time_index']]
    y = df['total']
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Proyectar 2026 (mes a mes)
    futuro = []
    etiquetas = []
    meses_txt = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
    
    for m in range(1, 13):
        futuro.append([2026 + (m-1)/12.0])
        etiquetas.append(meses_txt[m-1])
        
    predicciones = model.predict(futuro)
    predicciones = np.maximum(predicciones, 0)
    
    total_2026 = int(predicciones.sum())
    
    return total_2026, etiquetas, predicciones.tolist(), df['total'].tail(12).tolist(), df['anio'].tail(12).tolist()