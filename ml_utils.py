import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def preparar_mensual(col, modalidad=None):
    # 1. Filtro
    match_filter = {}
    if modalidad:
        match_filter["P_MODALIDADES"] = modalidad

    # 2. Pipeline EXACTO según tu captura de pantalla
    pipeline = [
        { "$match": match_filter },
        {
            "$group": {
                "_id": { 
                    "anio": "$ANIO",  # Campo visto en imagen: ANIO
                    "mes": "$MES"     # Campo visto en imagen: MES
                }, 
                # --- CORRECCIÓN FINAL ---
                # Sumamos el campo 'cantidad' (minúscula) que vimos en la foto
                "total": { "$sum": "$cantidad" } 
            }
        },
        { "$sort": { "_id.anio": 1, "_id.mes": 1 } }
    ]

    resultados = list(col.aggregate(pipeline))
    
    datos = []
    for d in resultados:
        id_doc = d.get("_id", {})
        try:
            val_anio = int(id_doc.get("anio") or 0)
            val_mes = int(id_doc.get("mes") or 0)
            
            if val_anio > 2000 and 1 <= val_mes <= 12:
                datos.append({
                    "anio": val_anio,
                    "mes": val_mes,
                    "total": d.get("total", 0)
                })
        except:
            continue

    return pd.DataFrame(datos)

def predecir_total_2026(col):
    df = preparar_mensual(col)
    
    # Validación robusta
    if df.empty or df['total'].sum() == 0:
        return 0, [], [], [], []

    # Crear índice temporal
    df['time_index'] = df['anio'] + (df['mes'] - 1) / 12.0
    
    X = df[['time_index']]
    y = df['total']
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Proyectar 2026
    meses_txt = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
    futuro_indices = []
    etiquetas = []
    
    for m in range(1, 13):
        futuro_indices.append(2026 + (m-1)/12.0)
        etiquetas.append(meses_txt[m-1])
        
    X_futuro = pd.DataFrame(futuro_indices, columns=['time_index'])
    
    predicciones = model.predict(X_futuro)
    predicciones = np.maximum(predicciones, 0)
    
    total_2026 = int(predicciones.sum())
    
    return total_2026, etiquetas, predicciones.tolist(), df['total'].tail(12).tolist(), df['anio'].tail(12).tolist()