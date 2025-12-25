import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def preparar_mensual(col, modalidad=None):
    """
    Agrupa las denuncias por Año y Mes contando cada documento ($sum: 1).
    """
    # 1. Filtro Opcional
    match_filter = {}
    if modalidad:
        # Busca en los campos habituales de SIDPOL
        match_filter["$or"] = [
            {"P_MODALIDADES": modalidad}, 
            {"modalidad": modalidad}
        ]

    # 2. Pipeline de Agregación
    # Intentamos agrupar por ANIO y MES (Mayúsculas es el estándar en tu DB grande)
    pipeline = [
        { "$match": match_filter },
        {
            "$group": {
                "_id": { 
                    "anio": "$ANIO", 
                    "mes": "$MES"
                }, 
                "total": { "$sum": 1 } # CONTEO REAL: 1 documento = 1 denuncia
            }
        },
        { "$sort": { "_id.anio": 1, "_id.mes": 1 } }
    ]

    resultados = list(col.aggregate(pipeline))
    
    # Plan B: Si no trajo nada, probamos minúsculas
    if not resultados:
        pipeline[1]["$group"]["_id"] = { "anio": "$anio", "mes": "$mes" }
        resultados = list(col.aggregate(pipeline))

    datos = []
    for d in resultados:
        id_doc = d.get("_id", {})
        try:
            # Limpieza y validación de datos
            val_anio = int(id_doc.get("anio") or id_doc.get("ANIO") or 0)
            val_mes = int(id_doc.get("mes") or id_doc.get("MES") or 0)
            
            # Filtramos años incoherentes (ej: año 1900 o nulos)
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
    """
    Genera la proyección para 2026 usando Regresión Lineal.
    """
    df = preparar_mensual(col)
    
    # Necesitamos al menos unos meses de historia para predecir
    if df.empty or len(df) < 6:
        return 0, [], [], [], []

    # Crear índice de tiempo continuo (Ej: 2024.0 para Enero, 2024.08 para Feb)
    # Esto ayuda a la IA a entender la continuidad del tiempo
    df['time_index'] = df['anio'] + (df['mes'] - 1) / 12.0
    
    X = df[['time_index']]
    y = df['total']
    
    # Entrenar el Modelo
    model = LinearRegression()
    model.fit(X, y)
    
    # Generar datos futuros para 2026 (12 meses)
    meses_txt = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
    futuro_indices = []
    etiquetas = []
    
    for m in range(1, 13):
        futuro_indices.append(2026 + (m-1)/12.0)
        etiquetas.append(meses_txt[m-1])
        
    # CORRECCIÓN DE WARNING: Creamos un DataFrame con el mismo nombre de columna
    X_futuro = pd.DataFrame(futuro_indices, columns=['time_index'])
    
    # Predecir
    predicciones = model.predict(X_futuro)
    predicciones = np.maximum(predicciones, 0) # Evitar números negativos
    
    total_2026 = int(predicciones.sum())
    
    # Retornamos todo lo necesario para los gráficos
    return total_2026, etiquetas, predicciones.tolist(), df['total'].tail(12).tolist(), df['anio'].tail(12).tolist()