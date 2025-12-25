import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

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
    # Intentamos primero con mayúsculas (ANIO, MES) y sumamos 'cantidad' (minúscula)
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
    
    # Fallback: Si no trajo nada, probamos agrupar por minúsculas (anio, mes)
    if not resultados:
        pipeline[1]["$group"]["_id"] = { "anio": "$anio", "mes": "$mes" }
        pipeline[1]["$group"]["total"] = { "$sum": "$cantidad" }
        resultados = list(col.aggregate(pipeline))

    # 3. Limpieza y estructuración en lista
    datos = []
    for d in resultados:
        id_doc = d.get("_id", {})
        try:
            val_anio = int(id_doc.get("anio") or 0)
            val_mes = int(id_doc.get("mes") or 0)
            
            # Filtramos datos incoherentes (ej: año 1900 o mes 13)
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
    USADO POR: Ruta /prediccion-2026 (El Gráfico de Tendencias)
    Retorna 5 valores: total, etiquetas, predicciones, historico_valores, historico_anios
    """
    df = preparar_mensual(col)
    
    # Si hay muy pocos datos, devolvemos vacíos para no romper la web
    if df.empty or len(df) < 12:
        return 0, [], [], [], []

    # Crear índice temporal continuo para la regresión
    df['time_index'] = df['anio'] + (df['mes'] - 1) / 12.0
    
    X = df[['time_index']]
    y = df['total']
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Generar proyección para 2026 (12 meses)
    meses_txt = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
    futuro = np.array([2026 + (m-1)/12.0 for m in range(1, 13)]).reshape(-1, 1)
    
    # Creamos DataFrame para evitar warnings de sklearn
    X_futuro = pd.DataFrame(futuro, columns=['time_index'])
    
    predicciones = model.predict(X_futuro)
    predicciones = np.maximum(predicciones, 0) # Evitar negativos
    
    total_2026 = int(predicciones.sum())
    
    # Retornamos los datos necesarios para Chart.js
    return total_2026, meses_txt, predicciones.tolist(), df['total'].tail(12).tolist(), df['anio'].tail(12).tolist()

def obtener_contexto_ia(col):
    """
    USADO POR: Ruta /agente-estrategico (Gemini AI)
    Retorna: total_2026, texto_resumen_historico
    """
    # Reutilizamos la lógica de predicción para consistencia
    # OJO: Aquí llamamos a predecir_total_2026 que está definida arriba
    total_2026, _, _, historico_val, historico_anio = predecir_total_2026(col)
    
    if total_2026 == 0:
        return 0, "Datos insuficientes"
    
    # Generamos un texto resumen de los últimos 3 meses reales
    texto_contexto = ""
    limit = min(3, len(historico_val))
    # Usamos slicing negativo para obtener los últimos
    for i in range(1, limit + 1):
        texto_contexto += f"[{historico_anio[-i]}: {int(historico_val[-i])} casos] "
            
    return total_2026, texto_contexto