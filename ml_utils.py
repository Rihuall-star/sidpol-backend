import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def preparar_mensual(col, modalidad=None):
    """
    Función base para extraer datos reales sumando la columna 'cantidad'.
    """
    # 1. Filtro
    match_filter = {}
    if modalidad:
        match_filter["$or"] = [
            {"P_MODALIDADES": modalidad}, 
            {"modalidad": modalidad}
        ]

    # 2. Pipeline (Suma de volúmenes reales: 'cantidad')
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
    
    # 3. Limpieza y validación
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
    """
    USADO POR: Ruta /prediccion-2026 (El Gráfico)
    Retorna: total, etiquetas, valores_predichos, historico_valores, historico_anios
    """
    df = preparar_mensual(col)
    
    if df.empty or len(df) < 12:
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
    
    # Retorna 5 valores (lo que espera app.py)
    return total_2026, etiquetas, predicciones.tolist(), df['total'].tail(12).tolist(), df['anio'].tail(12).tolist()

def obtener_contexto_ia(col):
    """
    USADO POR: Ruta /agente-estrategico (Gemini AI)
    Retorna: total_2026, texto_historico_resumido
    """
    # Reutilizamos la lógica base para no repetir código
    df = preparar_mensual(col)
    
    if df.empty or len(df) < 12:
        return 0, "Datos insuficientes"

    # Hacemos la predicción rápida para obtener el número total
    df['time_index'] = df['anio'] + (df['mes'] - 1) / 12.0
    model = LinearRegression()
    model.fit(df[['time_index']], df['total'])
    
    futuro = np.array([2026 + (m-1)/12.0 for m in range(1, 13)]).reshape(-1, 1)
    X_futuro = pd.DataFrame(futuro, columns=['time_index'])
    pred = model.predict(X_futuro)
    total_2026 = int(np.maximum(pred, 0).sum())
    
    # Generamos el texto de contexto (últimos 3 meses reales)
    ultimos = df.tail(3)
    texto_historico = ""
    for _, row in ultimos.iterrows():
        texto_historico += f"[{int(row['anio'])}-{int(row['mes'])}: {int(row['total'])} delitos] "
        
    return total_2026, texto_historico