import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def obtener_contexto_ia(col):
    """
    Extrae la proyección numérica y un resumen del histórico para pasárselo a Gemini.
    Usa la lógica CORRECTA de sumar el campo 'cantidad'.
    """
    # 1. Pipeline de Agregación (Suma de volúmenes reales)
    pipeline = [
        {
            "$group": {
                "_id": { "anio": "$ANIO", "mes": "$MES" },
                "total": { "$sum": "$cantidad" } # Sumamos el campo 'cantidad' (minúscula)
            }
        },
        { "$sort": { "_id.anio": 1, "_id.mes": 1 } }
    ]
    
    resultados = list(col.aggregate(pipeline))
    
    # 2. Limpieza de datos
    datos = []
    for d in resultados:
        id_doc = d.get("_id", {})
        try:
            val_anio = int(id_doc.get("anio") or 0)
            val_mes = int(id_doc.get("mes") or 0)
            # Validación básica
            if val_anio > 2000:
                datos.append({
                    "anio": val_anio,
                    "mes": val_mes,
                    "total": d.get("total", 0)
                })
        except:
            continue
            
    df = pd.DataFrame(datos)
    
    if df.empty or len(df) < 12:
        return 0, "Datos insuficientes"

    # 3. Proyección Lineal Rápida (Para tener el número del 2026)
    df['time_index'] = df['anio'] + (df['mes'] - 1) / 12.0
    X = df[['time_index']]
    y = df['total']
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Predecir todo 2026
    futuro = np.array([2026 + (m-1)/12.0 for m in range(1, 13)]).reshape(-1, 1)
    X_futuro = pd.DataFrame(futuro, columns=['time_index'])
    
    predicciones = model.predict(X_futuro)
    total_2026 = int(np.maximum(predicciones, 0).sum())
    
    # 4. Generar Contexto de Texto (Últimos 3 periodos reales para que la IA vea la tendencia)
    ultimos = df.tail(3)
    texto_historico = ""
    for _, row in ultimos.iterrows():
        texto_historico += f"[{row['anio']}-{row['mes']}: {int(row['total'])} casos] "
        
    return total_2026, texto_historico