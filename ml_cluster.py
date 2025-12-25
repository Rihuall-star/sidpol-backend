import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def preparar_matriz_departamento(col):
    pipeline = [
        {
            "$group": {
                "_id": { 
                    "dpto": "$departamento",  # Nombre real (minúscula)
                    "mod": "$modalidad",      # Nombre real (minúscula)
                    "anio": "$anio"
                },
                "total": { "$sum": "$prediccion" } # Sumamos el valor 'prediccion'
            }
        }
    ]
    
    resultados = list(col.aggregate(pipeline))
    datos = []
    
    for d in resultados:
        id_doc = d.get("_id", {})
        datos.append({
            "departamento": id_doc.get("dpto") or "DESCONOCIDO",
            "modalidad": id_doc.get("mod") or "GENERAL",
            "anio": id_doc.get("anio") or 0,
            "cantidad": d.get("total", 0)
        })
    return datos

def clusterizar_departamentos(col, n_clusters=3):
    datos = preparar_matriz_departamento(col)
    
    if not datos:
        return []
        
    df = pd.DataFrame(datos)
    
    # Agrupar por departamento
    matrix = df.pivot_table(index='departamento', columns='modalidad', values='cantidad', aggfunc='sum', fill_value=0)
    
    if matrix.empty:
        return []

    # Ajuste para evitar errores si hay pocos departamentos
    n_registros = len(matrix)
    k_real = min(n_clusters, n_registros)
    
    if k_real < 2:
        # Si solo hay 1 departamento, devolvemos directo sin IA
        resultado = []
        for dpto, row in matrix.iterrows():
            resultado.append({
                "departamento": dpto,
                "cluster": 0,
                "riesgo": "Datos insuficientes"
            })
        return resultado

    # IA: K-Means
    scaler = StandardScaler()
    matrix_scaled = scaler.fit_transform(matrix)
    
    kmeans = KMeans(n_clusters=k_real, random_state=42, n_init=10)
    matrix['cluster'] = kmeans.fit_predict(matrix_scaled)
    
    # Determinar Riesgo
    cluster_risk = matrix.groupby('cluster').sum().sum(axis=1).sort_values(ascending=False)
    risk_labels = {c_id: label for c_id, label in zip(cluster_risk.index, ["Alto", "Medio", "Bajo"])}

    resultado_final = []
    for dpto, row in matrix.iterrows():
        c_id = int(row['cluster'])
        resultado_final.append({
            "departamento": dpto,
            "cluster": c_id,
            "riesgo": risk_labels.get(c_id, "Bajo")
        })
    
    # Ordenar por riesgo (Alto primero)
    orden = {"Alto": 0, "Medio": 1, "Bajo": 2}
    resultado_final.sort(key=lambda x: orden.get(x["riesgo"], 3))
        
    return resultado_final