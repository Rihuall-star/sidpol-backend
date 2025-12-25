import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def preparar_matriz_departamento(col):
    """
    Cuenta el volumen real de denuncias por departamento.
    """
    pipeline = [
        {
            "$group": {
                "_id": { 
                    "dpto": "$DPTO_HECHO_NEW", # Campo estándar SIDPOL
                    "anio": "$ANIO"
                },
                "total": { "$sum": 1 } # Conteo real
            }
        }
    ]
    
    resultados = list(col.aggregate(pipeline))
    
    # Plan B (Minúsculas)
    if not resultados:
        pipeline[0]["$group"]["_id"] = { "dpto": "$departamento", "anio": "$anio" }
        resultados = list(col.aggregate(pipeline))

    datos = []
    for d in resultados:
        id_doc = d.get("_id", {})
        # Normalizamos nombres de departamentos
        dpto = id_doc.get("dpto") or id_doc.get("DPTO_HECHO_NEW") or "DESCONOCIDO"
        
        if dpto and dpto != "DESCONOCIDO":
            datos.append({
                "departamento": dpto,
                "cantidad": d.get("total", 0)
            })
    return datos

def clusterizar_departamentos(col, n_clusters=3):
    """
    Agrupa departamentos por nivel de riesgo (Volumen de denuncias).
    """
    datos = preparar_matriz_departamento(col)
    
    if not datos:
        return []
        
    df = pd.DataFrame(datos)
    
    # Sumamos todo el histórico para ver la carga total por región
    matrix = df.groupby('departamento')['cantidad'].sum().reset_index()
    
    if matrix.empty:
        return []

    # Ajuste de seguridad por si hay pocos departamentos en la DB
    n_registros = len(matrix)
    k_real = min(n_clusters, n_registros)
    
    if k_real < 2:
        return [{"departamento": r['departamento'], "cluster": 0, "riesgo": "Datos insuficientes"} for _, r in matrix.iterrows()]

    # K-Means basado en la cantidad total
    X = matrix[['cantidad']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=k_real, random_state=42, n_init=10)
    matrix['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Determinar etiquetas de riesgo dinámicamente
    # Calculamos el promedio de delitos por cluster y ordenamos
    cluster_avg = matrix.groupby('cluster')['cantidad'].mean().sort_values(ascending=False)
    riesgo_map = {}
    etiquetas_posibles = ["Alto", "Medio", "Bajo"]
    
    for i, cluster_id in enumerate(cluster_avg.index):
        if i < len(etiquetas_posibles):
            riesgo_map[cluster_id] = etiquetas_posibles[i]
        else:
            riesgo_map[cluster_id] = "Bajo"

    # Construir resultado final
    resultado_final = []
    for _, row in matrix.iterrows():
        c_id = int(row['cluster'])
        resultado_final.append({
            "departamento": row['departamento'],
            "cluster": c_id,
            "riesgo": riesgo_map.get(c_id, "Bajo")
        })
        
    # Ordenar visualmente: Primero los de Riesgo Alto
    orden_visual = {"Alto": 0, "Medio": 1, "Bajo": 2}
    resultado_final.sort(key=lambda x: orden_visual.get(x["riesgo"], 3))
    
    return resultado_final