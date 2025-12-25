import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def preparar_matriz_departamento(col):
    # Pipeline EXACTO según tu captura
    pipeline = [
        {
            "$group": {
                "_id": { 
                    "dpto": "$DPTO_HECHO_NEW", # Campo visto en imagen
                    "anio": "$ANIO"            # Campo visto en imagen
                },
                # Sumamos el volumen real de delitos
                "total": { "$sum": "$cantidad" } 
            }
        }
    ]
    
    resultados = list(col.aggregate(pipeline))
    
    datos = []
    for d in resultados:
        id_doc = d.get("_id", {})
        dpto = id_doc.get("dpto") or "DESCONOCIDO"
        
        if dpto != "DESCONOCIDO":
            datos.append({
                "departamento": dpto,
                "cantidad": d.get("total", 0)
            })
    return datos

def clusterizar_departamentos(col, n_clusters=3):
    datos = preparar_matriz_departamento(col)
    
    if not datos:
        return []
        
    df = pd.DataFrame(datos)
    
    # Agrupar histórico total por departamento
    matrix = df.groupby('departamento')['cantidad'].sum().reset_index()
    
    if matrix.empty:
        return []

    # Ajuste dinámico de clusters
    k_real = min(n_clusters, len(matrix))
    if k_real < 2:
        return [{"departamento": r['departamento'], "cluster": 0, "riesgo": "Datos insuficientes"} for _, r in matrix.iterrows()]

    # K-Means
    X = matrix[['cantidad']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=k_real, random_state=42, n_init=10)
    matrix['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Calcular Riesgo (Mayor cantidad = Riesgo Alto)
    cluster_avg = matrix.groupby('cluster')['cantidad'].mean().sort_values(ascending=False)
    riesgo_map = {}
    etiquetas = ["Alto", "Medio", "Bajo"]
    
    for i, cluster_id in enumerate(cluster_avg.index):
        if i < len(etiquetas):
            riesgo_map[cluster_id] = etiquetas[i]
        else:
            riesgo_map[cluster_id] = "Bajo"

    resultado_final = []
    for _, row in matrix.iterrows():
        c_id = int(row['cluster'])
        resultado_final.append({
            "departamento": row['departamento'],
            "cluster": c_id,
            "riesgo": riesgo_map.get(c_id, "Bajo")
        })
        
    # Ordenar: Alto -> Medio -> Bajo
    orden = {"Alto": 0, "Medio": 1, "Bajo": 2}
    resultado_final.sort(key=lambda x: orden.get(x["riesgo"], 3))
    
    return resultado_final