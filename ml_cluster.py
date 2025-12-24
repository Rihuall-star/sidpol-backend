import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def preparar_matriz_departamento(col):
    pipeline = [
        {
            "$group": {
                "_id": { 
                    "dpto": "$DPTO_HECHO_NEW", 
                    "mod": "$P_MODALIDADES" 
                },
                "total": { "$sum": 1 }
            }
        }
    ]
    
    resultados = list(col.aggregate(pipeline))
    datos = []
    for d in resultados:
        id_data = d.get("_id", {})
        datos.append({
            "departamento": id_data.get("dpto") or "OTRO",
            "modalidad": id_data.get("mod") or "OTRO",
            "cantidad": d.get("total") or 0
        })
    return datos

def clusterizar_departamentos(col, n_clusters=3):
    datos = preparar_matriz_departamento(col)
    if not datos:
        return []
        
    df = pd.DataFrame(datos)
    # Pivotar para tener modalidades como columnas y departamentos como filas
    matrix = df.pivot_table(index='departamento', columns='modalidad', values='cantidad', fill_value=0)
    
    # Escalar datos
    scaler = StandardScaler()
    matrix_scaled = scaler.fit_transform(matrix)
    
    # KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    matrix['cluster'] = kmeans.fit_predict(matrix_scaled)
    
    # Formatear para el frontend
    resultado_final = []
    for dpto, row in matrix.iterrows():
        resultado_final.append({
            "departamento": dpto,
            "cluster": int(row['cluster']),
            "riesgo": "Alto" if row['cluster'] == 0 else "Medio" if row['cluster'] == 1 else "Bajo"
        })
    return resultado_final