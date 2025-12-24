# ml_cluster.py
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

def preparar_matriz_departamento(col):
    # 1. Filtro
    match_filter = { "ANIO": { "$ne": None } }

    # 2. Pipeline
    pipeline = [
        { "$match": match_filter },
        {
            "$group": {
                # --- OJO AQUÍ ---
                "_id": { 
                    "dpto": "$DPTO_HECHO_NEW", # <--- Etiqueta "dpto" (minúscula)
                    "mod": "$P_MODALIDADES",   # <--- Etiqueta "mod" (minúscula)
                    "anio": "$ANIO"            # <--- Etiqueta "anio" (minúscula)
                },
                "total": { "$sum": 1 }
            }
        }
    ]

    # 3. Ejecutar
    resultados = list(col.aggregate(pipeline))

    # 4. Procesar
    datos = []
    for d in resultados:
        datos.append({
            "departamento": d["_id"]["dpto"], # Busca "dpto"
            "modalidad": d["_id"]["mod"],     # Busca "mod"
            "anio": d["_id"]["anio"],         # Busca "anio"
            "total": d["total"]
        })
    
    return datos

def clusterizar_departamentos(col, n_clusters=3):
    tabla = preparar_matriz_departamento(col)
    if tabla.empty:
        return []

    X = tabla.values
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    kmeans.fit(X)

    tabla["cluster"] = kmeans.labels_
    tabla.reset_index(inplace=True)

    # Convertimos a lista de dicts para mandarlo al template
    resultado = tabla.to_dict(orient="records")
    return resultado
