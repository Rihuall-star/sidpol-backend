# ml_cluster.py
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

def preparar_matriz_departamento(col):
    # 1. Definimos el filtro (match_filter)
    # Filtramos para asegurarnos de que el año exista
    match_filter = { "ANIO": { "$ne": None } }

    # 2. Creamos el Pipeline
    pipeline = [
        {
            "$match": match_filter
        },
        {
            "$group": {
                # Mapeo completo para Agentes
                "_id": { 
                    "dpto": "$DPTO_HECHO_NEW", 
                    "mod": "$P_MODALIDADES",
                    "anio": "$ANIO" 
                },
                "total": { "$sum": 1 }
            }
        }
    ]

    # 3. Ejecutamos consulta
    resultados = list(col.aggregate(pipeline))

    # 4. Procesamos resultados
    datos = []
    for d in resultados:
        datos.append({
            "departamento": d["_id"]["dpto"],
            "modalidad": d["_id"]["mod"],
            "anio": d["_id"]["anio"],
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
