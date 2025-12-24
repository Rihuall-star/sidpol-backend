# ml_cluster.py
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

def preparar_matriz_departamento(col):
    # --- PASO 1: DEFINIR LA VARIABLE PRIMERO ---
    # Filtramos para asegurarnos de que existe el año (ANIO)
    match_filter = { "ANIO": { "$ne": None } }

    # --- PASO 2: CREAR EL PIPELINE ---
    pipeline = [
        {
            "$match": match_filter # <--- Python ya sabe qué es esto
        },
        {
            "$group": {
                # Mapeo completo: Cloud -> Python
                "_id": { 
                    "dpto": "$DPTO_HECHO_NEW", 
                    "mod": "$P_MODALIDADES",
                    "anio": "$ANIO" 
                },
                "total": { "$sum": 1 }
            }
        }
    ]

    # --- PASO 3: EJECUTAR ---
    resultados = list(col.aggregate(pipeline))

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
