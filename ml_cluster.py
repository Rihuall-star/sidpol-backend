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
                "_id": { 
                    "dpto": "$DPTO_HECHO_NEW", 
                    "mod": "$P_MODALIDADES",
                    "anio": "$ANIO" 
                },
                "total": { "$sum": 1 }
            }
        }
    ]

    resultados = list(col.aggregate(pipeline))

    datos = []
    for d in resultados:
        # --- BLINDAJE ANTI-ERROR ---
        id_doc = d.get("_id", {})
        
        # Intentamos minúsculas, luego mayúsculas, luego un texto por defecto
        val_dpto = id_doc.get("dpto") or id_doc.get("DPTO_HECHO_NEW") or "DESCONOCIDO"
        val_mod = id_doc.get("mod") or id_doc.get("P_MODALIDADES") or "DESCONOCIDO"
        val_anio = id_doc.get("anio") or id_doc.get("ANIO") or 0

        datos.append({
            "departamento": val_dpto,
            "modalidad": val_mod,
            "anio": val_anio,
            "total": d.get("total", 0)
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
