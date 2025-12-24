# ml_cluster.py
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

def preparar_matriz_departamento(col):
    """
    Devuelve un DataFrame:
    índice = departamento
    columnas = modalidades
    valores = cantidad total (2018–2025)
    """
    pipeline = [
        {"$group": {
            "_id": {"dpto": "$DPTO_HECHO_NEW", "mod": "$P_MODALIDADES"},
            "total": {"$sum": "$cantidad"}
        }}
    ]
    datos = list(col.aggregate(pipeline))
    if not datos:
        return pd.DataFrame()

    filas = []
    for d in datos:
        filas.append({
            "departamento": d["_id"]["dpto"],
            "modalidad": d["_id"]["mod"],
            "total": d["total"]
        })
    df = pd.DataFrame(filas)

    # Pivot: departamento x modalidad
    tabla = df.pivot_table(
        index="departamento",
        columns="modalidad",
        values="total",
        aggfunc="sum",
        fill_value=0
    )

    # Normalizar por fila (para que sean proporciones)
    tabla = tabla.div(tabla.sum(axis=1), axis=0)
    return tabla

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
