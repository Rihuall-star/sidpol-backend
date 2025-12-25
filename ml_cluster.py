import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def preparar_matriz_departamento(col):
    # Traemos todo sin filtrar por año para ver qué encuentra
    pipeline = [
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
        id_doc = d.get("_id", {})
        # Usamos valores por defecto si no encuentra columnas
        datos.append({
            "departamento": id_doc.get("dpto") or id_doc.get("DPTO_HECHO_NEW") or "DESCONOCIDO",
            "modalidad": id_doc.get("mod") or id_doc.get("P_MODALIDADES") or "GENERAL",
            "anio": id_doc.get("anio") or id_doc.get("ANIO") or 0,
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

    # --- CORRECCIÓN DEL ERROR 500 ---
    # Si tenemos menos filas (departamentos) que clusters solicitados,
    # ajustamos el número de clusters al número de filas.
    n_registros = len(matrix)
    k_real = min(n_clusters, n_registros)
    
    # Si solo hay 1 registro, no podemos escalar ni clusterizar normal, lo devolvemos directo
    if k_real < 2:
        resultado_final = []
        for dpto, row in matrix.iterrows():
            resultado_final.append({
                "departamento": dpto,
                "cluster": 0,
                "riesgo": "Datos insuficientes"
            })
        return resultado_final

    # Escalar y K-Means normal
    scaler = StandardScaler()
    matrix_scaled = scaler.fit_transform(matrix)
    
    kmeans = KMeans(n_clusters=k_real, random_state=42, n_init=10)
    matrix['cluster'] = kmeans.fit_predict(matrix_scaled)
    
    # Asignar riesgo
    resultado_final = []
    for dpto, row in matrix.iterrows():
        c_id = int(row['cluster'])
        # Lógica simple de riesgo para visualización
        riesgo = "Alto" if c_id == 0 else "Medio" if c_id == 1 else "Bajo"
        resultado_final.append({
            "departamento": dpto,
            "cluster": c_id,
            "riesgo": riesgo
        })
        
    return resultado_final