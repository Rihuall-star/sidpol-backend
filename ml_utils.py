import pandas as pd # Importación al inicio para evitar errores

def preparar_mensual(col, modalidad=None):
    # 1. Filtro
    match_filter = {}
    if modalidad:
        match_filter["P_MODALIDADES"] = modalidad

    # 2. Pipeline
    pipeline = [
        { "$match": match_filter },
        {
            "$group": {
                "_id": { "anio": "$ANIO", "mes": "$MES" },
                "total": { "$sum": 1 }
            }
        },
        { "$sort": { "_id.anio": 1, "_id.mes": 1 } }
    ]

    resultados = list(col.aggregate(pipeline))
    
    datos_procesados = []
    for d in resultados:
        # --- CÓDIGO BLINDADO ---
        # Usamos .get() para que si no encuentra la llave, no explote
        id_doc = d.get("_id", {})
        
        # Buscamos 'anio' O 'ANIO'. Si no hay nada, ponemos 0.
        val_anio = id_doc.get("anio") or id_doc.get("ANIO") or 0
        
        # Buscamos 'mes' O 'MES'. Si no hay nada, ponemos 0.
        val_mes = id_doc.get("mes") or id_doc.get("MES") or 0
        
        datos_procesados.append({
            "anio": val_anio,
            "mes": val_mes,
            "total": d.get("total", 0)
        })

    df = pd.DataFrame(datos_procesados)
    return df