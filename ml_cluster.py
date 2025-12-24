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
        # --- CÓDIGO BLINDADO ---
        id_doc = d.get("_id", {})
        
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