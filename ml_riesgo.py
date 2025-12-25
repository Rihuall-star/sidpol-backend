import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

def preparar_dataset_riesgo(col, modalidad_objetivo):
    """
    Prepara los datos usando los campos 'trimestre' y 'cantidad' existentes en tu DB.
    """
    # 1. Filtro de Modalidad
    match_filter = {
        "$or": [
            {"P_MODALIDADES": modalidad_objetivo},
            {"modalidad": modalidad_objetivo}
        ]
    }

    # 2. Pipeline Optimizado (Usamos tus campos reales)
    pipeline = [
        { "$match": match_filter },
        {
            "$group": {
                "_id": { 
                    "anio": "$ANIO",             # Campo de tu imagen
                    "trimestre": "$trimestre",   # Campo de tu imagen (ej: "T1")
                    "dpto": "$DPTO_HECHO_NEW"    # Campo de tu imagen
                },
                "total": { "$sum": "$cantidad" } # ¡SUMAMOS LA CANTIDAD REAL!
            }
        }
    ]

    resultados = list(col.aggregate(pipeline))
    
    # Si falla, intentamos nombres en minúscula por seguridad
    if not resultados:
        pipeline[1]["$group"]["_id"] = { "anio": "$anio", "trimestre": "$trimestre", "dpto": "$departamento" }
        pipeline[1]["$group"]["total"] = { "$sum": "$cantidad" }
        resultados = list(col.aggregate(pipeline))

    datos = []
    trim_map = {"T1": 1, "T2": 2, "T3": 3, "T4": 4}

    for d in resultados:
        id_doc = d.get("_id", {})
        try:
            val_anio = int(id_doc.get("anio") or 0)
            val_trim = id_doc.get("trimestre") # Esperamos "T1", "T2"...
            val_dpto = id_doc.get("dpto") or "DESCONOCIDO"
            val_total = d.get("total", 0)
            
            # Solo procesamos si tenemos datos válidos
            if val_anio > 2000 and val_trim in trim_map:
                datos.append({
                    "anio": val_anio,
                    "trimestre": val_trim,
                    "trimestre_num": trim_map[val_trim],
                    "departamento": val_dpto,
                    "total": val_total
                })
        except:
            continue

    if not datos:
        return pd.DataFrame()

    df = pd.DataFrame(datos)
    
    # Crear etiqueta combinada para el gráfico (Ej: "2018-T1")
    df['periodo'] = df['anio'].astype(str) + "-" + df['trimestre']
    
    # Ordenar por tiempo
    df = df.sort_values(by=['anio', 'trimestre_num'])
    
    return df

def entrenar_modelo_riesgo(col, modalidad_objetivo):
    df = preparar_dataset_riesgo(col, modalidad_objetivo)
    
    if df.empty:
        return None, pd.DataFrame(), None

    # Codificar departamentos
    le_dpto = LabelEncoder()
    df['dpto_code'] = le_dpto.fit_transform(df['departamento'])
    
    # Variables X e y
    X = df[['anio', 'trimestre_num', 'dpto_code']]
    y = df['total']
    
    # Modelo
    modelo = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo.fit(X, y)
    
    return modelo, df, le_dpto

def predecir_valor_especifico(modelo, le_dpto, anio, trimestre_str, departamento):
    try:
        trim_map = {"T1": 1, "T2": 2, "T3": 3, "T4": 4}
        trim_num = trim_map.get(trimestre_str, 1)
        
        if departamento in le_dpto.classes_:
            dpto_code = le_dpto.transform([departamento])[0]
        else:
            return 0 

        X_pred = pd.DataFrame([[anio, trim_num, dpto_code]], columns=['anio', 'trimestre_num', 'dpto_code'])
        prediccion = modelo.predict(X_pred)[0]
        
        return int(prediccion)
    except Exception:
        return 0