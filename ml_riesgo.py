import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

def preparar_dataset_riesgo(col, modalidad_objetivo):
    """
    Prepara datos históricos agrupados por Trimestre, usando la columna 'cantidad'.
    """
    # 1. Filtro (Modalidad)
    match_filter = {
        "$or": [
            {"P_MODALIDADES": modalidad_objetivo},
            {"modalidad": modalidad_objetivo}
        ]
    }

    # 2. Agregación (Suma de 'cantidad')
    pipeline = [
        { "$match": match_filter },
        {
            "$group": {
                "_id": { 
                    "anio": "$ANIO", 
                    "trimestre": "$trimestre",
                    "dpto": "$DPTO_HECHO_NEW" 
                },
                "total": { "$sum": "$cantidad" } # Suma volumen real
            }
        }
    ]

    resultados = list(col.aggregate(pipeline))
    
    # Plan B (Minúsculas)
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
            val_trim = id_doc.get("trimestre")
            # Normalizamos el nombre del departamento (Mayúsculas y sin espacios extra)
            val_dpto = str(id_doc.get("dpto") or "DESCONOCIDO").upper().strip()
            val_total = d.get("total", 0)
            
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
    # Etiqueta para gráfico
    df['periodo'] = df['anio'].astype(str) + "-" + df['trimestre']
    
    return df

def entrenar_modelo_riesgo(col, modalidad_objetivo):
    df = preparar_dataset_riesgo(col, modalidad_objetivo)
    
    if df.empty:
        return None, pd.DataFrame(), None

    # Codificar Departamentos
    le_dpto = LabelEncoder()
    df['dpto_code'] = le_dpto.fit_transform(df['departamento'])
    
    # Entrenar Random Forest
    X = df[['anio', 'trimestre_num', 'dpto_code']]
    y = df['total']
    
    modelo = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo.fit(X, y)
    
    return modelo, df, le_dpto

def predecir_valor_especifico(modelo, le_dpto, anio, trimestre_str, departamento):
    try:
        # Normalizar inputs
        anio = int(anio)
        departamento = str(departamento).upper().strip()
        
        trim_map = {"T1": 1, "T2": 2, "T3": 3, "T4": 4}
        trim_num = trim_map.get(trimestre_str, 1)
        
        # Validar si el departamento existe en el entrenamiento
        if departamento in le_dpto.classes_:
            dpto_code = le_dpto.transform([departamento])[0]
        else:
            # Si es un dpto nuevo, no podemos predecir con precisión -> devolvemos 0
            print(f"Departamento '{departamento}' no encontrado en datos históricos.")
            return 0 

        # Crear DataFrame con nombres de columnas (importante para evitar warnings)
        X_pred = pd.DataFrame([[anio, trim_num, dpto_code]], 
                              columns=['anio', 'trimestre_num', 'dpto_code'])
        
        prediccion = modelo.predict(X_pred)[0]
        
        return int(prediccion)
    except Exception as e:
        print(f"Error en predicción puntual: {e}")
        return 0