import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

def preparar_dataset_riesgo(col, modalidad_objetivo):
    """
    Prepara los datos para el modelo, sumando la columna 'cantidad'.
    """
    match_filter = {
        "$or": [
            {"P_MODALIDADES": modalidad_objetivo},
            {"modalidad": modalidad_objetivo}
        ]
    }

    # Pipeline: Agrupar por Trimestre y Sumar Cantidad Real
    pipeline = [
        { "$match": match_filter },
        {
            "$group": {
                "_id": { 
                    "anio": "$ANIO", 
                    "trimestre": "$trimestre",
                    "dpto": "$DPTO_HECHO_NEW" 
                },
                "total": { "$sum": "$cantidad" } 
            }
        }
    ]

    resultados = list(col.aggregate(pipeline))
    
    # Fallback minúsculas
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
        except: continue

    if not datos: return pd.DataFrame()

    df = pd.DataFrame(datos)
    df['periodo'] = df['anio'].astype(str) + "-" + df['trimestre']
    df = df.sort_values(by=['anio', 'trimestre_num'])
    return df

def entrenar_modelo_riesgo(col, modalidad_objetivo):
    """ Entrena el modelo Random Forest """
    df = preparar_dataset_riesgo(col, modalidad_objetivo)
    
    if df.empty: return None, pd.DataFrame(), None

    le_dpto = LabelEncoder()
    df['dpto_code'] = le_dpto.fit_transform(df['departamento'])
    
    X = df[['anio', 'trimestre_num', 'dpto_code']]
    y = df['total']
    
    modelo = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo.fit(X, y)
    
    return modelo, df, le_dpto

# --- ESTA ES LA FUNCIÓN QUE FALTABA O DABA ERROR ---
def predecir_valor_especifico(modelo, le_dpto, anio, trimestre_str, departamento):
    """
    Realiza la predicción puntual para el simulador.
    """
    try:
        anio = int(anio)
        trim_map = {"T1": 1, "T2": 2, "T3": 3, "T4": 4}
        trim_num = trim_map.get(trimestre_str, 1)
        departamento = str(departamento).upper().strip()
        
        # Verificar si el departamento es conocido por el modelo
        if departamento in le_dpto.classes_:
            dpto_code = le_dpto.transform([departamento])[0]
        else:
            return 0 # Departamento nuevo/desconocido

        # Crear DataFrame con nombres de columnas para evitar warnings
        X_pred = pd.DataFrame([[anio, trim_num, dpto_code]], columns=['anio', 'trimestre_num', 'dpto_code'])
        
        prediccion = modelo.predict(X_pred)[0]
        return int(prediccion)
    except Exception as e:
        print(f"Error en predicción interna: {e}")
        return 0