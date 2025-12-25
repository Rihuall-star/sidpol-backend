import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

def preparar_dataset_riesgo(col, modalidad_objetivo):
    """
    Prepara los datos para el modelo de riesgo, agrupando por Trimestre.
    """
    # 1. Filtro: Buscamos la modalidad (ignorando mayúsculas/minúsculas)
    match_filter = {
        "$or": [
            {"P_MODALIDADES": modalidad_objetivo},
            {"modalidad": modalidad_objetivo}
        ]
    }

    # 2. Pipeline: Agrupamos por Año, Mes y Departamento
    pipeline = [
        { "$match": match_filter },
        {
            "$group": {
                "_id": { 
                    "anio": "$ANIO", 
                    "mes": "$MES",
                    "dpto": "$DPTO_HECHO_NEW" 
                },
                "total": { "$sum": 1 } # Contamos denuncias reales
            }
        }
    ]

    resultados = list(col.aggregate(pipeline))
    
    # Si falla con mayúsculas, probamos minúsculas (Plan B)
    if not resultados:
        pipeline[1]["$group"]["_id"] = { "anio": "$anio", "mes": "$mes", "dpto": "$departamento" }
        resultados = list(col.aggregate(pipeline))

    datos = []
    for d in resultados:
        id_doc = d.get("_id", {})
        try:
            # Extracción segura de datos (Blindaje)
            val_anio = int(id_doc.get("anio") or id_doc.get("ANIO") or 0)
            val_mes = int(id_doc.get("mes") or id_doc.get("MES") or 0)
            val_dpto = id_doc.get("dpto") or id_doc.get("DPTO_HECHO_NEW") or "DESCONOCIDO"
            
            if val_anio > 2000 and 1 <= val_mes <= 12 and val_dpto != "DESCONOCIDO":
                # Calcular Trimestre (1-3 -> T1, 4-6 -> T2, etc.)
                trim_num = (val_mes - 1) // 3 + 1
                trim_str = f"T{trim_num}"
                
                datos.append({
                    "anio": val_anio,
                    "trimestre": trim_str,
                    "trimestre_num": trim_num,
                    "departamento": val_dpto,
                    "total": d.get("total", 0)
                })
        except:
            continue

    if not datos:
        return pd.DataFrame()

    df = pd.DataFrame(datos)
    
    # 3. Agrupar ahora por TRIMESTRE (sumando los meses del trimestre)
    df_agrupado = df.groupby(['anio', 'trimestre', 'trimestre_num', 'departamento'])['total'].sum().reset_index()
    
    # Crear etiqueta combinada para el gráfico (Ej: "2024-T1")
    df_agrupado['periodo'] = df_agrupado['anio'].astype(str) + "-" + df_agrupado['trimestre']
    
    return df_agrupado

def entrenar_modelo_riesgo(col, modalidad_objetivo):
    """
    Entrena un Random Forest para predecir denuncias por trimestre y departamento.
    """
    df = preparar_dataset_riesgo(col, modalidad_objetivo)
    
    if df.empty:
        return None, pd.DataFrame(), None

    # Codificar el departamento a números para que la IA entienda
    le_dpto = LabelEncoder()
    df['dpto_code'] = le_dpto.fit_transform(df['departamento'])
    
    # Variables predictoras (X): Año, Trimestre (número), Departamento (código)
    X = df[['anio', 'trimestre_num', 'dpto_code']]
    y = df['total']
    
    # Modelo Random Forest (como en tu captura)
    modelo = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo.fit(X, y)
    
    return modelo, df, le_dpto

def predecir_valor_especifico(modelo, le_dpto, anio, trimestre_str, departamento):
    """
    Usa el modelo entrenado para predecir un caso puntual.
    """
    try:
        # Convertir T1 -> 1
        trim_map = {"T1": 1, "T2": 2, "T3": 3, "T4": 4}
        trim_num = trim_map.get(trimestre_str, 1)
        
        # Convertir nombre de departamento a código
        # Si el departamento no existe en el entrenamiento, usamos un código promedio o lanzamos error controlado
        if departamento in le_dpto.classes_:
            dpto_code = le_dpto.transform([departamento])[0]
        else:
            return 0 # Departamento desconocido para el modelo

        X_pred = pd.DataFrame([[anio, trim_num, dpto_code]], columns=['anio', 'trimestre_num', 'dpto_code'])
        prediccion = modelo.predict(X_pred)[0]
        
        return int(prediccion)
    except Exception as e:
        print(f"Error en predicción puntual: {e}")
        return 0