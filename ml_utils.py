import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from pymongo import MongoClient
from dotenv import load_dotenv

# ==========================================
# 1. CONFIGURACIÓN DE BASE DE DATOS (CRUCIAL)
# ==========================================
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

try:
    client = MongoClient(MONGO_URI)
    # Asegúrate que este sea el nombre real de tu BD en Atlas
    db = client['denuncias_db'] 
    print("✅ Conexión a MongoDB exitosa en ml_utils")
except Exception as e:
    print(f"❌ Error conectando a MongoDB: {e}")
    db = None

# ==========================================
# 2. FUNCIONES DE PREDICCIÓN 2026 (TU CÓDIGO)
# ==========================================

def preparar_mensual(col, modalidad=None):
    """
    Función base: Extrae datos reales sumando la columna 'cantidad'.
    Soporta nombres de campos mixtos (ANIO/anio, MES/mes) por seguridad.
    """
    # 1. Filtro de modalidad (si se especifica)
    match_filter = {}
    if modalidad:
        match_filter["$or"] = [
            {"P_MODALIDADES": modalidad}, 
            {"modalidad": modalidad}
        ]

    # 2. Pipeline de Agregación
    pipeline = [
        { "$match": match_filter },
        {
            "$group": {
                "_id": { "anio": "$ANIO", "mes": "$MES" },
                "total": { "$sum": "$cantidad" } 
            }
        },
        { "$sort": { "_id.anio": 1, "_id.mes": 1 } }
    ]

    resultados = list(col.aggregate(pipeline))
    
    # Fallback: Si no trajo nada, probamos agrupar por minúsculas
    if not resultados:
        pipeline[1]["$group"]["_id"] = { "anio": "$anio", "mes": "$mes" }
        pipeline[1]["$group"]["total"] = { "$sum": "$cantidad" }
        resultados = list(col.aggregate(pipeline))

    # 3. Limpieza y estructuración
    datos = []
    for d in resultados:
        id_doc = d.get("_id", {})
        try:
            val_anio = int(id_doc.get("anio") or 0)
            val_mes = int(id_doc.get("mes") or 0)
            
            # Filtramos datos incoherentes
            if val_anio > 2000 and 1 <= val_mes <= 12:
                datos.append({
                    "anio": val_anio,
                    "mes": val_mes,
                    "total": d.get("total", 0)
                })
        except:
            continue

    return pd.DataFrame(datos)

def predecir_total_2026(col):
    """
    Retorna 5 valores: total, etiquetas, predicciones, historico_valores, historico_anios
    """
    df = preparar_mensual(col)
    
    if df.empty or len(df) < 12:
        return 0, [], [], [], []

    # Crear índice temporal continuo
    df['time_index'] = df['anio'] + (df['mes'] - 1) / 12.0
    
    X = df[['time_index']]
    y = df['total']
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Generar proyección para 2026
    meses_txt = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
    futuro = np.array([2026 + (m-1)/12.0 for m in range(1, 13)]).reshape(-1, 1)
    X_futuro = pd.DataFrame(futuro, columns=['time_index'])
    
    predicciones = model.predict(X_futuro)
    predicciones = np.maximum(predicciones, 0) # Evitar negativos
    
    total_2026 = int(predicciones.sum())
    
    return total_2026, meses_txt, predicciones.tolist(), df['total'].tail(12).tolist(), df['anio'].tail(12).tolist()

def obtener_contexto_ia(col):
    total_2026, _, _, historico_val, historico_anio = predecir_total_2026(col)
    
    if total_2026 == 0:
        return 0, "Datos insuficientes"
    
    texto_contexto = ""
    limit = min(3, len(historico_val))
    for i in range(1, limit + 1):
        texto_contexto += f"[{historico_anio[-i]}: {int(historico_val[-i])} casos] "
            
    return total_2026, texto_contexto

# ==========================================
# 3. FUNCIONES DEL SIMULADOR DE RIESGO (FALTABAN ESTAS)
# ==========================================

def entrenar_modelo_riesgo(col, modalidad_objetivo="Extorsión"):
    """
    Prepara el modelo Random Forest para el Agente Logístico
    """
    data = list(col.find(
        {"P_MODALIDADES": modalidad_objetivo},
        {"ANIO": 1, "trimestre": 1, "DPTO_HECHO_NEW": 1, "cantidad": 1, "_id": 0}
    ))
    
    if not data:
        return None, pd.DataFrame(), None

    df = pd.DataFrame(data)
    
    # Normalización de nombres (por si acaso vienen distintos)
    df.rename(columns={
        "ANIO": "anio", 
        "DPTO_HECHO_NEW": "departamento", 
        "cantidad": "total"
    }, inplace=True)

    # Limpieza básica
    df = df.dropna()
    df['anio'] = pd.to_numeric(df['anio'], errors='coerce').fillna(0).astype(int)
    
    # Mapeo de Trimestres
    trim_map = {"T1": 1, "T2": 2, "T3": 3, "T4": 4}
    df['trimestre_num'] = df['trimestre'].map(trim_map).fillna(1).astype(int)

    # Codificación de Departamentos
    le_dpto = LabelEncoder()
    df['dpto_code'] = le_dpto.fit_transform(df['departamento'].astype(str))

    # Entrenar Random Forest
    X = df[['anio', 'trimestre_num', 'dpto_code']]
    y = df['total']
    
    rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
    rf_model.fit(X, y)
    
    # Agregamos una columna 'periodo' para gráficos
    df['periodo'] = df['anio'].astype(str) + "-" + df['trimestre']
    
    return rf_model, df, le_dpto

def predecir_valor_especifico(modelo, le_dpto, anio, trimestre_str, departamento):
    """
    Usa el modelo entrenado para predecir un valor puntual
    """
    if not modelo: return 0
    
    try:
        trim_map = {"T1": 1, "T2": 2, "T3": 3, "T4": 4}
        trim_num = trim_map.get(trimestre_str, 1)
        
        # Manejo de error si el departamento no existe en el entrenamiento
        try:
            dpto_code = le_dpto.transform([departamento])[0]
        except:
            # Si es un dpto nuevo, usamos el código 0 o promedio (aquí simplificamos)
            dpto_code = 0 
            
        prediccion = modelo.predict([[anio, trim_num, dpto_code]])
        return int(max(prediccion[0], 0))
    except Exception as e:
        print(f"Error predicción específica: {e}")
        return 0