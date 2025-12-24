# ml_riesgo.py
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def preparar_dataset_riesgo(col, modalidad_objetivo="Extorsión"):
    """
    Arma un dataset a nivel:
      - año
      - trimestre (T1..T4)
      - departamento
      - total de denuncias de la modalidad objetivo
    """
    pipeline = [
        {"$match": {"P_MODALIDADES": modalidad_objetivo}},
        {"$group": {
            "_id": {
                "anio": "$ANIO",
                "trimestre": "$trimestre",
                "dpto": "$DPTO_HECHO_NEW"
            },
            "total": {"$sum": "$cantidad"}
        }},
        {"$sort": {
            "_id.anio": 1,
            "_id.trimestre": 1,
            "_id.dpto": 1
        }}
    ]

    datos = list(col.aggregate(pipeline))
    if not datos:
        return pd.DataFrame()

    filas = []
    for d in datos:
        filas.append({
            "anio": d["_id"]["anio"],
            "trimestre": d["_id"]["trimestre"],
            "departamento": d["_id"]["dpto"],
            "total": d["total"]
        })
    df = pd.DataFrame(filas)

    # Convertir T1..T4 -> 1..4
    mapa_tri = {"T1": 1, "T2": 2, "T3": 3, "T4": 4}
    df["tri_num"] = df["trimestre"].map(mapa_tri)

    return df


def entrenar_modelo_riesgo(col, modalidad_objetivo="Extorsión"):
    """
    Entrena un RandomForestRegressor para predecir el número
    de denuncias trimestrales de la modalidad objetivo.
    """
    df = preparar_dataset_riesgo(col, modalidad_objetivo)
    if df.empty:
        return None, None

    X = df[["anio", "tri_num"]]  # features simples (puedes ampliar)
    y = df["total"]

    modelo = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

    modelo.fit(X, y)
    return modelo, df
