"""
ETL de SIDPOL: CSV de datos abiertos → validación → MongoDB → precálculo.

Uso:
    python etl_carga_mongo.py [ruta.csv]

- Lee la conexión desde MONGO_URI (archivo .env), nunca desde el código.
- Detecta el separador (el dataset oficial usa ';').
- Valida tipos y rangos; las filas inválidas se descartan y se informan.
- Carga en una colección temporal y la reemplaza de una vez (la web nunca ve datos a medias).
- Al final ejecuta el precálculo: capa gold + resultados, con una nueva versión del dataset.
"""
import csv
import logging
import os
import sys

import pandas as pd

COLUMNAS = ["ANIO", "MES", "DPTO_HECHO_NEW", "PROV_HECHO", "DIST_HECHO", "UBIGEO_HECHO", "P_MODALIDADES", "cantidad"]
LOTE = 20_000
log = logging.getLogger("etl")


def leer_csv(ruta):
    with open(ruta, encoding="utf-8-sig") as f:
        separador = csv.Sniffer().sniff(f.readline(), delimiters=";,|\t").delimiter
    df = pd.read_csv(ruta, sep=separador, encoding="utf-8-sig", dtype={"UBIGEO_HECHO": str})
    faltan = [c for c in COLUMNAS if c not in df.columns]
    if faltan:
        raise ValueError(f"Faltan columnas en el CSV: {faltan}")
    return df[COLUMNAS]


def validar(df):
    """Devuelve (filas válidas, cantidad de filas descartadas)."""
    df = df.copy()
    for c in ("ANIO", "MES", "cantidad"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    validas = df["ANIO"].between(2000, 2100) & df["MES"].between(1, 12) & (df["cantidad"] >= 0)
    descartadas = int((~validas).sum())
    df = df[validas].astype({"ANIO": int, "MES": int, "cantidad": int})
    for c in ("DPTO_HECHO_NEW", "PROV_HECHO", "DIST_HECHO", "P_MODALIDADES"):
        df[c] = df[c].where(df[c].notna(), None).map(lambda v: v.strip() if isinstance(v, str) else v)
    df["trimestre"] = "T" + ((df["MES"] - 1) // 3 + 1).astype(str)
    df["anio_trimestre"] = df["ANIO"].astype(str) + "-" + df["trimestre"]
    return df, descartadas


def cargar(df, db):
    temporal = db["denuncias_tmp"]
    temporal.drop()
    registros = df.where(pd.notna(df), None).to_dict(orient="records")
    for i in range(0, len(registros), LOTE):
        temporal.insert_many(registros[i:i + LOTE], ordered=False)
    try:
        temporal.rename("denuncias", dropTarget=True)
    except Exception as e:  # noqa: BLE001  (algunos planes compartidos de Atlas no permiten renombrar)
        log.warning("rename no disponible, se reemplaza la colección directamente: %s", e)
        db["denuncias"].delete_many({})
        for i in range(0, len(registros), LOTE):
            db["denuncias"].insert_many([{k: v for k, v in r.items() if k != "_id"} for r in registros[i:i + LOTE]],
                                        ordered=False)
        temporal.drop()
    return len(registros)


def main(argv):
    from dotenv import load_dotenv
    from pymongo import MongoClient

    from sidpol import precalculo
    from sidpol.registro import configurar_logs
    from sidpol.repositorio import Repositorio

    load_dotenv()
    configurar_logs()
    ruta = argv[1] if len(argv) > 1 else "denuncias_sidpol.csv"
    uri = os.getenv("MONGO_URI")
    if not uri:
        log.error("falta MONGO_URI en el entorno o en .env")
        return 1

    df, descartadas = validar(leer_csv(ruta))
    log.info("CSV validado", extra={"filas": len(df), "descartadas": descartadas})
    db = MongoClient(uri, serverSelectionTimeoutMS=15000)[os.getenv("MONGO_DB", "denuncias_db")]
    n = cargar(df, db)
    log.info("datos cargados", extra={"documentos": n})
    print(precalculo.ejecutar(Repositorio(db), nueva_version=True))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
