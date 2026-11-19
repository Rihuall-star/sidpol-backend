"""
ETL de SIDPOL: CSV de datos abiertos → validación → MongoDB → precálculo.

Uso:
    python etl_carga_mongo.py denuncias_sidpol.csv                 # un archivo
    python etl_carga_mongo.py "datos/DATASET_*.csv"                # varios (uno por año)
    python etl_carga_mongo.py datos/                               # todos los CSV de una carpeta
    python etl_carga_mongo.py datos/ --solo-validar                # revisa sin cargar nada

- Lee la conexión desde MONGO_URI (archivo .env), nunca desde el código.
- Detecta el separador (las versiones del dataset usan ';' o ',').
- Acepta los nombres de columna de todas las versiones publicadas:
  ANIO/MES (hasta oct-2025) y ANIO_REGISTRO/MES_REGISTRO (versión 2026).
- Valida tipos y rangos; las filas inválidas (p. ej. la fila de total general) se descartan y se informan.
- Carga en una colección temporal y la reemplaza de una vez (la web nunca ve datos a medias).
- Al final ejecuta el precálculo con una nueva versión del dataset, que guarda la fecha de corte y los archivos.
"""
import csv
import glob
import logging
import os
import sys

import pandas as pd

COLUMNAS = ["ANIO", "MES", "DPTO_HECHO_NEW", "PROV_HECHO", "DIST_HECHO", "UBIGEO_HECHO", "P_MODALIDADES", "cantidad"]
ALIAS = {"ANIO_REGISTRO": "ANIO", "MES_REGISTRO": "MES", "ANIO_HECHO": "ANIO", "MES_HECHO": "MES"}
LOTE = 20_000
log = logging.getLogger("etl")


def expandir_rutas(entradas):
    """Archivos, carpetas o patrones → lista ordenada de CSV."""
    rutas = []
    for e in entradas:
        if os.path.isdir(e):
            rutas += glob.glob(os.path.join(e, "*.csv"))
        elif any(c in e for c in "*?["):
            rutas += glob.glob(e)
        else:
            rutas.append(e)
    rutas = sorted(dict.fromkeys(rutas))
    if not rutas:
        raise FileNotFoundError(f"No se encontraron CSV en: {entradas}")
    return rutas


def leer_csv(ruta):
    with open(ruta, encoding="utf-8-sig") as f:
        separador = csv.Sniffer().sniff(f.readline(), delimiters=";,|\t").delimiter
    df = pd.read_csv(ruta, sep=separador, encoding="utf-8-sig", dtype={"UBIGEO_HECHO": str}, low_memory=False)
    df = df.rename(columns={c: ALIAS.get(c.strip(), c.strip()) for c in df.columns})
    faltan = [c for c in COLUMNAS if c not in df.columns]
    if faltan:
        raise ValueError(f"Faltan columnas en {os.path.basename(ruta)}: {faltan}")
    return df[COLUMNAS]


def leer_varios(rutas):
    return pd.concat([leer_csv(r) for r in rutas], ignore_index=True)


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


def resumen(df):
    """Datos de la versión: periodo cubierto, total y modalidades."""
    ultimo_anio = int(df["ANIO"].max())
    return {
        "desde": f"{int(df['ANIO'].min())}-01",
        "corte": f"{ultimo_anio}-{int(df.loc[df['ANIO'] == ultimo_anio, 'MES'].max()):02d}",
        "denuncias": int(df["cantidad"].sum()),
        "filas": len(df),
        "modalidades": sorted(df["P_MODALIDADES"].dropna().unique().tolist()),
        "por_anio": {str(a): int(v) for a, v in df.groupby("ANIO")["cantidad"].sum().items()},
    }


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
    import argparse

    from dotenv import load_dotenv

    from sidpol.registro import configurar_logs

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("entradas", nargs="*", default=["denuncias_sidpol.csv"], help="CSV, carpetas o patrones")
    parser.add_argument("--solo-validar", action="store_true", help="lee y valida, pero no carga en MongoDB")
    args = parser.parse_args(argv[1:])

    load_dotenv()
    configurar_logs()
    rutas = expandir_rutas(args.entradas)
    df, descartadas = validar(leer_varios(rutas))
    info = resumen(df)
    log.info("CSV validado", extra={"archivos": len(rutas), "descartadas": descartadas,
                                    **{k: v for k, v in info.items() if k != "por_anio"}})
    if args.solo_validar:
        print({"archivos": [os.path.basename(r) for r in rutas], "descartadas": descartadas, **info})
        return 0

    from pymongo import MongoClient

    from sidpol import precalculo
    from sidpol.repositorio import Repositorio

    uri = os.getenv("MONGO_URI")
    if not uri:
        log.error("falta MONGO_URI en el entorno o en .env")
        return 1
    db = MongoClient(uri, serverSelectionTimeoutMS=15000)[os.getenv("MONGO_DB", "denuncias_db")]
    n = cargar(df, db)
    log.info("datos cargados", extra={"documentos": n})
    repo = Repositorio(db)
    print(precalculo.ejecutar(repo, nueva_version=True,
                              datos_version={**info, "archivos": [os.path.basename(r) for r in rutas],
                                             "descartadas": descartadas}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
