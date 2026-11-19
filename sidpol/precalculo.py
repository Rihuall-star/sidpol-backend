"""
Job de precálculo (batch). Se ejecuta después de cargar datos o en cada despliegue:

    python -m sidpol.precalculo               # índices + capa gold + resultados
    python -m sidpol.precalculo --nueva-version   # además marca una carga nueva del dataset
    flask --app app precalcular

1. Crea los índices que usan las consultas.
2. Construye `gold_mensual`: denuncias por año, mes, departamento y modalidad
   (unos 17 mil documentos frente a los 337 mil crudos). Se arma en una colección temporal
   y se reemplaza de una vez, así la web nunca ve una capa a medio construir.
3. Calcula y guarda todos los resultados (secciones, SARIMA, K-Means, contexto de la IA).
"""
import argparse
import logging
import sys
import time
from datetime import datetime

from pymongo import ASCENDING, DESCENDING

from sidpol.repositorio import GOLD, Repositorio
from sidpol.servicios import resultados

log = logging.getLogger(__name__)


def crear_indices(repo):
    indices = {
        "denuncias": [[("P_MODALIDADES", ASCENDING), ("ANIO", ASCENDING)],
                      [("DPTO_HECHO_NEW", ASCENDING), ("P_MODALIDADES", ASCENDING)],
                      [("ANIO", ASCENDING), ("MES", ASCENDING)]],
        "auditoria": [[("fecha", DESCENDING)]],
        "usuarios": [[("username", ASCENDING)]],
    }
    for coleccion, lista in indices.items():
        for campos in lista:
            try:
                repo.db[coleccion].create_index(campos)
            except Exception as e:  # noqa: BLE001
                log.warning("no se pudo crear el índice", extra={"coleccion": coleccion, "error": str(e)})
    try:  # las respuestas guardadas de la IA se borran solas a los 7 días
        repo.db["cache_ia"].create_index("creado", expireAfterSeconds=7 * 24 * 3600)
    except Exception as e:  # noqa: BLE001
        log.warning("no se pudo crear el índice TTL de cache_ia: %s", e)


def construir_gold(repo):
    filas = repo.denuncias.aggregate([
        {"$group": {"_id": {"a": "$ANIO", "m": "$MES", "p": "$DPTO_HECHO_NEW", "mod": "$P_MODALIDADES"},
                    "cantidad": {"$sum": "$cantidad"}}},
    ], allowDiskUse=True)
    docs = []
    for f in filas:
        k = f["_id"]
        doc = {"ANIO": k.get("a"), "MES": k.get("m"), "DPTO_HECHO_NEW": k.get("p"),
               "P_MODALIDADES": k.get("mod"), "cantidad": f["cantidad"]}
        try:
            trimestre = f"T{(int(doc['MES']) - 1) // 3 + 1}"
            doc.update(trimestre=trimestre, anio_trimestre=f"{int(doc['ANIO'])}-{trimestre}")
        except (TypeError, ValueError):
            pass
        docs.append(doc)

    temporal = repo.db[GOLD + "_tmp"]
    temporal.drop()
    if docs:
        temporal.insert_many(docs)
        temporal.create_index([("P_MODALIDADES", ASCENDING), ("DPTO_HECHO_NEW", ASCENDING), ("ANIO", ASCENDING)])
        try:
            temporal.rename(GOLD, dropTarget=True)   # reemplazo en un solo paso
        except Exception as e:  # noqa: BLE001  (algunos planes compartidos de Atlas no permiten renombrar)
            log.warning("rename no disponible, se reemplaza la capa gold directamente: %s", e)
            repo.gold.delete_many({})
            repo.gold.insert_many(docs)
            temporal.drop()
    else:
        repo.gold.drop()
    repo.olvidar_memoria()
    return len(docs)


def ejecutar(repo, nueva_version=False, datos_version=None):
    inicio = time.perf_counter()
    crear_indices(repo)
    if nueva_version:
        repo.registrar_version(datetime.now().strftime("%Y%m%d%H%M%S"), repo.denuncias.estimated_document_count(),
                               **(datos_version or {}))
    n_gold = construir_gold(repo)
    tiempos = resultados.precalcular_todo(repo)
    resumen = {"gold_documentos": n_gold, "tiempos": tiempos, "version": repo.version_dataset(ttl=0),
               "segundos": round(time.perf_counter() - inicio, 1)}
    log.info("precálculo terminado", extra=resumen)
    return resumen


def registrar_comando(app):
    @app.cli.command("precalcular")
    def precalcular():
        """Construye la capa gold y guarda los resultados precalculados."""
        print(ejecutar(app.extensions["sidpol_repo"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nueva-version", action="store_true", help="marca una nueva carga del dataset")
    parser.add_argument("--tolerante", action="store_true",
                        help="no falla si la base no responde (útil en el build de Render)")
    args = parser.parse_args(argv)

    import os
    from dotenv import load_dotenv
    from pymongo import MongoClient
    from sidpol.registro import configurar_logs

    load_dotenv()
    configurar_logs()
    try:
        cliente = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017/"), serverSelectionTimeoutMS=15000)
        repo = Repositorio(cliente[os.getenv("MONGO_DB", "denuncias_db")])
        repo.ping()
        print(ejecutar(repo, nueva_version=args.nueva_version))
    except Exception as e:  # noqa: BLE001
        log.error("el precálculo falló: %s", e)
        if not args.tolerante:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
