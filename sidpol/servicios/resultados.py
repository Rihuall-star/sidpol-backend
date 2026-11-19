"""
Capa de resultados precalculados.

Cada cálculo pesado (secciones, SARIMA, K-Means, contexto de la IA) se guarda en la colección
`resultados` junto con la versión del dataset y la versión del cálculo. Una visita solo lee
un documento; el cálculo se repite únicamente si cambian los datos o el código del cálculo.

Antes, todo vivía en la memoria del proceso: cada vez que Render "despertaba" el servidor
gratuito había que volver a ajustar SARIMA. Ahora el resultado sobrevive a reinicios y despliegues.
"""
import json
import logging
import threading
import time
from datetime import datetime

from sidpol.servicios import analitica, modelos

log = logging.getLogger(__name__)

# nombre -> (versión del cálculo, función que recibe el repositorio)
# Sube la versión de un cálculo cuando cambie su código o la forma de su resultado.
CALCULOS = {
    "panorama": (1, lambda r: analitica.panorama(r.agregada())),
    "delitos": (2, lambda r: analitica.delitos(r.agregada())),
    "territorio": (1, lambda r: analitica.territorio(r.agregada(), r.denuncias)),
    "modalidades": (1, lambda r: analitica.evolucion_modalidades(r.agregada())),
    "proyeccion": (2, lambda r: modelos.proyeccion(r.agregada())),
    "mapa": (1, lambda r: modelos.datos_mapa(r.agregada())),
    "clusters": (1, lambda r: modelos.clusters_departamentos(r.agregada())),
    "simulador": (1, lambda r: modelos.datos_simulador(r.agregada())),
}

_memoria = {}            # (nombre, versión del dataset) -> valor
_bloqueo = threading.Lock()


def _json(valor):
    return json.dumps(valor, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def obtener(repo, nombre):
    """Devuelve el resultado vigente: memoria del proceso → colección `resultados` → cálculo."""
    version_calculo, funcion = CALCULOS[nombre]
    version = repo.version_dataset()
    clave = (nombre, version)
    if clave in _memoria:
        return _memoria[clave]

    with _bloqueo:  # evita que dos visitas simultáneas hagan el mismo cálculo pesado
        if clave in _memoria:
            return _memoria[clave]
        doc = repo.resultados.find_one({"_id": nombre})
        if doc and doc.get("version_dataset") == version and doc.get("version_calculo") == version_calculo:
            valor = json.loads(doc["valor"])
        else:
            valor = _calcular_y_guardar(repo, nombre, version)
        for k in [k for k in _memoria if k[0] == nombre]:
            del _memoria[k]
        _memoria[clave] = valor
        return valor


def _calcular_y_guardar(repo, nombre, version):
    version_calculo, funcion = CALCULOS[nombre]
    inicio = time.perf_counter()
    valor = funcion(repo)
    texto = _json(valor)
    segundos = round(time.perf_counter() - inicio, 2)
    repo.resultados.replace_one(
        {"_id": nombre},
        {"_id": nombre, "version_dataset": version, "version_calculo": version_calculo,
         "valor": texto, "segundos": segundos, "calculado": datetime.now()},
        upsert=True,
    )
    log.info("resultado calculado", extra={"resultado": nombre, "segundos": segundos})
    return json.loads(texto)  # misma forma que si se hubiera leído de la base


def precalcular_todo(repo):
    """Recalcula y guarda todos los resultados. Devuelve el tiempo de cada uno."""
    olvidar()
    version = repo.version_dataset(ttl=0)
    tiempos = {}
    for nombre in CALCULOS:
        inicio = time.perf_counter()
        _memoria[(nombre, version)] = _calcular_y_guardar(repo, nombre, version)
        tiempos[nombre] = round(time.perf_counter() - inicio, 2)
    return tiempos


def olvidar():
    """Vacía la memoria del proceso (los documentos guardados en la base se mantienen)."""
    _memoria.clear()
