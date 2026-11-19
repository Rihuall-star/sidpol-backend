"""
Repositorio: el único módulo que conoce las colecciones de MongoDB.

Las rutas y los servicios piden datos al repositorio en lugar de tocar la base directamente,
así se puede cambiar el almacenamiento o probar todo con una base en memoria (mongomock).

Colecciones:
- denuncias      datos crudos del dataset (un registro por distrito, mes y modalidad)
- gold_mensual   agregado por año, mes, departamento y modalidad (lo construye el precálculo)
- resultados     cálculos ya hechos (secciones, modelos), por versión del dataset
- meta           versión del dataset cargado
- usuarios, auditoria, estadisticas, cache_ia
"""
import time
from datetime import datetime

from pymongo import ReturnDocument

GOLD = "gold_mensual"


class Repositorio:
    def __init__(self, db):
        self.db = db
        self._version = (None, 0.0)       # (versión, momento de la lectura)
        self._gold_disponible = (None, 0.0)

    # ---------- colecciones ----------
    @property
    def denuncias(self):
        return self.db["denuncias"]

    @property
    def gold(self):
        return self.db[GOLD]

    @property
    def resultados(self):
        return self.db["resultados"]

    def agregada(self):
        """Colección para análisis por departamento y modalidad: la gold si existe, si no la cruda.
        Tiene los mismos nombres de campo, así los cálculos funcionan igual con cualquiera de las dos."""
        valor, leido = self._gold_disponible
        if valor is None or time.monotonic() - leido > 300:
            valor = self.gold.estimated_document_count() > 0
            self._gold_disponible = (valor, time.monotonic())
        return self.gold if valor else self.denuncias

    # ---------- versión del dataset ----------
    def version_dataset(self, ttl=300):
        """Identifica la carga de datos actual. La escribe el ETL; si no existe, se usa el conteo."""
        valor, leido = self._version
        if valor is None or time.monotonic() - leido > ttl:
            meta = self.db["meta"].find_one({"_id": "dataset"})
            valor = meta["version"] if meta else f"n{self.denuncias.estimated_document_count()}"
            self._version = (valor, time.monotonic())
        return valor

    def registrar_version(self, version, registros, **detalle):
        """`detalle`: fecha de corte, archivos de origen, denuncias por año… (lo que entregue el ETL)."""
        self.db["meta"].replace_one(
            {"_id": "dataset"},
            {"_id": "dataset", "version": version, "registros": registros, "cargado": datetime.now(), **detalle},
            upsert=True,
        )
        self.olvidar_memoria()

    def info_dataset(self):
        return self.db["meta"].find_one({"_id": "dataset"}, {"_id": 0}) or {}

    def olvidar_memoria(self):
        self._version = (None, 0.0)
        self._gold_disponible = (None, 0.0)

    def ping(self):
        self.db.command("ping")

    # ---------- usuarios ----------
    def buscar_usuario(self, username):
        return self.db["usuarios"].find_one({"username": username})

    def crear_usuario(self, username, password_hash, rol):
        if self.buscar_usuario(username):
            return False
        self.db["usuarios"].insert_one({"username": username, "password": password_hash, "rol": rol})
        return True

    def marcar_actividad(self, username):
        self.db["usuarios"].update_one({"username": username}, {"$set": {"last_seen": datetime.now()}})

    def usuarios_activos(self, desde):
        return list(self.db["usuarios"].find({"last_seen": {"$gt": desde}}, {"password": 0}))

    def consumir_cuota_ia(self, username, dia):
        """Suma una consulta del día al usuario y devuelve cuántas lleva (None si no existe)."""
        doc = self.db["usuarios"].find_one_and_update(
            {"username": username},
            [{"$set": {
                "ia_fecha": dia,
                "ia_usadas": {"$cond": [
                    {"$eq": ["$ia_fecha", dia]},
                    {"$add": [{"$ifNull": ["$ia_usadas", 0]}, 1]},
                    1,
                ]},
            }}],
            return_document=ReturnDocument.AFTER,
        )
        return None if doc is None else doc.get("ia_usadas", 0)

    # ---------- auditoría ----------
    def registrar_evento(self, usuario, evento, ip):
        ahora = datetime.now()
        self.db["auditoria"].insert_one({
            "usuario": str(usuario)[:50], "evento": evento, "fecha": ahora,
            "fecha_str": ahora.strftime("%Y-%m-%d %H:%M:%S"), "ip": ip,
        })

    def historial(self, n=50):
        return list(self.db["auditoria"].find().sort("fecha", -1).limit(n))

    def contar_visita(self):
        doc = self.db["estadisticas"].find_one_and_update(
            {"_id": "contador_home"}, {"$inc": {"cantidad": 1}},
            upsert=True, return_document=ReturnDocument.AFTER,
        )
        return doc.get("cantidad", 0) if doc else 0

    # ---------- caché de respuestas de IA ----------
    def leer_cache_ia(self, clave):
        doc = self.db["cache_ia"].find_one({"_id": clave})
        return doc["texto"] if doc else None

    def guardar_cache_ia(self, clave, texto, **extra):
        self.db["cache_ia"].replace_one(
            {"_id": clave}, {"_id": clave, "texto": texto, "creado": datetime.now(), **extra}, upsert=True)
