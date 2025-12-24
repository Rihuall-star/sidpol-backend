# ver_errores.py
from pymongo import MongoClient

# Conexión (Ajusta si tu URI es diferente)
client = MongoClient("mongodb://localhost:27017/")
db = client["sidpol_db"] # OJO: Pon el nombre real de tu BD
col = db["denuncias"]    # OJO: Pon el nombre real de tu colección

print("--- AUDITORÍA DE VARIANTES DE EXTORSIÓN ---")

pipeline = [
    {
        "$match": {
            # Buscamos cualquier cosa que se parezca a "EXTOR"
            "P_MODALIDADES": {"$regex": "EXTOR", "$options": "i"}
        }
    },
    {
        "$group": {
            "_id": "$P_MODALIDADES", # Agrupamos por cómo está escrito
            "total": {"$sum": 1}     # Contamos cuántos hay de cada uno
        }
    },
    { "$sort": {"total": -1} }
]

resultados = list(col.aggregate(pipeline))

total_general = 0
for r in resultados:
    print(f"VARIANTE: '{r['_id']}' \t---> CANTIDAD: {r['total']}")
    total_general += r['total']

print("---------------------------------------------")
print(f"TOTAL REAL SUMADO: {total_general}")