import os
from pymongo import MongoClient
from dotenv import load_dotenv

# 1. Cargar configuración
load_dotenv()
uri_nube = os.getenv('MONGO_URI_ATLAS')

if not uri_nube:
    print("❌ ERROR: No tienes MONGO_URI_ATLAS en tu .env")
    exit()

print("🔌 Conectando a la Nube...")
try:
    client = MongoClient(uri_nube)
    db = client['denuncias_db'] # <--- CONFIRMA SI TU BD SE LLAMA ASÍ
    col = db['denuncias']       # <--- CONFIRMA SI TU COLECCIÓN SE LLAMA ASÍ

    # 2. Traer un solo documento para ver su estructura
    dato = col.find_one()

    if dato:
        print("\n✅ ¡DATO ENCONTRADO! Así se ven tus campos en la nube:\n")
        print(dato)
        print("\n----------------------------------------------------")
        print("🔍 BUSCA LOS NOMBRES DE LAS COLUMNAS:")
        print("¿Dice 'anio' o 'AÑO' o 'year'?")
        print("¿Dice 'dpto' o 'DEPARTAMENTO'?")
    else:
        print("⚠️ La colección está vacía. No hay datos para leer.")

except Exception as e:
    print(f"❌ Error: {e}")