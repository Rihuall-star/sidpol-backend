import os
from pymongo import MongoClient
from dotenv import load_dotenv

# 1. Cargar las llaves del archivo .env
load_dotenv()

# 2. Obtener la URI de la nube
uri_nube = os.getenv('MONGO_URI_ATLAS')

if not uri_nube:
    print("❌ ERROR: No encontré la variable 'MONGO_URI_ATLAS' en el archivo .env")
    print("Asegúrate de haberla creado.")
    exit()

print("🔌 Conectando a MongoDB Atlas...")

try:
    # 3. Conexión Segura
    client = MongoClient(uri_nube)
    db = client['denuncias_db'] 
    users_col = db['usuarios']

    usuario = "admin" # El usuario que quieres ascender

    # 4. Actualizar
    resultado = users_col.update_one(
        {"username": usuario},
        {"$set": {"rol": "admin"}}
    )

    if resultado.matched_count > 0:
        print(f"✅ ¡LISTO! El usuario '{usuario}' ahora es ADMIN en la nube.")
    else:
        print(f"⚠️ No se encontró al usuario '{usuario}'. ¿Ya lo creaste en la web?")

except Exception as e:
    print(f"❌ Error de conexión: {e}")