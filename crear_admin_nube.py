import os
from pymongo import MongoClient
from werkzeug.security import generate_password_hash
from dotenv import load_dotenv

# 1. Cargar configuración
load_dotenv()
uri_nube = os.getenv('MONGO_URI_ATLAS')

if not uri_nube:
    print("❌ ERROR: No se encontró 'MONGO_URI_ATLAS' en el archivo .env")
    exit()

print("🔌 Conectando a la Nube...")

try:
    # 2. Conectar
    client = MongoClient(uri_nube)
    db = client['denuncias_db']
    users_col = db['usuarios']

    # 3. Datos del Super Admin
    usuario = "admin"
    password_plano = "Admin2025!"  # <--- Tu contraseña deseada
    password_encriptado = generate_password_hash(password_plano)

    # 4. CREAR (o Actualizar si ya existe)
    # Usamos update_one con upsert=True para que funcione siempre
    users_col.update_one(
        {"username": usuario},
        {
            "$set": {
                "password": password_encriptado,
                "rol": "admin"  # <--- Aquí le damos el poder directamente
            }
        },
        upsert=True # Si no existe, lo crea. Si existe, lo actualiza.
    )

    print(f"✅ ¡ÉXITO! Usuario '{usuario}' creado/actualizado en MongoDB Atlas.")
    print(f"🔑 Contraseña: {password_plano}")
    print("🚀 Ahora intenta iniciar sesión en tu web publicada.")

except Exception as e:
    print(f"❌ Error: {e}")