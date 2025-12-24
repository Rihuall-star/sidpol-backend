import os
from pymongo import MongoClient
from dotenv import load_dotenv

# 1. Cargar configuración
load_dotenv()
uri_nube = os.getenv('MONGO_URI_ATLAS')

if not uri_nube:
    print("❌ ERROR: No tienes MONGO_URI_ATLAS en tu .env")
    exit()

print("🔌 Conectando a la Nube para corregir nombres...")

try:
    client = MongoClient(uri_nube)
    db = client['denuncias_db']
    col = db['denuncias']

    # 2. DEFINIR LOS CAMBIOS
    # Formato: "NOMBRE_VIEJO": "nombre_nuevo_que_quiere_tu_codigo"
    cambios = {
        "ANIO": "anio",
        "MES": "mes",
        "DPTO_HECHO_NEW": "dpto",
        "PROV_HECHO": "provincia",
        "DIST_HECHO": "distrito",
        "P_MODALIDADES": "modalidad" # Para que coincida con tus filtros
    }

    print("🛠️ Ejecutando cambio masivo de nombres...")
    
    # 3. EJECUTAR EL RENOMBRADO
    # update_many con $rename busca el campo viejo y le pone el nombre nuevo
    resultado = col.update_many({}, {"$rename": cambios})

    print(f"✅ ¡LISTO! Se procesaron los documentos.")
    print(f"Modificados: {resultado.modified_count}")
    print("🚀 Ahora tu base de datos habla el mismo idioma que tu Python.")

except Exception as e:
    print(f"❌ Error: {e}")