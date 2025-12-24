import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()
uri_nube = os.getenv('MONGO_URI_ATLAS')

print("🚑 INICIANDO ROLLBACK (Revertir cambios)...")

try:
    client = MongoClient(uri_nube)
    db = client['denuncias_db']
    col = db['denuncias']

    # INVERTIMOS LA LÓGICA:
    # Ahora cambiamos de "anio" (el nuevo) -> a "ANIO" (el original)
    cambios_rollback = {
        "anio": "ANIO",
        "mes": "MES",
        "dpto": "DPTO_HECHO_NEW",
        "provincia": "PROV_HECHO",
        "distrito": "DIST_HECHO",
        "modalidad": "P_MODALIDADES"
    }

    print("🛠️ Restaurando nombres originales...")
    
    resultado = col.update_many({}, {"$rename": cambios_rollback})

    print(f"✅ ¡ROLLBACK COMPLETADO!")
    print(f"Documentos restaurados: {resultado.modified_count}")
    print("Por favor, verifica que tus estadísticas volvieron a la normalidad.")

except Exception as e:
    print(f"❌ Error: {e}")