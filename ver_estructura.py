from pymongo import MongoClient
import pprint

# Conexión
client = MongoClient("mongodb://localhost:27017/")
db = client["sidpol_db"] 
col = db["denuncias"]

# 1. ¿Hay datos?
total_docs = col.count_documents({})
print(f"--- DIAGNÓSTICO DE SALUD DE LA BD ---")
print(f"Total de documentos en la colección: {total_docs}")

if total_docs == 0:
    print("❌ ERROR CRÍTICO: La colección está vacía. Debes volver a cargar el CSV.")
else:
    print("✅ La colección tiene datos.")
    
    # 2. Ver la estructura real
    print("\n--- ESTRUCTURA DE UN DOCUMENTO REAL ---")
    un_dato = col.find_one()
    pprint.pprint(un_dato)
    
    print("\n---------------------------------------")
    print("TAREA: Mira la lista de arriba.")
    print("Busca en qué campo aparecen palabras como 'Hurto', 'Robo', 'Extorsión'.")
    print("¿Se llama 'P_MODALIDADES'? ¿O se llama 'MODALIDAD', 'TIPO_DELITO', 'SUB_TIPO'?")