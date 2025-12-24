import pandas as pd
from pymongo import MongoClient

# ========= CONFIGURACIÓN =========

# Nombre del archivo CSV (ajusta según tu caso)
CSV_FILE = "denuncias_sidpol.csv"

# URI de conexión a MongoDB
# Si usas MongoDB local, deja esta:
MONGO_URI = "mongodb://localhost:27017"

# Nombre de la base de datos y colección
DB_NAME = "denuncias_db"
COLLECTION_NAME = "denuncias"

# ========= FUNCIONES AUXILIARES =========

def get_trimestre(mes: int) -> str:
    """Devuelve el trimestre (T1, T2, T3, T4) según el mes (1-12)."""
    if mes in [1, 2, 3]:
        return "T1"
    elif mes in [4, 5, 6]:
        return "T2"
    elif mes in [7, 8, 9]:
        return "T3"
    else:
        return "T4"


def cargar_csv_en_dataframe(csv_path: str) -> pd.DataFrame:
    """Lee el CSV y devuelve un DataFrame de pandas."""
    print(f"Leyendo CSV: {csv_path} ...")
    df = pd.read_csv(csv_path)

    print("Columnas encontradas:")
    print(df.columns)

    # Asegurar que existen columnas ANIO y MES (ajusta los nombres si difieren)
    # Si tus columnas se llaman distinto (ej. 'ANIO', 'MES', 'DPTO_HECHO_NEW'), cámbialas aquí.
    if "ANIO" not in df.columns or "MES" not in df.columns:
        raise ValueError("El CSV debe tener columnas 'ANIO' y 'MES'. Ajusta el script si tienen otro nombre.")

    # Crear columna de trimestre y año-trimestre
    df["trimestre"] = df["MES"].apply(get_trimestre)
    df["anio_trimestre"] = df["ANIO"].astype(str) + "-" + df["trimestre"]

    print("Primeras filas del DataFrame después de procesar:")
    print(df.head())

    return df


def conectar_mongo(uri: str) -> MongoClient:
    """Devuelve un cliente de MongoDB conectado."""
    print(f"Conectando a MongoDB en {uri} ...")
    client = MongoClient(uri)
    print("Conexión a MongoDB OK.")
    return client


def cargar_dataframe_a_mongo(df: pd.DataFrame, client: MongoClient, db_name: str, collection_name: str):
    """Inserta los registros del DataFrame en MongoDB."""
    db = client[db_name]
    collection = db[collection_name]

    # Opcional: borrar datos anteriores
    resp = collection.delete_many({})
    print(f"Documentos eliminados anteriormente en la colección: {resp.deleted_count}")

    # Convertir DataFrame a lista de diccionarios (documentos)
    registros = df.to_dict(orient="records")

    if not registros:
        print("No hay registros para insertar. Revisa el CSV.")
        return

    print(f"Insertando {len(registros)} documentos en MongoDB...")
    result = collection.insert_many(registros)
    print(f"Insertados {len(result.inserted_ids)} documentos en la colección '{collection_name}'.")


# ========= MAIN =========

if __name__ == "__main__":
    try:
        # 1. Leer y procesar el CSV
        df = cargar_csv_en_dataframe(CSV_FILE)

        # 2. Conectarse a MongoDB
        client = conectar_mongo(MONGO_URI)

        # 3. Cargar DataFrame a Mongo
        cargar_dataframe_a_mongo(df, client, DB_NAME, COLLECTION_NAME)

        print("Proceso ETL finalizado correctamente ✅")

    except Exception as e:
        print("Ocurrió un error durante el proceso ETL:")
        print(e)
