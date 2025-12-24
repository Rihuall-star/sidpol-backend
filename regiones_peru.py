# regiones_peru.py
# Mapeo simple de departamento -> región tradicional

regiones_departamentos = {
    # Costa
    "TUMBES": "COSTA",
    "PIURA": "COSTA",
    "LAMBAYEQUE": "COSTA",
    "LA LIBERTAD": "COSTA",
    "ANCASH": "COSTA",
    "LIMA METROPOLITANA": "COSTA",
    "REGION LIMA": "COSTA",
    "ICA": "COSTA",
    "AREQUIPA": "COSTA",
    "MOQUEGUA": "COSTA",
    "TACNA": "COSTA",
    "PROV. CONST. DEL CALLAO": "COSTA",

    # Sierra
    "CAJAMARCA": "SIERRA",
    "AYACUCHO": "SIERRA",
    "APURIMAC": "SIERRA",
    "CUSCO": "SIERRA",
    "HUANCAVELICA": "SIERRA",
    "JUNIN": "SIERRA",
    "PASCO": "SIERRA",
    "PUNO": "SIERRA",
    "HUANUCO": "SIERRA",

    # Selva (incluye ceja de selva para simplificar)
    "AMAZONAS": "SELVA",
    "LORETO": "SELVA",
    "SAN MARTIN": "SELVA",
    "UCAYALI": "SELVA",
    "MADRE DE DIOS": "SELVA",
}
