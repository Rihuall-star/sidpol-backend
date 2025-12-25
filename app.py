# ============================================================
# app.py — Proyecto Analítica de Denuncias + Chat IA
# ============================================================
import pandas as pd
from datetime import datetime
from ml_riesgo import entrenar_modelo_riesgo
import os
from ml_utils import predecir_total_2026
from mongo_queries import total_denuncias, ranking_departamentos, top_modalidades, modalidad_mas_frecuente, tendencia_modalidad, comparar_dos_anios
from ml_cluster import clusterizar_departamentos
import re
from sklearn.cluster import KMeans
import numpy as np
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps

from flask import (
    Flask, render_template, request, redirect,
    url_for, session, flash
)
from functools import wraps
from dotenv import load_dotenv
from pymongo import MongoClient

# ---- Módulos del motor IA ----
from gemini_client import preguntar_gemini
from chat_logic import construir_contexto

# Cargar variables del .env
load_dotenv()

# Inicializar Flask
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "clave_por_defecto_insegura")

# ================================
#  CONEXIÓN A MONGO
# ================================
#MONGO_URI = os.getenv("MONGO_URI")
#DB_NAME = os.getenv("DB_NAME")
#COLLECTION_NAME = os.getenv("COLLECTION_NAME")

#client = MongoClient(MONGO_URI)
#db = client[DB_NAME]
#col = db[COLLECTION_NAME]
# ============================================================
#  LOGIN / LOGOUT
# ============================================================
# --- CONFIGURACIÓN DE CONEXIÓN (Local vs Nube) ---
# Si existe la variable 'MONGO_URI' (en Render), la usa.
# Si no (en tu PC), usa localhost para que sigas trabajando normal.
mongo_uri = os.environ.get('MONGO_URI')

if not mongo_uri:
    # Conexión Local (Tu PC)
    mongo_uri = "mongodb://localhost:27017/"

# Conexión Maestra
client = MongoClient(mongo_uri)
db = client['denuncias_db'] # Asegúrate que este nombre sea igual al de Atlas
col = db['denuncias']
# -------------------------------------------------

# ============================================================
#  LOGIN / LOGOUT
# ============================================================
# --- DECORADOR: SOLO ADMIN ---
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Verificamos si el rol en la sesión es 'admin'
        if session.get('rol') != 'admin':
            flash('⛔ Acceso denegado: Se requieren privilegios de Administrador.', 'error')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function

# --- DECORADOR: LOGIN REQUERIDO ---
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Si no hay usuario en la sesión, lo manda al login
        if 'user' not in session:
            flash('Por favor inicia sesión para acceder.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/logout')
def logout():
    session.clear() # Borra todos los datos de la memoria (usuario, rol, etc.)
    flash('Has cerrado sesión correctamente.', 'success')
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        users_col = db['usuarios']
        user_data = users_col.find_one({"username": username})

        # --- AQUÍ ESTÁ EL CAMBIO CLAVE ---
        # Usamos check_password_hash para comparar la clave encriptada de la BD
        # con la clave normal que escribió el usuario.
        if user_data and check_password_hash(user_data['password'], password):
            session['user'] = username
            session['rol'] = user_data.get('rol', 'invitado')
            return redirect(url_for('index'))
        else:
            flash('Usuario o contraseña incorrectos', 'error')
            return redirect(url_for('login'))

    return render_template('login.html')



# ============================================================
#  PÁGINA PRINCIPAL
# ============================================================

@app.route("/")
@login_required
def index():
    # Total de registros (filas del dataset)
    total_registros = col.count_documents({})

    # Suma total de la columna "cantidad"
    pipeline = [
        {"$group": {"_id": None, "total": {"$sum": "$cantidad"}}}
    ]
    res = list(col.aggregate(pipeline))
    total_denuncias = res[0]["total"] if res else 0

    # Años disponibles en la base
    anios = sorted(col.distinct("ANIO"))

    return render_template(
        "index.html",
        total_registros=total_registros,
        total_denuncias=total_denuncias,
        anios=anios
    )



# ============================================================
#  RUTA: Resumen anual
# ============================================================

@app.route("/resumen-anual")
@login_required
def resumen_anual():
    pipeline = [
        {"$group": {"_id": "$ANIO", "total": {"$sum": "$cantidad"}}},
        {"$sort": {"_id": 1}}
    ]
    datos = list(col.aggregate(pipeline))

    # Listas para el gráfico
    labels = [doc["_id"] for doc in datos]      # Años
    valores = [doc["total"] for doc in datos]   # Totales de denuncias por año

    # La tabla puede usar directamente la misma lista
    tabla = datos

    return render_template(
        "resumen_anual.html",
        labels=labels,
        valores=valores,
        tabla=tabla
    )

# ============================================================
#  RUTA: Departamentos
# ============================================================

@app.route('/departamentos')
@login_required
def departamentos():
    data_raw = ranking_departamentos(col, n=30) # Traemos 30 por si acaso
    
    # 1. IMPRIMIR EN CONSOLA (Para depurar)
    print("--- NOMBRES EN BASE DE DATOS ---")
    
    map_mapping = {
        # Copia aquí el diccionario que ya tienes...
        "AMAZONAS": "pe-am", "CAJAMARCA": "pe-cj", "LA LIBERTAD": "pe-ll",
        "LAMBAYEQUE": "pe-lb", "PIURA": "pe-pi", "SAN MARTIN": "pe-sm", 
        "TUMBES": "pe-tu", "LORETO": "pe-lo", "ANCASH": "pe-an", 
        "CALLAO": "pe-cl", "PROV. CONST. DEL CALLAO": "pe-cl",
        "HUANUCO": "pe-hc", "JUNIN": "pe-ju", "PASCO": "pe-pa", 
        "UCAYALI": "pe-uc", "LIMA METROPOLITANA": "pe-li", "REGION LIMA": "pe-lr", 
        "ICA": "pe-ic", "APURIMAC": "pe-ap", "AREQUIPA": "pe-ar", 
        "AYACUCHO": "pe-ay", "CUSCO": "pe-cs", "MADRE DE DIOS": "pe-md", 
        "MOQUEGUA": "pe-mq", "TACNA": "pe-ta",
        
        # CODIGOS CONFIRMADOS:
        "PUNO": "pe-pu",
        "HUANCAVELICA": "pe-hv" 
    }

    data_mapa = []
    top_5 = data_raw[:5]

    for d in data_raw:
        # Limpieza básica
        nombre_bd = str(d["_id"]).strip().upper()
        
        # Imprime para ver si tiene tilde en tu consola negra
        print(f"Procesando: '{nombre_bd}' - Total: {d['total']}") 
        
        code = None
        
        # 1. Búsqueda directa
        if nombre_bd in map_mapping:
            code = map_mapping[nombre_bd]
        
        # 2. Búsqueda "Inteligente" (MATCH PARCIAL)
        if code is None:
            # Usamos "startswith" o partes de la palabra para evitar problemas de tildes
            if nombre_bd.startswith("HUANCAV"): # Captura HUANCAVELICA y HUANCAVÉLICA
                code = "pe-hv"
            elif "PUNO" in nombre_bd:
                code = "pe-pu"
            elif "LIMA" in nombre_bd:
                code = "pe-li"
            elif "CALLAO" in nombre_bd:
                code = "pe-cl"

        if code:
            data_mapa.append([code, d["total"]])
        else:
            print(f"⚠️ ALERTA: No se encontró código mapa para '{nombre_bd}'")

    return render_template('departamentos.html', 
                           data_mapa=data_mapa, 
                           top_5=top_5)


# ============================================================
#  RUTA: Departamentos por cápita
# ============================================================

@app.route("/departamentos-percapita")
@login_required
def departamentos_percapita():
    # Totales por departamento
    pipeline = [
        {"$group": {"_id": "$DPTO_HECHO_NEW", "total": {"$sum": "$cantidad"}}},
        {"$sort": {"_id": 1}}
    ]
    datos = list(col.aggregate(pipeline))

    poblaciones = {
        "MADRE DE DIOS": 184083,
        "AREQUIPA": 1523839,
        "LAMBAYEQUE": 1367029,
        "MOQUEGUA": 200973,
        "ICA": 1004829,
        "TUMBES": 280723,
        "TACNA": 397737,
        "APURIMAC": 436820,
        "JUNIN": 1418738,
        "CUSCO": 1428028,
        "ANCASH": 1202828,
        "HUANUCO": 782039,
        "AMAZONAS": 458022,
        "LA LIBERTAD": 2078028,
        "PIURA": 2138730,
        "AYACUCHO": 671182,
        "UCAYALI": 568028,
        "SAN MARTIN": 924292,
        "PASCO": 278028,
        "CAJAMARCA": 1503836,
        "PUNO": 1268093,
        "LORETO": 1138637,
        "HUANCAVELICA": 371038,
        "LIMA METROPOLITANA": 11810722,
        "REGION LIMA": 1092827,
        "PROV. CONST. DEL CALLAO": 1147628,
    }

    tabla = []
    for d in datos:
        depto = d["_id"]
        total = d["total"]
        pob = poblaciones.get(depto)
        if pob:
            tasa = (total / pob) * 100000
        else:
            tasa = None
        tabla.append({
            "departamento": depto,
            "total": total,
            "poblacion": pob,
            "tasa": tasa
        })

    # Ordenar por tasa descendente, ignorando N/D
    tabla = sorted(
        tabla,
        key=lambda x: x["tasa"] if x["tasa"] is not None else 0,
        reverse=True
    )

    labels = [row["departamento"] for row in tabla]
    valores = [row["tasa"] or 0 for row in tabla]

    return render_template(
        "departamentos_percapita.html",
        labels=labels,
        valores=valores,
        tabla=tabla
    )



# ============================================================
#  RUTA: Modalidades
# ============================================================

@app.route("/modalidades")
@login_required
def modalidades():
    pipeline = [
        {"$group": {"_id": "$P_MODALIDADES", "total": {"$sum": "$cantidad"}}},
        {"$sort": {"total": -1}}
    ]
    datos = list(col.aggregate(pipeline))

    labels = [doc["_id"] for doc in datos]
    valores = [doc["total"] for doc in datos]
    tabla = datos

    return render_template(
        "modalidades.html",
        labels=labels,
        valores=valores,
        tabla=tabla
    )



# ============================================================
#  RUTA: Detalle por modalidad
# ============================================================

@app.route("/modalidad-detalle/<modalidad>")
@login_required
def modalidad_detalle(modalidad):
    pipeline = [
        {"$match": {"P_MODALIDADES": modalidad}},
        {"$group": {"_id": "$ANIO", "total": {"$sum": "$cantidad"}}},
        {"$sort": {"_id": 1}}
    ]
    datos = list(col.aggregate(pipeline))
    return render_template("modalidad_detalle.html", modalidad=modalidad, datos=datos)


# ============================================================
#  RUTA: Trimestres
# ============================================================

@app.route("/trimestres")
@login_required
def trimestres():
    pipeline = [
        {"$group": {"_id": "$anio_trimestre", "total": {"$sum": "$cantidad"}}},
        {"$sort": {"_id": 1}}
    ]
    datos = list(col.aggregate(pipeline))

    labels = [doc["_id"] for doc in datos]
    valores = [doc["total"] for doc in datos]
    tabla = datos

    return render_template(
        "trimestres.html",
        labels=labels,
        valores=valores,
        tabla=tabla
    )



# ============================================================
#  RUTA: Regiones
# ============================================================

@app.route("/regiones")
@login_required
def regiones():
    # 1) Primero agrupamos por departamento
    pipeline = [
        {"$group": {
            "_id": "$DPTO_HECHO_NEW",
            "total": {"$sum": "$cantidad"}
        }}
    ]
    datos = list(col.aggregate(pipeline))

    # 2) Mapa Departamento -> Región (aprox. académico)
    mapa_regiones = {
        # Costa
        "TUMBES": "Costa",
        "PIURA": "Costa",
        "LAMBAYEQUE": "Costa",
        "LA LIBERTAD": "Costa",
        "ANCASH": "Costa",
        "LIMA METROPOLITANA": "Costa",
        "REGION LIMA": "Costa",
        "PROV. CONST. DEL CALLAO": "Costa",
        "ICA": "Costa",
        "AREQUIPA": "Costa",
        "MOQUEGUA": "Costa",
        "TACNA": "Costa",

        # Sierra
        "CAJAMARCA": "Sierra",
        "AMAZONAS": "Sierra",       # puedes moverla a Selva si tu profe lo exige
        "HUANUCO": "Sierra",
        "PASCO": "Sierra",
        "JUNIN": "Sierra",
        "HUANCAVELICA": "Sierra",
        "AYACUCHO": "Sierra",
        "APURIMAC": "Sierra",
        "CUSCO": "Sierra",
        "PUNO": "Sierra",

        # Selva
        "LORETO": "Selva",
        "SAN MARTIN": "Selva",
        "UCAYALI": "Selva",
        "MADRE DE DIOS": "Selva",
    }

    # 3) Acumulamos totales por región
    acumulado = {}
    for d in datos:
        depto = d["_id"]
        total = d["total"]
        region = mapa_regiones.get(depto, "Sin clasificar")
        acumulado[region] = acumulado.get(region, 0) + total

    # 4) Armamos tabla y listas para el gráfico
    tabla = [
        {"_id": region, "total": total}
        for region, total in acumulado.items()
    ]
    # ordenar de mayor a menor
    tabla = sorted(tabla, key=lambda x: x["total"], reverse=True)

    labels = [fila["_id"] for fila in tabla]
    valores = [fila["total"] for fila in tabla]

    return render_template(
        "regiones.html",
        labels=labels,
        valores=valores,
        tabla=tabla
    )



# ============================================================
#  RUTA: Predicción delitos 2026
# ============================================================

@app.route('/prediccion-2026')
@login_required
def prediccion_2026():
    # USAMOS LA COLECCIÓN GRANDE
    col = db['denuncias']
    
    total, etiquetas, valores, historico, anios = predecir_total_2026(col)
    
    return render_template('prediccion_2026.html', 
                           total=total, 
                           etiquetas=etiquetas, 
                           valores=valores)

# ============================================================
#  CHAT IA: Conversar con Gemini + Dataset real
# ============================================================

@app.route("/chat-ia", methods=["GET", "POST"])
@login_required
def chat_ia():
    if "chat_historial" not in session:
        session["chat_historial"] = []
    historial = session["chat_historial"]

    if request.method == "POST":
        mensaje = request.form.get("mensaje", "").strip()

        if mensaje:

            # 🚀 Nuevo motor de IA
            contexto = construir_contexto(col, mensaje)

            respuesta = preguntar_gemini(mensaje, contexto=contexto)

            historial.append({"rol": "usuario", "texto": mensaje})
            historial.append({"rol": "ia", "texto": respuesta})

            session["chat_historial"] = historial

    return render_template("chat_ia.html", historial=historial)

# ============================================================
#  Clustering de departamentos (KMeans, groupby, numpy, matplotlib)
# ============================================================

@app.route('/cluster-departamentos')
@login_required
def cluster_departamentos():
    # 1. Obtener datos desde MongoDB
    pipeline = [
        {
            "$group": {
                "_id": "$DPTO_HECHO_NEW",
                "total": {"$sum": "$cantidad"} 
            }
        },
        {"$sort": {"total": 1}}
    ]
    data_bd = list(col.aggregate(pipeline))
    
    # 2. Limpieza
    data_clean = [d for d in data_bd if d["_id"] is not None and str(d["_id"]).strip() != ""]
    
    if len(data_clean) < 3:
        return "No hay suficientes datos para generar clusters."

    # 3. K-Means
    nombres = [d["_id"] for d in data_clean]
    valores = np.array([d["total"] for d in data_clean]).reshape(-1, 1)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(valores)

    # 4. Ordenar Clusters (Semaforización)
    centroides = kmeans.cluster_centers_.flatten()
    indices_ordenados = np.argsort(centroides) 
    mapa_orden = {old_idx: new_idx for new_idx, old_idx in enumerate(indices_ordenados)}
    
    resultados = []
    colores = ['#28a745', '#ffc107', '#dc3545'] # Verde, Amarillo, Rojo
    etiquetas = ['Riesgo Bajo', 'Riesgo Medio', 'Riesgo Crítico']

    for i, nombre in enumerate(nombres):
        label_original = labels[i]
        label_ordenado = mapa_orden[label_original] 
        
        resultados.append({
            "departamento": nombre,
            "total": int(valores[i][0]),
            "cluster": int(label_ordenado),
            "color": colores[label_ordenado],
            "etiqueta": etiquetas[label_ordenado]
        })

    stats = {
        "bajo": sum(1 for r in resultados if r['cluster'] == 0),
        "medio": sum(1 for r in resultados if r['cluster'] == 1),
        "alto": sum(1 for r in resultados if r['cluster'] == 2)
    }

    # --- AQUÍ ESTÁ EL CAMBIO CLAVE ---
    return render_template('cluster_departamentos.html', data=resultados, stats=stats)

# ============================================================
# riesgo-modalidad
# ============================================================

@app.route("/riesgo-modalidad", methods=["GET", "POST"])
@login_required
def riesgo_modalidad():
    modalidad = "Extorsión"  # puedes cambiarlo o parametrizarlo luego

    modelo, df_hist = entrenar_modelo_riesgo(col, modalidad_objetivo=modalidad)

    prediccion = None
    entrada = None

    # Valores por defecto
    anios_disp = []
    trimestres_disp = []
    dptos = []
    labels = []
    valores = []

    if modelo is not None and df_hist is not None and not df_hist.empty:
        # Listas para el formulario
        anios_disp = sorted(df_hist["anio"].unique().tolist())
        trimestres_disp = ["T1", "T2", "T3", "T4"]
        dptos = sorted(df_hist["departamento"].unique().tolist())

        # Serie histórica trimestral nacional (para el gráfico)
        df_hist_total = (
            df_hist.groupby(["anio", "trimestre"], as_index=False)["total"].sum()
        )
        df_hist_total["label"] = (
            df_hist_total["anio"].astype(str) + "-" + df_hist_total["trimestre"]
        )
        labels = df_hist_total["label"].tolist()
        valores = df_hist_total["total"].tolist()

        if request.method == "POST":
            anio_sel = int(request.form.get("anio"))
            tri_sel = request.form.get("trimestre")
            dpto_sel = request.form.get("departamento")

            mapa_tri = {"T1": 1, "T2": 2, "T3": 3, "T4": 4}
            tri_num = mapa_tri.get(tri_sel, 1)

            X_new = pd.DataFrame([{
                "anio": anio_sel,
                "tri_num": tri_num
            }])

            y_pred = modelo.predict(X_new)[0]
            prediccion = round(y_pred)

            entrada = {
                "anio": anio_sel,
                "trimestre": tri_sel,
                "departamento": dpto_sel
            }

            # 👉 OPCIONAL: guardar la simulación en MongoDB
            db["simulaciones_riesgo"].insert_one({
                "modalidad": modalidad,
                "anio": anio_sel,
                "trimestre": tri_sel,
                "departamento": dpto_sel,
                "prediccion": prediccion,
                "creado_en": datetime.utcnow()
            })

    return render_template(
        "riesgo_modalidad.html",
        modalidad=modalidad,
        prediccion=prediccion,
        entrada=entrada,
        labels=labels,
        valores=valores,
        anios_disp=anios_disp,
        trimestres_disp=trimestres_disp,
        dptos=dptos
    )

# ============================================================
# Análisis Trimestral (Extorsión vs Homicidio)
# ============================================================
# --- AGREGAR ESTO EN TU APP.PY ---

@app.route('/comparativa-foco')
@login_required
def comparativa_foco():
    # 1. Definir los filtros exactos del PDF
    deptos_clave = [
        "LIMA METROPOLITANA", "AREQUIPA", "AYACUCHO", 
        "LA LIBERTAD", "LAMBAYEQUE"
    ]
    modalidades_clave = ["Extorsión", "Homicidio"]
    
    # 2. Pipeline de Agregación
    pipeline = [
        {
            "$match": {
                "DPTO_HECHO_NEW": {"$in": deptos_clave},
                "P_MODALIDADES": {"$in": modalidades_clave}
            }
        },
        {
            "$group": {
                "_id": {
                    "trimestre": "$trimestre",  # Asegúrate que tu ETL creó este campo (T1, T2...)
                    "modalidad": "$P_MODALIDADES"
                },
                "total": {"$sum": "$cantidad"}
            }
        }
    ]
    
    resultados = list(col.aggregate(pipeline))
    
    # 3. Estructurar datos para fácil uso en Chart.js
    # Inicializamos en 0 por si algún trimestre no tiene datos
    datos_chart = {
        "Extorsión": {"T1": 0, "T2": 0, "T3": 0, "T4": 0},
        "Homicidio": {"T1": 0, "T2": 0, "T3": 0, "T4": 0}
    }
    
    for r in resultados:
        tri = r["_id"].get("trimestre")
        mod = r["_id"].get("modalidad")
        cant = r["total"]
        
        # Solo procesamos si el trimestre es válido (T1-T4) y la modalidad es una de las 2
        if tri in ["T1", "T2", "T3", "T4"] and mod in datos_chart:
            datos_chart[mod][tri] = cant

    return render_template('comparativa_foco.html', datos=datos_chart)

# ============================================================
# reporte_lima
# ============================================================
@app.route('/reporte-lima')
@login_required
def reporte_lima():
    pipeline = [
        {
            "$match": {
                # 1. FILTRO GEOGRÁFICO:
                # Buscamos "LIMA" en el campo que vimos en la foto (DPTO_HECHO_NEW)
                "DPTO_HECHO_NEW": {"$regex": "LIMA", "$options": "i"},
                
                # 2. FILTRO DE MODALIDAD:
                "P_MODALIDADES": {
                    "$in": [
                        re.compile("EXTORSI", re.IGNORECASE), 
                        re.compile("HOMICIDI", re.IGNORECASE)
                    ]
                }
            }
        },
        {
            "$project": {
                # Usamos el nombre exacto que vimos en la foto: "ANIO"
                "anio": "$ANIO", 
                "trimestre": {"$ifNull": ["$trimestre", "T1"]},
                "cantidad_real": "$cantidad", # <--- ¡LA CLAVE DEL ÉXITO!
                
                # Normalizamos el nombre de la modalidad
                "modalidad_norm": {
                    "$cond": {
                        "if": {"$regexMatch": {"input": "$P_MODALIDADES", "regex": "EXTORSI", "options": "i"}},
                        "then": "Extorsión",
                        "else": "Homicidio"
                    }
                }
            }
        },
        {
            "$group": {
                "_id": {
                    "anio": "$anio",
                    "trimestre": "$trimestre",
                    "modalidad": "$modalidad_norm"
                },
                # AQUÍ ESTABA EL ERROR: Antes sumábamos 1, ahora sumamos la cantidad real
                "total": {"$sum": "$cantidad_real"} 
            }
        },
        { "$sort": {"_id.anio": 1, "_id.trimestre": 1} }
    ]

    datos = list(col.aggregate(pipeline))

    # --- AUDITORÍA DE CALIBRACIÓN ---
    print("\n--- 🔍 DIAGNÓSTICO FINAL ---")
    total_ext_2023 = sum(d['total'] for d in datos if d['_id']['anio'] == 2018 and d['_id']['modalidad'] == 'Extorsión')
    # Nota: Puse 2018 porque vi ese año en tu foto, pero sumará todos.
    
    if datos:
        print(f"✅ ¡Datos encontrados! Primer registro procesado: {datos[0]}")
    else:
        print("⚠️ ALERTA: Sigue saliendo vacío. Verifica que 'denuncias_db' esté bien puesto en la conexión.")
    # --------------------------------

    labels = []
    data_extorsion = []
    data_homicidio = []
    temp_data = {}

    for d in datos:
        anio = d["_id"].get("anio")
        tri = d["_id"].get("trimestre")
        mod = d["_id"].get("modalidad")
        total = d["total"]
        
        if not anio: continue

        clave = f"{anio}-{tri}"
        if clave not in labels: labels.append(clave)
        if clave not in temp_data: temp_data[clave] = {"Extorsión": 0, "Homicidio": 0}
        
        temp_data[clave][mod] = total

    labels.sort()
    for l in labels:
        data_extorsion.append(temp_data[l]["Extorsión"])
        data_homicidio.append(temp_data[l]["Homicidio"])

    return render_template('reporte_lima.html', labels=labels, extorsion=data_extorsion, homicidio=data_homicidio)

# ============================================================
# Agente "Centinela" de Políticas Públicas
# ============================================================
# --- NUEVO AGENTE: ESTRATEGA DE SEGURIDAD ---
@app.route('/agente-estrategico')
@login_required
def agente_estrategico():
    # USAMOS LA COLECCIÓN GRANDE
    col = db['denuncias']
    
    total_2026, _, _, _, _ = predecir_total_2026(col)
    
    # Formateamos el número con comas para que se vea bien (ej: 1,230,500)
    total_fmt = "{:,}".format(total_2026)
    mensaje = f"Basado en el análisis de millones de registros históricos, la IA proyecta {total_fmt} incidentes para el 2026."
    
    return render_template('agente_estrategico.html', total=total_2026, analisis=mensaje)

# ============================================================
# RUN Agente de Asignación Táctica
# ============================================================
@app.route('/agente-logistico')
@login_required
def agente_logistico():
    # USAMOS LA COLECCIÓN GRANDE
    col = db['denuncias']
    
    datos_clusters = clusterizar_departamentos(col, n_clusters=3)
    
    return render_template('agente_logistico.html', clusters=datos_clusters)

# --- IMPORTANTE: Asegúrate de tener esto arriba del todo ---
# from werkzeug.security import generate_password_hash
# from functools import wraps

# --- Decorador para permitir acceso solo al Admin ---
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if session.get('rol') != 'admin':
            flash('Acceso denegado. Solo administradores.', 'error')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function
# ============================================================
# RUTA PARA CREAR USUARIOS
# ============================================================
# --- RUTA PARA CREAR USUARIOS ---
# --- IMPORTANTE: Asegúrate de tener esto arriba del todo ---
# from werkzeug.security import generate_password_hash
# from functools import wraps

# --- Decorador para permitir acceso solo al Admin ---
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if session.get('rol') != 'admin':
            flash('Acceso denegado. Solo administradores.', 'error')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function

# --- RUTA PARA CREAR USUARIOS ---
@app.route('/crear_usuario', methods=['GET', 'POST'])
@login_required # Obliga a estar logueado
@admin_required # Obliga a ser admin (usando el decorador de arriba)
def crear_usuario():
    if request.method == 'POST':
        # 1. Obtener datos del formulario
        username = request.form['username']
        password = request.form['password']
        rol = request.form['rol']

        # 2. Conectar a BD
        users_col = db['usuarios']

        # 3. Validar si ya existe
        if users_col.find_one({"username": username}):
            flash('El usuario ya existe.', 'error')
        else:
            # 4. Encriptar y Guardar
            hashed_password = generate_password_hash(password)
            users_col.insert_one({
                "username": username,
                "password": hashed_password,
                "rol": rol
            })
            flash(f'Usuario {username} creado exitosamente.', 'success')
            return redirect(url_for('crear_usuario'))

    return render_template('crear_usuario.html')
# ============================================================
# RUN
# ============================================================
@app.route('/espiar-datos')
def espiar_datos():
    try:
        col = db['simulaciones_riesgo']
        # Trae un documento cualquiera
        dato = col.find_one()
        return f"<h1>Lo que hay en la base de datos:</h1><p>{str(dato)}</p>"
    except Exception as e:
        return f"<h1>Error espiando:</h1><p>{str(e)}</p>"



if __name__ == "__main__":
    app.run(debug=True)
