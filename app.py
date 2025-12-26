# ============================================================
# app.py — Proyecto Analítica de Denuncias + Chat IA + Auditoría
# ============================================================
import os
import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from functools import wraps
from flask import (
    Flask, render_template, request, redirect, jsonify,
    url_for, session, flash
)
from flask_login import LoginManager, login_user, logout_user, login_required, current_user, UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
from pymongo import MongoClient
from sklearn.cluster import KMeans

# ---- Módulos del Proyecto ----
from ml_utils import (
    db, predecir_total_2026, obtener_contexto_ia, 
    entrenar_modelo_riesgo, predecir_valor_especifico
)
from mongo_queries import ranking_departamentos
from ml_cluster import clusterizar_departamentos
from ml_llm import (
    consultar_estratega_ia, analizar_riesgo_ia, consultar_chat_general
)

# Cargar variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "secreto_seguro")

# ================================
#  CONFIGURACIÓN FLASK-LOGIN
# ================================
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Modelo de Usuario para Flask-Login
class User(UserMixin):
    def __init__(self, user_data):
        self.id = user_data['username'] # Flask-Login usa 'id' como identificador
        self.username = user_data['username']
        self.rol = user_data.get('rol', 'invitado')

@login_manager.user_loader
def load_user(user_id):
    users_col = db['usuarios']
    user_data = users_col.find_one({"username": user_id})
    if user_data:
        return User(user_data)
    return None

# ================================
#  CONEXIÓN A MONGO (Backup si ml_utils falla)
# ================================
if db is None:
    print("⚠️ Advertencia: db no importada de ml_utils, intentando conexión local...")
    mongo_uri = os.environ.get('MONGO_URI', "mongodb://localhost:27017/")
    client = MongoClient(mongo_uri)
    db = client['denuncias_db']

col = db['denuncias']

# ============================================================
#  DECORADORES PERSONALIZADOS
# ============================================================
# Nota: login_required ya viene de flask_login, pero mantenemos admin_required
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Verifica rol usando current_user de Flask-Login o session
        rol = session.get('rol')
        if not rol and current_user.is_authenticated:
            rol = current_user.rol
            
        if rol != 'admin':
            flash('⛔ Acceso denegado: Se requieren privilegios de Administrador.', 'error')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function

# ============================================================
#  RADAR DE USUARIOS ONLINE (AUDITORÍA EN TIEMPO REAL)
# ============================================================
@app.before_request
def update_last_seen():
    if current_user.is_authenticated:
        try:
            # Actualizamos 'last_seen' cada vez que el usuario hace algo
            db['usuarios'].update_one(
                {"username": current_user.username},
                {"$set": {"last_seen": datetime.now()}}
            )
        except Exception as e:
            print(f"Error actualizando last_seen: {e}")

# ============================================================
#  LOGIN / LOGOUT
# ============================================================
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        users_col = db['usuarios']
        user_data = users_col.find_one({"username": username})

        if user_data and check_password_hash(user_data['password'], password):
            # Login exitoso
            user = User(user_data)
            login_user(user)
            
            # Guardar en sesión también por compatibilidad
            session['user'] = username
            session['rol'] = user_data.get('rol', 'invitado')
            session['logged_in'] = True # Bandera simple

            # --- AUDITORÍA: GUARDAR LOG DE ENTRADA ---
            try:
                db['auditoria'].insert_one({
                    "usuario": username,
                    "evento": "Inicio de Sesión",
                    "fecha": datetime.now(),
                    "fecha_str": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "ip": request.remote_addr
                })
            except Exception as e:
                print(f"Error guardando auditoría: {e}")
            # -----------------------------------------

            return redirect(url_for('index'))
        else:
            flash('Usuario o contraseña incorrectos', 'error')
            return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    session.clear()
    flash('Has cerrado sesión correctamente.', 'success')
    return redirect(url_for('login'))

# ============================================================
#  PÁGINA PRINCIPAL (DASHBOARD)
# ============================================================
@app.route("/")
@login_required
def index():
    visitas = 0
    try:
        stats_col = db['estadisticas']
        stats_col.update_one(
            {'_id': 'contador_home'},
            {'$inc': {'cantidad': 1}},
            upsert=True
        )
        dato_visitas = stats_col.find_one({'_id': 'contador_home'})
        if dato_visitas:
            visitas = dato_visitas.get('cantidad', 0)
    except Exception as e:
        print(f"⚠️ Error en contador de visitas: {e}")

    total_registros = col.count_documents({})
    
    pipeline = [{"$group": {"_id": None, "total": {"$sum": "$cantidad"}}}]
    res = list(col.aggregate(pipeline))
    total_denuncias = res[0]["total"] if res else 0

    anios = sorted(col.distinct("ANIO"))

    return render_template(
        "index.html",
        total_registros=total_registros,
        total_denuncias=total_denuncias,
        anios=anios,
        visitas=visitas
    )

# ============================================================
#  MÓDULO: AUDITORÍA Y USUARIOS ACTIVOS
# ============================================================
@app.route('/usuarios-activos')
@login_required
def usuarios_activos():
    try:
        # 1. Usuarios Online (Activos en los últimos 5 min)
        limite_tiempo = datetime.now() - timedelta(minutes=5)
        users_col = db['usuarios']
        usuarios_online = list(users_col.find({"last_seen": {"$gt": limite_tiempo}}))
        
        # 2. Historial de Auditoría (Últimos 50 eventos)
        auditoria_col = db['auditoria']
        historial = list(auditoria_col.find().sort("fecha", -1).limit(50))
        
        return render_template('usuarios_activos.html', online=usuarios_online, historial=historial)
        
    except Exception as e:
        print(f"Error en reporte usuarios: {e}")
        return render_template('usuarios_activos.html', online=[], historial=[])

# ============================================================
#  RUTAS DE REPORTES BÁSICOS
# ============================================================
@app.route("/resumen-anual")
@login_required
def resumen_anual():
    pipeline = [
        {"$group": {"_id": "$ANIO", "total": {"$sum": "$cantidad"}}},
        {"$sort": {"_id": 1}}
    ]
    datos = list(col.aggregate(pipeline))
    labels = [doc["_id"] for doc in datos]
    valores = [doc["total"] for doc in datos]
    return render_template("resumen_anual.html", labels=labels, valores=valores, tabla=datos)

@app.route('/departamentos')
@login_required
def departamentos():
    # Traemos TODOS los departamentos, no solo 30
    data_raw = ranking_departamentos(col, n=50) 
    
    # DICCIONARIO MAESTRO: Base de datos -> Código Highcharts (pe-xx)
    map_mapping = {
        "AMAZONAS": "pe-am",
        "ANCASH": "pe-an",
        "APURIMAC": "pe-ap",
        "AREQUIPA": "pe-ar",
        "AYACUCHO": "pe-ay",
        "CAJAMARCA": "pe-cj",
        "CALLAO": "pe-cl",
        "PROV. CONST. DEL CALLAO": "pe-cl",  # Caso especial Callao
        "CUSCO": "pe-cs",
        "HUANCAVELICA": "pe-hv",
        "HUANUCO": "pe-hc",
        "ICA": "pe-ic",
        "JUNIN": "pe-ju",
        "LA LIBERTAD": "pe-ll",
        "LAMBAYEQUE": "pe-lb",
        "LIMA METROPOLITANA": "pe-li",       # Lima Metro
        "REGION LIMA": "pe-lr",              # Lima Provincias
        "LORETO": "pe-lo",
        "MADRE DE DIOS": "pe-md",
        "MOQUEGUA": "pe-mq",
        "PASCO": "pe-pa",
        "PIURA": "pe-pi",
        "PUNO": "pe-pu",
        "SAN MARTIN": "pe-sm",
        "TACNA": "pe-ta",
        "TUMBES": "pe-tu",
        "UCAYALI": "pe-uc"
    }

    data_mapa = []
    # Top 5 para la tabla lateral
    top_5 = data_raw[:5]

    for d in data_raw:
        # Limpieza CRÍTICA: Convertir a mayúsculas y quitar espacios extra
        nombre_bd = str(d["_id"]).strip().upper()
        
        # 1. Búsqueda exacta
        code = map_mapping.get(nombre_bd)
        
        # 2. Búsqueda de respaldo (Por si acaso)
        if not code:
            if "LIMA" in nombre_bd: code = "pe-li"
            elif "CALLAO" in nombre_bd: code = "pe-cl"
            
        if code:
            data_mapa.append([code, d["total"]])
        else:
            # Esto imprimirá en la consola de Render qué departamento está fallando
            print(f"⚠️ No mapeado: {nombre_bd}")

    return render_template('departamentos.html', data_mapa=data_mapa, top_5=top_5)

@app.route("/departamentos-percapita")
@login_required
def departamentos_percapita():
    pipeline = [
        {"$group": {"_id": "$DPTO_HECHO_NEW", "total": {"$sum": "$cantidad"}}},
        {"$sort": {"_id": 1}}
    ]
    datos = list(col.aggregate(pipeline))
    
    # Poblaciones aproximadas (INEI)
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
        # ... completa las demás
    }

    tabla = []
    for d in datos:
        depto = d["_id"]
        total = d["total"]
        pob = poblaciones.get(depto, 1000000) # Default para evitar error div/0
        tasa = (total / pob) * 100000
        tabla.append({"departamento": depto, "total": total, "poblacion": pob, "tasa": tasa})

    tabla = sorted(tabla, key=lambda x: x["tasa"], reverse=True)
    labels = [r["departamento"] for r in tabla]
    valores = [r["tasa"] for r in tabla]

    return render_template("departamentos_percapita.html", labels=labels, valores=valores, tabla=tabla)

@app.route("/regiones")
@login_required
def regiones():
    pipeline = [{"$group": {"_id": "$DPTO_HECHO_NEW", "total": {"$sum": "$cantidad"}}}]
    datos = list(col.aggregate(pipeline))
    
    # DICCIONARIO EXACTO (Copiado de tus datos reales)
    mapa_regiones = {
        # COSTA
        "TUMBES": "Costa", 
        "PIURA": "Costa", 
        "LAMBAYEQUE": "Costa", 
        "LA LIBERTAD": "Costa", 
        "ANCASH": "Costa", 
        "LIMA METROPOLITANA": "Costa",      # <--- ¡AQUÍ ESTABA EL FANTASMA!
        "REGION LIMA": "Costa",             # <--- Y AQUÍ
        "PROV. CONST. DEL CALLAO": "Costa", 
        "CALLAO": "Costa",
        "ICA": "Costa", 
        "AREQUIPA": "Costa", 
        "MOQUEGUA": "Costa", 
        "TACNA": "Costa",

        # SIERRA
        "CAJAMARCA": "Sierra", 
        "HUANUCO": "Sierra", 
        "PASCO": "Sierra", 
        "JUNIN": "Sierra", 
        "HUANCAVELICA": "Sierra", 
        "AYACUCHO": "Sierra", 
        "APURIMAC": "Sierra", 
        "CUSCO": "Sierra", 
        "PUNO": "Sierra",

        # SELVA
        "AMAZONAS": "Selva", 
        "LORETO": "Selva", 
        "SAN MARTIN": "Selva", 
        "UCAYALI": "Selva", 
        "MADRE DE DIOS": "Selva"
    }
    
    acumulado = {"Costa": 0, "Sierra": 0, "Selva": 0}
    
    for d in datos:
        nombre_bd = str(d["_id"]).strip().upper()
        
        # Clasificación directa
        region = mapa_regiones.get(nombre_bd)
        
        if region:
            acumulado[region] += d["total"]
        else:
            # Si falla, miramos en los logs (Render) qué nombre raro apareció
            print(f"⚠️ DATO SIN REGIÓN: '{nombre_bd}'") 
            
            # Fallback de seguridad para no perder datos (Asumimos Costa para Lima/Callao raros)
            if "LIMA" in nombre_bd or "CALLAO" in nombre_bd:
                acumulado["Costa"] += d["total"]

    tabla = [
        {"_id": "Costa", "total": acumulado["Costa"]},
        {"_id": "Sierra", "total": acumulado["Sierra"]},
        {"_id": "Selva", "total": acumulado["Selva"]}
    ]
    
    tabla = sorted(tabla, key=lambda x: x["total"], reverse=True)
    
    return render_template("regiones.html", 
                           labels=[x["_id"] for x in tabla], 
                           valores=[x["total"] for x in tabla], 
                           tabla=tabla)

# ============================================================
#  RUTAS DE ANÁLISIS TÉCNICO
# ============================================================
@app.route("/modalidades")
@login_required
def modalidades():
    pipeline = [
        {"$group": {"_id": "$P_MODALIDADES", "total": {"$sum": "$cantidad"}}},
        {"$sort": {"total": -1}}
    ]
    datos = list(col.aggregate(pipeline))
    return render_template("modalidades.html", labels=[d["_id"] for d in datos], valores=[d["total"] for d in datos], tabla=datos)

@app.route("/trimestres")
@login_required
def trimestres():
    pipeline = [
        {"$group": {"_id": "$anio_trimestre", "total": {"$sum": "$cantidad"}}},
        {"$sort": {"_id": 1}}
    ]
    datos = list(col.aggregate(pipeline))
    return render_template("trimestres.html", labels=[d["_id"] for d in datos], valores=[d["total"] for d in datos], tabla=datos)

@app.route('/departamentos')
@login_required
def departamentos():
    data_raw = ranking_departamentos(col, n=50) 
    
    # DICCIONARIO MAESTRO MAPA (Highcharts codes)
    map_mapping = {
        # --- LAS LIMAS (CLAVE DEL PROBLEMA) ---
        "LIMA METROPOLITANA": "pe-li",       # El código 'pe-li' es la zona principal (Azul Oscuro)
        "REGION LIMA": "pe-lr",              # El código 'pe-lr' es la región externa
        "LIMA": "pe-li",                     # Por si acaso
        
        # --- EL CALLAO ---
        "PROV. CONST. DEL CALLAO": "pe-cl",
        "CALLAO": "pe-cl",

        # --- EL RESTO DEL PERÚ ---
        "AMAZONAS": "pe-am",
        "ANCASH": "pe-an",
        "APURIMAC": "pe-ap",
        "AREQUIPA": "pe-ar",
        "AYACUCHO": "pe-ay",
        "CAJAMARCA": "pe-cj",
        "CUSCO": "pe-cs",
        "HUANCAVELICA": "pe-hv",
        "HUANUCO": "pe-hc",
        "ICA": "pe-ic",
        "JUNIN": "pe-ju",
        "LA LIBERTAD": "pe-ll",
        "LAMBAYEQUE": "pe-lb",
        "LORETO": "pe-lo",
        "MADRE DE DIOS": "pe-md",
        "MOQUEGUA": "pe-mq",
        "PASCO": "pe-pa",
        "PIURA": "pe-pi",
        "PUNO": "pe-pu",
        "SAN MARTIN": "pe-sm",
        "TACNA": "pe-ta",
        "TUMBES": "pe-tu",
        "UCAYALI": "pe-uc"
    }

    data_mapa = []
    top_5 = data_raw[:5]

    for d in data_raw:
        nombre_bd = str(d["_id"]).strip().upper()
        
        # Mapeo exacto
        code = map_mapping.get(nombre_bd)
        
        if code:
            data_mapa.append([code, d["total"]])
        else:
            print(f"⚠️ NO MAPEADO EN MAPA: {nombre_bd}")

    return render_template('departamentos.html', data_mapa=data_mapa, top_5=top_5)

@app.route('/comparativa-foco')
@login_required
def comparativa_foco():
    pipeline = [
        {"$match": {"P_MODALIDADES": {"$in": ["Extorsión", "Homicidio"]}}},
        {"$group": {"_id": {"trimestre": "$trimestre", "mod": "$P_MODALIDADES"}, "total": {"$sum": "$cantidad"}}}
    ]
    res = list(col.aggregate(pipeline))
    
    chart_data = {"Extorsión": {}, "Homicidio": {}}
    for r in res:
        mod = r["_id"].get("mod")
        tri = r["_id"].get("trimestre")
        if mod in chart_data and tri:
            chart_data[mod][tri] = r["total"]
            
    return render_template('comparativa_foco.html', datos=chart_data)

@app.route('/reporte-lima')
@login_required
def reporte_lima():
    pipeline = [
        {
            "$match": {
                "DPTO_HECHO_NEW": {"$regex": "LIMA", "$options": "i"},
                "P_MODALIDADES": {"$in": [re.compile("EXTORSI", re.I), re.compile("HOMICIDI", re.I)]}
            }
        },
        {
            "$project": {
                "anio": "$ANIO", "trimestre": {"$ifNull": ["$trimestre", "T1"]}, "cantidad": "$cantidad",
                "modalidad": {"$cond": [{"$regexMatch": {"input": "$P_MODALIDADES", "regex": "EXTORSI", "options": "i"}}, "Extorsión", "Homicidio"]}
            }
        },
        {"$group": {"_id": {"anio": "$anio", "trim": "$trimestre", "mod": "$modalidad"}, "total": {"$sum": "$cantidad"}}},
        {"$sort": {"_id.anio": 1, "_id.trim": 1}}
    ]
    datos = list(col.aggregate(pipeline))
    
    labels = sorted(list(set(f"{d['_id']['anio']}-{d['_id']['trim']}" for d in datos)))
    data_ext = []
    data_hom = []
    
    temp = {l: {"Extorsión": 0, "Homicidio": 0} for l in labels}
    for d in datos:
        key = f"{d['_id']['anio']}-{d['_id']['trim']}"
        temp[key][d['_id']['mod']] = d['total']
        
    for l in labels:
        data_ext.append(temp[l]["Extorsión"])
        data_hom.append(temp[l]["Homicidio"])
        
    return render_template('reporte_lima.html', labels=labels, extorsion=data_ext, homicidio=data_hom)

# ============================================================
#  MÓDULOS DE INTELIGENCIA ARTIFICIAL
# ============================================================
@app.route('/prediccion-2026')
@login_required
def prediccion_2026():
    total, etiquetas, valores, _, _ = predecir_total_2026(col)
    return render_template('prediccion_2026.html', total=total, etiquetas=etiquetas, valores=valores)

@app.route('/agente-estrategico')
@login_required
def agente_estrategico():
    try:
        total_2026, texto_historico = obtener_contexto_ia(col)
        analisis_ia = consultar_estratega_ia(total_2026, texto_historico, "Tendencia Extorsión/Homicidio")
        return render_template('agente_estrategico.html', total="{:,}".format(total_2026), analisis=analisis_ia)
    except:
        return render_template('agente_estrategico.html', total="Calculando...", analisis="IA Reiniciando...")

@app.route('/riesgo-modalidad', methods=['GET', 'POST'])
@login_required
def riesgo_modalidad():
    modalidad = "Extorsión"
    anio_sim, trim_sim, dpto_sim = 2025, "T1", "LIMA METROPOLITANA"
    resultado_sim, analisis_ia_txt = 0, None
    
    # 1. INICIALIZAR VARIABLES DE GRÁFICO (¡Esto faltaba!)
    grafico_labels = []
    grafico_data = []
    deptos_list = []
    anios_list = []

    # 2. Entrenar modelo y obtener datos históricos
    modelo, df_hist, le_dpto = entrenar_modelo_riesgo(col, modalidad)
    
    # 3. Si hay datos históricos, llenamos las listas del gráfico
    if not df_hist.empty:
        # Agrupamos datos para el gráfico de línea (Tendencia Nacional)
        df_nacional = df_hist.groupby(['periodo', 'anio', 'trimestre_num'])['total'].sum().reset_index()
        df_nacional = df_nacional.sort_values(by=['anio', 'trimestre_num'])
        
        grafico_labels = df_nacional['periodo'].tolist()
        grafico_data = df_nacional['total'].tolist()
        
        deptos_list = sorted(df_hist['departamento'].unique().tolist())
        anios_list = sorted(df_hist['anio'].unique().tolist())
        if 2026 not in anios_list: anios_list.append(2026)
    
    # 4. Procesar Simulación (POST)
    if request.method == 'POST' and modelo:
        try:
            anio_sim = int(request.form.get('anio'))
            trim_sim = request.form.get('trimestre')
            dpto_sim = request.form.get('departamento')
            
            resultado_sim = predecir_valor_especifico(modelo, le_dpto, anio_sim, trim_sim, dpto_sim)
            
            # Solo llamamos a la IA si Gemini no está bloqueado (try/catch interno)
            analisis_ia_txt = analizar_riesgo_ia(resultado_sim, modalidad, dpto_sim, trim_sim, anio_sim)
        except Exception as e:
            print(f"Error en simulación POST: {e}")
        
    return render_template('riesgo_modalidad.html', 
                           departamentos=deptos_list,
                           anios_disp=anios_list,
                           trimestres_disp=["T1", "T2", "T3", "T4"],
                           entrada={"anio": anio_sim, "trimestre": trim_sim, "departamento": dpto_sim},
                           resultado=resultado_sim, 
                           analisis_ia=analisis_ia_txt,
                           # PASAMOS LAS VARIABLES DE GRÁFICO SEGURAS
                           labels=grafico_labels, 
                           valores=grafico_data)

# ============================================================
# CHATBOT IA: CONTEXTO INTELIGENTE (MEJORADO)
# ============================================================
@app.route('/chat-ia', methods=['POST'])
@login_required
def chat_ia():
    mensaje = request.form.get('mensaje', '').strip()
    if not mensaje: return jsonify({'respuesta': "No entendí."})
    
    try:
        col = db['denuncias']
        contexto_acumulado = "ERES UN ANALISTA DE INTELIGENCIA POLICIAL (SIDPOL).\n"
        
        # -----------------------------------------------------
        # 1. CONTEXTO GENERAL (Siempre se envía)
        # -----------------------------------------------------
        # A. Histórico Anual
        pipeline_anual = [{"$group": {"_id": "$ANIO", "total": {"$sum": "$cantidad"}}}, {"$sort": {"_id": 1}}]
        datos_anual = list(col.aggregate(pipeline_anual))
        txt_anual = ", ".join([f"{d['_id']}: {d['total']:,}" for d in datos_anual if str(d['_id']).isdigit()])
        contexto_acumulado += f"HISTORIAL NACIONAL POR AÑO: {txt_anual}.\n"

        # B. Top 5 Modalidades (Nacional) - Para que sepa de qué delitos hablamos
        pipeline_mod = [
            {"$group": {"_id": "$P_MODALIDADES", "total": {"$sum": "$cantidad"}}},
            {"$sort": {"total": -1}},
            {"$limit": 5}
        ]
        datos_mod = list(col.aggregate(pipeline_mod))
        txt_mod = ", ".join([f"{d['_id']} ({d['total']:,})" for d in datos_mod])
        contexto_acumulado += f"TOP 5 DELITOS (NACIONAL): {txt_mod}.\n"

        # -----------------------------------------------------
        # 2. DETECCIÓN INTELIGENTE (¿El usuario habla de un lugar?)
        # -----------------------------------------------------
        deptos_clave = [
            "AMAZONAS", "ANCASH", "APURIMAC", "AREQUIPA", "AYACUCHO", "CAJAMARCA", 
            "CALLAO", "CUSCO", "HUANCAVELICA", "HUANUCO", "ICA", "JUNIN", "LA LIBERTAD", 
            "LAMBAYEQUE", "LIMA", "LORETO", "MADRE DE DIOS", "MOQUEGUA", "PASCO", 
            "PIURA", "PUNO", "SAN MARTIN", "TACNA", "TUMBES", "UCAYALI"
        ]
        
        mensaje_upper = mensaje.upper()
        depto_detectado = None
        
        # Buscamos si el mensaje contiene algún departamento
        for d in deptos_clave:
            if d in mensaje_upper:
                depto_detectado = d
                break 
        
        # SI ENCONTRAMOS UN LUGAR, BUSCAMOS SU DATA ESPECÍFICA
        if depto_detectado:
            # Consulta a MongoDB filtrando por ese departamento
            pipeline_local = [
                # Usamos regex para que "LIMA" coincida con "LIMA METROPOLITANA" o "REGION LIMA"
                {"$match": {"DPTO_HECHO_NEW": {"$regex": depto_detectado, "$options": "i"}}},
                {"$group": {"_id": "$P_MODALIDADES", "total": {"$sum": "$cantidad"}}},
                {"$sort": {"total": -1}},
                {"$limit": 3} # Traemos los 3 delitos más comunes de esa zona
            ]
            datos_local = list(col.aggregate(pipeline_local))
            
            if datos_local:
                txt_local = ", ".join([f"{d['_id']} ({d['total']:,})" for d in datos_local])
                contexto_acumulado += f"\n--- DATOS ESPECÍFICOS PARA '{depto_detectado}' ---\n"
                contexto_acumulado += f"Principales delitos en esta región: {txt_local}.\n"
            else:
                contexto_acumulado += f"\n(Nota: No se encontraron datos específicos para {depto_detectado} en la BD).\n"

        # -----------------------------------------------------
        # 3. CONSULTA A GEMINI (Con toda la info nueva)
        # -----------------------------------------------------
        # Se envía el mensaje del usuario + el contexto enriquecido
        respuesta = consultar_chat_general(mensaje, contexto_datos=contexto_acumulado)
        return jsonify({'respuesta': respuesta})

    except Exception as e:
        print(f"Error chat avanzado: {e}")
        return jsonify({'respuesta': "Ocurrió un error consultando la base de datos."})

# ============================================================
#  ADMINISTRACIÓN DE USUARIOS
# ============================================================
@app.route('/crear_usuario', methods=['GET', 'POST'])
@login_required
@admin_required
def crear_usuario():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        rol = request.form['rol']
        
        users_col = db['usuarios']
        if users_col.find_one({"username": username}):
            flash('Usuario ya existe.', 'error')
        else:
            users_col.insert_one({
                "username": username,
                "password": generate_password_hash(password),
                "rol": rol
            })
            flash(f'Usuario {username} creado.', 'success')
            return redirect(url_for('crear_usuario'))
            
    return render_template('crear_usuario.html')

if __name__ == "__main__":
    app.run(debug=True)