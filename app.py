# ============================================================
# app.py — Proyecto Analítica de Denuncias + Chat IA + Auditoría
# ============================================================
import os
import re
from urllib.parse import urlparse
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
from werkzeug.middleware.proxy_fix import ProxyFix
from flask_wtf.csrf import CSRFProtect, CSRFError
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from dotenv import load_dotenv
from pymongo import MongoClient
from sklearn.cluster import KMeans

# ---- Módulos del Proyecto ----
from markupsafe import Markup, escape
from ml_utils import (
    db, predecir_total_2026, proyeccion_2026, obtener_contexto_ia,
    entrenar_modelo_riesgo, predecir_valor_especifico
)
import analitica
from mongo_queries import ranking_departamentos
from ml_cluster import clusterizar_departamentos
from ml_llm import (
    consultar_estratega_ia, analizar_riesgo_ia, consultar_chat_general,
    MENSAJE_CUOTA_GOOGLE, MENSAJE_ERROR, MENSAJE_LIMITE
)

# Cargar variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ["SECRET_KEY"]

# ============================================================
#  SEGURIDAD (OWASP Top 10)
# ============================================================
EN_PRODUCCION = bool(os.getenv("RENDER"))  # Render define RENDER=true

# IP real del visitante detrás del proxy de Render (para auditoría y rate limiting)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

# Cookies de sesión endurecidas (A07: fallas de identificación y autenticación)
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=EN_PRODUCCION,       # solo HTTPS en producción
    REMEMBER_COOKIE_HTTPONLY=True,
    REMEMBER_COOKIE_SECURE=EN_PRODUCCION,
    PERMANENT_SESSION_LIFETIME=timedelta(hours=8),
    WTF_CSRF_TIME_LIMIT=None,                  # el token dura lo que dure la sesión
)

# Protección CSRF en todos los formularios y peticiones POST (A01)
csrf = CSRFProtect(app)

# Límite de peticiones (A07: fuerza bruta / abuso)
limiter = Limiter(get_remote_address, app=app, storage_uri="memory://", default_limits=[])

# Cabeceras de seguridad HTTP (A05: configuración de seguridad incorrecta)
CSP = "; ".join([
    "default-src 'self'",
    "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com "
    "https://code.jquery.com",
    "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com https://fonts.googleapis.com",
    "font-src 'self' https://fonts.gstatic.com https://cdnjs.cloudflare.com https://cdn.jsdelivr.net data:",
    "img-src 'self' data: blob:",
    "connect-src 'self'",
    "object-src 'none'",
    "base-uri 'self'",
    "form-action 'self'",
    "frame-ancestors 'none'",
])


@app.after_request
def cabeceras_seguridad(resp):
    resp.headers["Content-Security-Policy"] = CSP
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    resp.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    if EN_PRODUCCION:
        resp.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return resp


@app.template_filter("formato_ia")
def formato_ia(texto):
    """Da formato básico (negritas, viñetas, saltos de línea) al texto de la IA.
    Primero escapa todo el HTML, así una respuesta maliciosa no puede inyectar código (XSS)."""
    t = str(escape(texto or ""))
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"(?m)^\s*[\*\-]\s+", "• ", t)
    return Markup(t.replace("\n", "<br>"))


@app.errorhandler(CSRFError)
def error_csrf(e):
    if request.path == "/chat-ia":
        return jsonify({"respuesta": "Tu sesión expiró. Recarga la página e intenta de nuevo."}), 400
    flash("La sesión del formulario expiró. Intenta de nuevo.", "error")
    return redirect(_url_segura(request.referrer) or url_for("login"))


def _url_segura(url):
    """Devuelve la ruta solo si apunta a este mismo sitio (evita redirecciones abiertas)."""
    if not url:
        return None
    destino = urlparse(url)
    if destino.netloc and destino.netloc != request.host:
        return None
    if destino.scheme and destino.scheme not in ("http", "https"):
        return None
    ruta = destino.path or "/"
    if not ruta.startswith("/") or ruta.startswith("//"):
        return None
    return ruta + (f"?{destino.query}" if destino.query else "")


@app.errorhandler(429)
def demasiadas_peticiones(e):
    if request.path == "/chat-ia":
        return jsonify({"respuesta": "Demasiadas consultas seguidas. Espera un minuto."}), 429
    flash("Demasiados intentos. Espera un minuto antes de volver a intentar.", "error")
    return redirect(url_for("login"))

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
@limiter.limit("5 per minute; 30 per hour", methods=["POST"])
def login():
    if request.method == 'POST':
        username = str(request.form.get('username', '')).strip()
        password = str(request.form.get('password', ''))

        users_col = db['usuarios']
        user_data = users_col.find_one({"username": username})

        if user_data and check_password_hash(user_data['password'], password):
            # Login exitoso: sesión nueva para evitar fijación de sesión
            session.clear()
            session.permanent = True
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
            # --- AUDITORÍA: intento fallido (útil para detectar fuerza bruta) ---
            try:
                db['auditoria'].insert_one({
                    "usuario": username[:50],
                    "evento": "Intento fallido",
                    "fecha": datetime.now(),
                    "fecha_str": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "ip": request.remote_addr
                })
            except Exception as e:
                print(f"Error guardando auditoría: {e}")
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

    try:
        proyeccion = proyeccion_2026(col)
    except Exception as e:
        print(f"Proyección no disponible en el dashboard: {e}")
        proyeccion = None

    return render_template("index.html", pan=analitica.panorama(col), dl=analitica.delitos(col),
                           ter=analitica.territorio(col), proyeccion=proyeccion, visitas=visitas)

# ============================================================
#  MÓDULO: AUDITORÍA Y USUARIOS ACTIVOS
# ============================================================
@app.route('/usuarios-activos')
@login_required
@admin_required
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
def datos_mapa():
    """Datos del mapa de calor por departamento (códigos verificados de Highcharts)."""
    # Traemos 50 para asegurar que vengan todos, incluso los pequeños
    data_raw = ranking_departamentos(col, n=50) 
    
    # DICCIONARIO MAESTRO MAPA (Highcharts codes)
    map_mapping = {
        # --- LAS LIMAS (SOLUCIÓN DEL PROBLEMA) ---
        # Códigos verificados en @highcharts/map-collection 2.3.0 (countries/pe/pe-all)
        "LIMA METROPOLITANA": "pe-145",      # 'Lima Province' (distrito capital)
        "REGION LIMA": "pe-lr",              # Lima Provincias (alrededor de la capital)
        "LIMA": "pe-145",                    # Respaldo por si aparece solo "LIMA"

        # --- EL CALLAO ---
        # Ojo: en pe-all, 'pe-cl' se llama "Callao" pero sus coordenadas (-15.2, -70.0) son de PUNO.
        # El Callao real es 'pe-3341' (-12.09, -77.22).
        "PROV. CONST. DEL CALLAO": "pe-3341",
        "CALLAO": "pe-3341",

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
        "PUNO": "pe-cl",                     # polígono de Puno (mal etiquetado como "Callao" en pe-all)
        "SAN MARTIN": "pe-sm",
        "TACNA": "pe-ta",
        "TUMBES": "pe-tu",
        "UCAYALI": "pe-uc"
    }

    # Nombre que se muestra en el mapa (corrige etiquetas erróneas del archivo de Highcharts)
    nombres_visibles = {
        "LIMA METROPOLITANA": "Lima Metropolitana",
        "REGION LIMA": "Lima Provincias",
        "PROV. CONST. DEL CALLAO": "Callao",
    }

    data_mapa = []

    # Registros sin departamento (DPTO_HECHO_NEW vacío) no se pueden ubicar en el mapa
    data_raw = [d for d in data_raw if d.get("_id")]
    top_5 = data_raw[:5]

    for d in data_raw:
        # Limpieza: Mayúsculas y sin espacios a los lados
        nombre_bd = str(d["_id"]).strip().upper()

        # Mapeo exacto (un departamento puede ocupar más de un polígono)
        codes = map_mapping.get(nombre_bd)
        if isinstance(codes, str):
            codes = [codes]

        if codes:
            nombre = nombres_visibles.get(nombre_bd, analitica.titulo(nombre_bd))
            for code in codes:
                data_mapa.append({"hc-key": code, "value": d["total"], "name": nombre})
        else:
            # Esto nos ayudará a ver en los logs de Render si algún nombre sigue fallando
            print(f"⚠️ NO MAPEADO EN MAPA: {nombre_bd}")

    return data_mapa, top_5



def datos_clusters():
    """Clasifica los departamentos en 3 niveles de riesgo con K-Means."""
    # Agrupamos datos por departamento
    pipeline = [
        {"$group": {"_id": "$DPTO_HECHO_NEW", "total": {"$sum": "$cantidad"}}},
        {"$sort": {"total": 1}}
    ]
    data_bd = list(col.aggregate(pipeline))
    
    # Limpieza de nulos
    data_clean = [d for d in data_bd if d["_id"]]
    
    if len(data_clean) < 3: return [], {"bajo": 0, "medio": 0, "alto": 0}

    # Preparamos datos para Machine Learning
    nombres = [d["_id"] for d in data_clean]
    valores = np.array([d["total"] for d in data_clean]).reshape(-1, 1)

    # Ejecutamos K-Means
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(valores)
    labels = kmeans.labels_

    # Ordenamos los clusters para que 0=Bajo, 1=Medio, 2=Alto (Semaforización)
    centroides = kmeans.cluster_centers_.flatten()
    mapa_orden = {old: new for new, old in enumerate(np.argsort(centroides))}
    
    colores = ['#10b981', '#f59e0b', '#f43f5e'] # Verde, Amarillo, Rojo
    etiquetas = ['Bajo', 'Medio', 'Alto']
    
    resultados = []
    for i, nom in enumerate(nombres):
        idx = mapa_orden[labels[i]]
        resultados.append({
            "departamento": nom, 
            "total": int(valores[i][0]),
            "cluster": idx, 
            "color": colores[idx], 
            "etiqueta": etiquetas[idx]
        })
        
    stats = {
        "bajo": sum(1 for r in resultados if r['cluster'] == 0),
        "medio": sum(1 for r in resultados if r['cluster'] == 1),
        "alto": sum(1 for r in resultados if r['cluster'] == 2)
    }
    
    return resultados, stats



# ============================================================
#  RUTAS ANTIGUAS → SECCIONES NUEVAS (no se rompen enlaces)
# ============================================================
def _redirigir(endpoint, ancla=None):
    url = url_for(endpoint)
    return redirect(f"{url}#{ancla}" if ancla else url)


@app.route("/resumen-anual")
@login_required
def resumen_anual():
    return _redirigir("index")


@app.route("/trimestres")
@login_required
def trimestres():
    return _redirigir("index")


@app.route("/modalidades")
@login_required
def modalidades():
    return _redirigir("delitos")


@app.route("/comparativa-foco")
@login_required
def comparativa_foco():
    return _redirigir("delitos", "extorsion")


@app.route("/reporte-lima")
@login_required
def reporte_lima():
    return _redirigir("delitos", "extorsion")


@app.route("/departamentos")
@login_required
def departamentos():
    return _redirigir("territorio", "mapa")


@app.route("/departamentos-percapita")
@login_required
def departamentos_percapita():
    return _redirigir("territorio", "tasa")


@app.route("/regiones")
@login_required
def regiones():
    return _redirigir("territorio", "regiones")


@app.route("/cluster-departamentos")
@login_required
def cluster_departamentos():
    return _redirigir("territorio", "radar")


# ============================================================
#  SECCIONES DE LA HISTORIA
# ============================================================
@app.route("/delitos")
@login_required
def delitos():
    return render_template("delitos.html", d=analitica.delitos(col))


@app.route("/territorio")
@login_required
def territorio():
    data_mapa, _ = datos_mapa()
    clusters, stats = datos_clusters()
    return render_template("territorio.html", t=analitica.territorio(col),
                           data_mapa=data_mapa, clusters=clusters, stats=stats)


@app.route("/metodologia")
@login_required
def metodologia():
    try:
        p = proyeccion_2026(col)
    except Exception:
        p = None
    return render_template("metodologia.html", pan=analitica.panorama(col), p=p,
                           t=analitica.territorio(col), registros=col.estimated_document_count())

# ============================================================
#  MÓDULOS DE INTELIGENCIA ARTIFICIAL
# ============================================================
@app.route('/prediccion-2026')
@login_required
def prediccion_2026():
    try:
        p = proyeccion_2026(col)
    except Exception as e:
        print(f"Error en proyección SARIMA: {e}")
        p = None
    return render_template('prediccion_2026.html', p=p)

@app.route('/agente-estrategico')
@login_required
def agente_estrategico():
    try:
        total_2026, texto_historico = obtener_contexto_ia(col)
    except Exception as e:
        print(f"Error en contexto del estratega: {e}")
        return render_template('agente_estrategico.html', total="No disponible",
                               analisis="No se pudo calcular la proyección en este momento.")

    # El análisis se genera una vez al día y se guarda: así las visitas no gastan cuota de Gemini
    clave = f"estratega-{datetime.now():%Y-%m-%d}-{total_2026}"
    cache = db['cache_ia'].find_one({"_id": clave})
    if cache:
        analisis_ia = cache["texto"]
    else:
        contexto_modalidades = contexto_crecimiento_modalidades(col, top=7)
        analisis_ia = consultar_estratega_ia(total_2026, f"{texto_historico}\n{contexto_modalidades}",
                                             "Extorsión y delitos patrimoniales")
        if analisis_ia not in (MENSAJE_CUOTA_GOOGLE, MENSAJE_ERROR, MENSAJE_LIMITE):
            db['cache_ia'].replace_one({"_id": clave}, {"_id": clave, "texto": analisis_ia,
                                                        "creado": datetime.now()}, upsert=True)
    return render_template('agente_estrategico.html', total="{:,}".format(total_2026), analisis=analisis_ia)

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
def contexto_crecimiento_modalidades(col, top=10):
    """Resume cómo cambió cada modalidad: 2021 vs 2024 (años completos) y ene-oct 2025 vs ene-oct 2024."""
    def por_modalidad(match):
        filas = col.aggregate([
            {"$match": match},
            {"$group": {"_id": {"m": "$P_MODALIDADES", "a": "$ANIO"}, "total": {"$sum": "$cantidad"}}},
        ])
        res = {}
        for f in filas:
            m, a = f["_id"].get("m"), f["_id"].get("a")
            if m:
                res.setdefault(m, {})[int(a)] = f["total"]
        return res

    def var(nuevo, viejo):
        return f"{(nuevo - viejo) / viejo * 100:+.0f}%" if viejo else "n/d"

    completos = por_modalidad({"ANIO": {"$in": [2021, 2024]}})
    parciales = por_modalidad({"ANIO": {"$in": [2024, 2025]}, "MES": {"$lte": 10}})
    principales = sorted(completos, key=lambda m: completos[m].get(2024, 0), reverse=True)[:top]

    lineas = []
    for m in principales:
        c, p = completos.get(m, {}), parciales.get(m, {})
        lineas.append(
            f"{m}: 2021={c.get(2021, 0):,} → 2024={c.get(2024, 0):,} ({var(c.get(2024, 0), c.get(2021, 0))}); "
            f"ene-oct 2025 vs ene-oct 2024: {var(p.get(2025, 0), p.get(2024, 0))}"
        )
    return "EVOLUCIÓN POR MODALIDAD (top 10 de 2024):\n" + "\n".join(lineas) + "\n"


@app.route('/chat-ia', methods=['POST'])
@login_required
@limiter.limit("10 per minute")
def chat_ia():
    mensaje = request.form.get('mensaje', '').strip()[:500]  # limita el tamaño del prompt
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
        contexto_acumulado += "NOTA: 2025 tiene datos solo hasta octubre (año incompleto).\n"

        # C. Evolución por modalidad (para preguntas de crecimiento o tendencia)
        contexto_acumulado += contexto_crecimiento_modalidades(col)

        # D. Proyección 2026 (modelo SARIMA validado)
        try:
            contexto_acumulado += f"PROYECCIÓN: {obtener_contexto_ia(col)[1]}\n"
        except Exception as e:
            print(f"Proyección no disponible para el chat: {e}")

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
    # El modo debug solo se activa explícitamente en local (nunca en producción)
    app.run(debug=os.getenv("FLASK_DEBUG") == "1")
