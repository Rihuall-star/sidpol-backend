# SIDPOL: Análisis y proyección delictiva en el Perú con Machine Learning e IA generativa

Plataforma web de inteligencia de datos sobre las **denuncias policiales del Perú (SIDPOL, enero 2018 – octubre 2025, más de 337,000 registros)**. Combina analítica descriptiva, machine learning y un asistente de IA generativa que responde preguntas en lenguaje natural consultando la base de datos.

> Proyecto académico: curso *Tecnologías Emergentes en Inteligencia Artificial*, Maestría en Ingeniería de Sistemas e Informática, Universidad Nacional Mayor de San Marcos (2025).

**🔗 Demo:** https://sidpol-proyecto.onrender.com
**Usuario:** `demo` · **Contraseña:** `SidpolDemo2026`

> Cuenta de solo lectura con un límite diario de consultas a la IA. El servidor gratuito puede tardar unos 30 segundos en despertar la primera vez.

| Dashboard | Radar de riesgo (K-Means) | Proyección 2026 |
|---|---|---|
| ![Dashboard](docs/img/dashboard.png) | ![Clusters](docs/img/clusters.png) | ![Proyección 2026](docs/img/prediccion.png) |

---

## ✨ Funcionalidades

| Módulo | Qué hace |
|---|---|
| **ETL y calidad de datos** | Normaliza los nombres de los 25 departamentos, agrega trimestres y corrige el conteo de denuncias (suma el campo `cantidad` en lugar de contar filas). |
| **Clustering K-Means** | Agrupa los departamentos por perfil de incidencia por modalidad (en porcentajes, para que el tamaño de Lima no sesgue el resultado) en 3 niveles de riesgo: bajo, medio y crítico. |
| **Análisis trimestral** | Compara la evolución de extorsión y homicidio en departamentos clave con el *Aggregation Framework* de MongoDB. |
| **Análisis regional y per cápita** | Distribución por Costa, Sierra y Selva, y tasas ajustadas por población. |
| **Proyección 2026** | Regresión lineal sobre la serie mensual para estimar la tendencia del 2.º semestre de 2026, total y por modalidad. |
| **Asistente RAG (Gemini)** | Chat que consulta MongoDB antes de responder, para que la respuesta se base en los datos y no en el conocimiento general del modelo. |
| **Agentes** | Un *agente estratégico* genera planes de acción a partir de la proyección y un *agente logístico* asigna recursos según los clusters. |
| **Mapas de calor** | Visualización geoespacial de las zonas críticas (Highcharts Maps). |
| **Autenticación y roles** | Login con Flask-Login, contraseñas con hash (Werkzeug), rutas de administración y auditoría protegidas por rol, y límite diario de consultas a la IA para la cuenta demo. |

## 📊 Hallazgos principales
- **La extorsión crece de forma sostenida desde 2021**, mientras que los homicidios mantienen un crecimiento moderado.
- **Lima Metropolitana** queda sola en el cluster de riesgo crítico (más de 2.3 millones de denuncias); Arequipa, Lambayeque, La Libertad y Piura encabezan el nivel medio.
- **La Costa concentra más del 60%** de las denuncias del país.
- Corregir el conteo (suma de `cantidad` frente a conteo de filas) mostró una incidencia **mayor a la que se obtiene con un conteo simple**.

## 🏗️ Arquitectura

```
CSV SIDPOL ──► ETL (Pandas) ──► MongoDB ──► Flask (lógica + ML) ──► Bootstrap / Chart.js / Plotly / Highcharts
                                   ▲              │
                                   └── RAG ◄──────┴──► Google Gemini API
```

| Capa | Tecnología |
|---|---|
| Datos | MongoDB (PyMongo, Aggregation Framework) |
| Backend | Python 3.10, Flask, Flask-Login, Gunicorn |
| ML | Scikit-learn (K-Means, regresión lineal), Pandas, NumPy |
| IA generativa | Google Gemini API (Gemini 3.5 Flash) |
| Frontend | Jinja2, Bootstrap 5, Chart.js, Plotly.js, Highcharts Maps |
| Despliegue | Render + MongoDB Atlas |

## 📁 Estructura

```
app.py                 # Rutas Flask, autenticación y roles
etl_carga_mongo.py     # Carga y normalización del CSV en MongoDB
mongo_queries.py       # Pipelines de agregación
ml_cluster.py          # Clustering K-Means
ml_utils.py            # Series mensuales y regresión lineal
ml_riesgo.py           # Riesgo por modalidad
chat_logic.py          # Construcción del contexto RAG
gemini_client.py       # Cliente de la API de Gemini
regiones_peru.py       # Clasificación Costa / Sierra / Selva
poblacion_peru.py      # Población para tasas per cápita
templates/  static/    # Interfaz web
```

## 🔐 Seguridad

La aplicación aplica controles alineados con **OWASP Top 10 (2021)**:

| Riesgo OWASP | Control implementado |
|---|---|
| **A01** Pérdida de control de acceso | Roles (`admin` / `analista` / `invitado`) con decoradores `login_required` y `admin_required`; auditoría y gestión de usuarios solo para administradores. Protección **CSRF** (Flask-WTF) en todos los formularios y en el chat. |
| **A02** Fallas criptográficas | Contraseñas con hash (Werkzeug, PBKDF2/scrypt); HTTPS con **HSTS**; cookies `Secure` en producción. |
| **A03** Inyección / XSS | Consultas a MongoDB con parámetros tipados; escape de HTML en las respuestas del chat antes de insertarlas en la página; plantillas Jinja2 con autoescape. |
| **A04** Diseño inseguro | Cuenta demo de solo lectura con **límite diario de consultas a la IA**, guardado en la base de datos. |
| **A05** Configuración incorrecta | Cabeceras **Content-Security-Policy**, `X-Frame-Options: DENY`, `X-Content-Type-Options`, `Referrer-Policy` y `Permissions-Policy`; secretos solo por variables de entorno (la app no arranca sin `SECRET_KEY`). |
| **A06** Componentes vulnerables | **Dependabot** semanal para dependencias de Python. |
| **A07** Fallas de autenticación | **Rate limiting** en el login (5 intentos por minuto y 30 por hora por IP); sesión nueva en cada login (contra la fijación de sesión); cookies `HttpOnly` y `SameSite=Lax`; expiración de la sesión a las 8 horas. |
| **A09** Fallas de registro y monitoreo | Auditoría de inicios de sesión **y de intentos fallidos**, con la IP real del visitante (`ProxyFix` detrás del proxy de Render). |

**Pendiente (próximas mejoras):** CSP basada en *nonces* para eliminar `'unsafe-inline'`, análisis estático con CodeQL y pruebas automatizadas en GitHub Actions.

## 🚀 Ejecución local

```bash
git clone https://github.com/Rihuall-star/sidpol-backend.git
cd sidpol-backend
python -m venv venv
source venv/bin/activate      # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Crea un archivo `.env` (está excluido en `.gitignore`):

```env
MONGO_URI=mongodb://localhost:27017/
GEMINI_API_KEY=tu_api_key
SECRET_KEY=una_cadena_larga_y_aleatoria
# Opcionales
GEMINI_MODEL=gemini-3.5-flash
LIMITE_IA_INVITADO=30
```

Carga los datos y levanta la aplicación:

```bash
python etl_carga_mongo.py
flask --app app run
```

## ⚠️ Limitaciones y trabajo futuro
- La proyección usa **regresión lineal simple**, que capta la tendencia general pero no la estacionalidad. Como siguiente paso se plantean SARIMA o LSTM.
- El análisis es a nivel de **departamento**. Para bajar a distrito o cuadrante se necesitaría georreferenciar cada denuncia.
- Las proyecciones son **exploratorias** y con fines académicos, no pronósticos oficiales.

## 📄 Fuente de datos
Dataset abierto de denuncias policiales (SIDPOL), publicado en la Plataforma Nacional de Datos Abiertos del Perú. (datosabiertos.gob.pe).

## 👤 Autor
**Ricardo Huamaní Allcca**: Ingeniero de Redes y Ciberseguridad, maestrando en la UNMSM.
[LinkedIn](https://www.linkedin.com/in/ricardo-huamani-allcca) · [ORCID](https://orcid.org/0000-0002-1214-2917)
