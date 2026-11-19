"""Las 6 secciones de la historia y el simulador. Solo leen resultados ya calculados."""
import logging

from flask import Blueprint, redirect, render_template, request, url_for
from flask_login import login_required

from sidpol.extensiones import repo
from sidpol.servicios import ia, modelos, resultados

bp = Blueprint("secciones", __name__)
log = logging.getLogger(__name__)


def _r(nombre):
    """Resultado precalculado; si algo falla, la página se muestra sin ese bloque."""
    try:
        return resultados.obtener(repo(), nombre)
    except Exception as e:  # noqa: BLE001
        log.error("resultado no disponible", extra={"resultado": nombre, "error": str(e)})
        return None


@bp.route("/")
@login_required
def index():
    try:
        visitas = repo().contar_visita()
    except Exception:  # noqa: BLE001
        visitas = 0
    return render_template("index.html", pan=_r("panorama"), dl=_r("delitos"), ter=_r("territorio"),
                           proyeccion=_r("proyeccion"), visitas=visitas)


@bp.route("/delitos")
@login_required
def delitos():
    return render_template("delitos.html", d=_r("delitos"))


@bp.route("/territorio")
@login_required
def territorio():
    cl = _r("clusters") or {"clusters": [], "stats": {"bajo": 0, "medio": 0, "alto": 0}}
    return render_template("territorio.html", t=_r("territorio"), data_mapa=_r("mapa") or [],
                           clusters=cl["clusters"], stats=cl["stats"])


@bp.route("/metodologia")
@login_required
def metodologia():
    return render_template("metodologia.html", pan=_r("panorama"), p=_r("proyeccion"), t=_r("territorio"),
                           registros=repo().denuncias.estimated_document_count())


@bp.route("/prediccion-2026")
@login_required
def prediccion_2026():
    return render_template("prediccion_2026.html", p=_r("proyeccion"))


# El modelo del simulador se entrena una vez por versión del dataset (con ~800 filas tarda milisegundos)
_simulador = {"version": None, "modelo": None}


def _modelo_simulador(datos):
    version = repo().version_dataset()
    if _simulador["version"] != version:
        _simulador.update(version=version, modelo=modelos.entrenar_simulador(datos))
    return _simulador["modelo"]


@bp.route("/riesgo-modalidad", methods=["GET", "POST"])
@login_required
def riesgo_modalidad():
    datos = _r("simulador")
    entrada = {"anio": max((datos or {}).get("anios") or [2025]), "trimestre": "T1", "departamento": "LIMA METROPOLITANA"}
    resultado, analisis = 0, None

    if request.method == "POST" and datos:
        try:
            entrada = {"anio": int(request.form.get("anio")), "trimestre": request.form.get("trimestre"),
                       "departamento": request.form.get("departamento")}
            modelo, codificador = _modelo_simulador(datos)
            resultado = modelos.predecir_simulador(modelo, codificador, **entrada)
            analisis = ia.analizar_riesgo(resultado, datos["modalidad"], entrada["departamento"],
                                          entrada["trimestre"], entrada["anio"])
        except (TypeError, ValueError) as e:
            log.warning("simulación con datos no válidos: %s", e)

    return render_template(
        "riesgo_modalidad.html", modalidad=(datos or {}).get("modalidad", "Extorsión"),
        departamentos=(datos or {}).get("departamentos", []), anios_disp=(datos or {}).get("anios", []),
        trimestres_disp=modelos.TRIMESTRES, entrada=entrada, resultado=resultado, analisis_ia=analisis,
        labels=(datos or {}).get("labels", []), valores=(datos or {}).get("valores", []))


# ---------- Rutas antiguas → secciones nuevas (no se rompen enlaces) ----------
REDIRECCIONES = {
    "/resumen-anual": ("secciones.index", None), "/trimestres": ("secciones.index", None),
    "/modalidades": ("secciones.delitos", None), "/comparativa-foco": ("secciones.delitos", "extorsion"),
    "/reporte-lima": ("secciones.delitos", "extorsion"), "/departamentos": ("secciones.territorio", "mapa"),
    "/departamentos-percapita": ("secciones.territorio", "tasa"),
    "/regiones": ("secciones.territorio", "regiones"), "/cluster-departamentos": ("secciones.territorio", "radar"),
}


def _crear_redireccion(endpoint, ancla):
    @login_required
    def vista():
        url = url_for(endpoint)
        return redirect(f"{url}#{ancla}" if ancla else url)
    return vista


for _ruta, (_endpoint, _ancla) in REDIRECCIONES.items():
    bp.add_url_rule(_ruta, "antigua" + _ruta.replace("-", "_").replace("/", "_"),
                    _crear_redireccion(_endpoint, _ancla))
