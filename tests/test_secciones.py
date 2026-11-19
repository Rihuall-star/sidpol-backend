"""Pruebas de la estructura por secciones (Panorama, Delitos, Territorio, Metodología)."""
import pytest
from conftest import DISTRITOS, USUARIOS, login

from sidpol.servicios import analitica


def test_analitica_detecta_los_hallazgos(datos_completos):
    d = analitica.delitos(datos_completos)
    ext = d["extorsion"]
    assert ext["var_2021"] > 50           # la extorsión crece fuerte
    assert ext["record"] is True          # 2025 parcial supera años completos previos
    t = analitica.territorio(datos_completos)
    assert t["focos"][0]["distrito"] == "Trujillo"
    assert 1 <= t["distritos_50"] <= len(DISTRITOS)
    assert sum(r["pct"] for r in t["regiones"]) == pytest.approx(100, abs=0.5)
    p = analitica.panorama(datos_completos)
    assert p["anio_fin"] == 2025 and p["mes_fin"] == 10 and p["periodo"] == "ene–oct"


def test_evolucion_por_modalidad(datos_completos):
    filas = analitica.evolucion_modalidades(datos_completos)
    ext = next(f for f in filas if f["modalidad"] == "Extorsión")
    assert ext["anio_completo"] == 2024 and ext["var_completo"] > 50
    texto = analitica.texto_evolucion(filas, top=3)
    assert texto.startswith("EVOLUCIÓN POR MODALIDAD") and texto.count("\n") == 4


@pytest.mark.parametrize("ruta,texto", [
    ("/", "¿Qué está pasando?"),
    ("/delitos", "¿Qué delitos crecen y cuáles bajan?"),
    ("/territorio", "¿Dónde ocurre?"),
    ("/metodologia", "¿Por qué creerle a los datos?"),
])
def test_secciones_cargan_con_su_pregunta(client, datos_completos, ruta, texto):
    login(client, "demo", USUARIOS["demo"][0])
    r = client.get(ruta)
    assert r.status_code == 200, ruta
    html = r.get_data(as_text=True)
    assert texto in html
    assert "Qué significa" in html or ruta == "/metodologia"


def test_delitos_con_la_version_2026_del_dataset(client, datos_completos):
    """La versión 2026 trae Secuestro en lugar de Homicidio: la sección se adapta sola."""
    datos_completos.update_many({"P_MODALIDADES": "Homicidio"}, {"$set": {"P_MODALIDADES": "Secuestro"}})
    d = analitica.delitos(datos_completos)
    assert d["segunda_nombre"] == "Secuestro" and d["segunda"]["modalidad"] == "Secuestro"
    login(client, "demo", USUARIOS["demo"][0])
    html = client.get("/delitos").get_data(as_text=True)
    assert "Extorsión y secuestro por trimestre" in html and "homicidio" not in html.lower()


def test_secciones_sin_datos_no_se_rompen(client, base_limpia):
    login(client, "demo", USUARIOS["demo"][0])
    for ruta in ["/", "/delitos", "/territorio", "/metodologia", "/riesgo-modalidad"]:
        assert client.get(ruta).status_code == 200, ruta


@pytest.mark.parametrize("antigua,nueva", [
    ("/modalidades", "/delitos"), ("/reporte-lima", "/delitos#extorsion"),
    ("/departamentos", "/territorio#mapa"), ("/cluster-departamentos", "/territorio#radar"),
    ("/regiones", "/territorio#regiones"), ("/resumen-anual", "/"),
])
def test_rutas_antiguas_redirigen(client, antigua, nueva):
    login(client, "demo", USUARIOS["demo"][0])
    r = client.get(antigua)
    assert r.status_code == 302 and r.headers["Location"].endswith(nueva)


def test_menu_tiene_las_seis_secciones(client, datos_completos):
    login(client, "demo", USUARIOS["demo"][0])
    html = client.get("/").get_data(as_text=True)
    for nombre in ["Panorama", "Delitos", "Territorio", "Proyección", "IA estratégica", "Metodología"]:
        assert nombre in html


def test_mapa_y_niveles_de_riesgo(client, datos_completos):
    login(client, "demo", USUARIOS["demo"][0])
    html = client.get("/territorio").get_data(as_text=True)
    assert '"hc-key": "pe-145"' in html        # Lima Metropolitana en el mapa
    assert "crítico (" in html


def test_simulador_estima_por_departamento_y_trimestre(client, datos_completos):
    from conftest import csrf_de
    login(client, "demo", USUARIOS["demo"][0])
    html = client.get("/riesgo-modalidad").get_data(as_text=True)
    assert "casos de extorsión" in html          # la modalidad aparece en el titular
    r = client.post("/riesgo-modalidad", data={"anio": 2025, "trimestre": "T2", "departamento": "LA LIBERTAD",
                                               "csrf_token": csrf_de(html)})
    assert r.status_code == 200
    assert "Proyección Estimada" in r.get_data(as_text=True)


def test_metodologia_muestra_el_diagrama_de_arquitectura(client, datos_completos):
    login(client, "demo", USUARIOS["demo"][0])
    html = client.get("/metodologia").get_data(as_text=True)
    assert "img/arquitectura.svg" in html
    r = client.get("/static/img/arquitectura.svg")
    assert r.status_code == 200 and b"<animateMotion" in r.data   # el diagrama está animado
