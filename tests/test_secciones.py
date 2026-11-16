"""Pruebas de la estructura por secciones (Panorama, Delitos, Territorio, Metodología)."""
import pytest
from conftest import USUARIOS, login

import analitica

MODALIDADES = ["Extorsión", "Homicidio", "Estafa", "Hurto", "Robo",
               "Violencia contra la mujer e integrantes", "Otros"]
DISTRITOS = [("TRUJILLO", "LA LIBERTAD"), ("SAN JUAN DE LURIGANCHO", "LIMA METROPOLITANA"),
             ("SULLANA", "PIURA"), ("AREQUIPA", "AREQUIPA"), ("CUSCO", "CUSCO")]


@pytest.fixture
def datos_completos(base_limpia):
    """Dataset pequeño con todos los campos reales: 2018-2025 (hasta octubre)."""
    col = base_limpia["denuncias"]
    col.delete_many({})
    docs = []
    for anio in range(2018, 2026):
        for mes in range(1, 13):
            if anio == 2025 and mes > 10:
                break
            for i, mod in enumerate(MODALIDADES):
                for j, (dist, dpto) in enumerate(DISTRITOS):
                    base = 50 + 10 * i + 5 * j
                    if mod == "Extorsión":
                        base = (5 + (anio - 2018) * 12 * (5 - j)) * (1.5 if anio == 2025 else 1)  # crece y se concentra
                    docs.append({"ANIO": anio, "MES": mes, "trimestre": f"T{(mes - 1) // 3 + 1}",
                                 "anio_trimestre": f"{anio}-T{(mes - 1) // 3 + 1}",
                                 "DPTO_HECHO_NEW": dpto, "PROV_HECHO": dist, "DIST_HECHO": dist,
                                 "P_MODALIDADES": mod, "cantidad": int(base)})
    col.insert_many(docs)
    analitica.limpiar_cache()
    yield col
    col.delete_many({})


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


def test_secciones_sin_datos_no_se_rompen(client, base_limpia):
    base_limpia["denuncias"].delete_many({})
    login(client, "demo", USUARIOS["demo"][0])
    for ruta in ["/", "/delitos", "/territorio", "/metodologia"]:
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
