"""Pruebas del modelo de proyección 2026 (SARIMA) con una serie sintética."""
import numpy as np
import pytest
from conftest import USUARIOS, login

from sidpol.servicios import modelos


@pytest.fixture
def serie_sintetica(base_limpia):
    """Serie mensual 2018-2025 (hasta oct) con tendencia, estacionalidad, ruido y caída tipo pandemia."""
    col = base_limpia["denuncias"]
    rng = np.random.default_rng(0)
    docs, i = [], 0
    for anio in range(2018, 2026):
        for mes in range(1, 13):
            if anio == 2025 and mes > 10:
                break
            valor = 60000 + 300 * i + 6000 * np.sin(2 * np.pi * mes / 12) + rng.normal(0, 1000)
            if (anio == 2020 and mes >= 3) or (anio == 2021 and mes <= 2):
                valor *= 0.5
            docs.append({"ANIO": anio, "MES": mes, "cantidad": int(valor)})
            i += 1
    col.insert_many(docs)
    yield col


def test_proyeccion_estructura_y_coherencia(serie_sintetica):
    r = modelos.proyeccion(serie_sintetica)
    assert r is not None
    assert r["anio_actual"] == 2025 and r["anio_objetivo"] == 2026   # proyecta el año siguiente al último dato
    assert len(r["pred_objetivo"]) == 12
    assert abs(r["total_objetivo"] - sum(r["pred_objetivo"])) <= 12
    assert r["total_objetivo_min"] < r["total_objetivo"] < r["total_objetivo_max"]
    assert r["ultimo_mes"] == "Oct 2025" and not r["anio_completo"]
    assert r["cierre_actual"] > r["real_actual"]   # el cierre suma los meses que faltan
    # Las series del gráfico tienen todas el mismo largo (24 reales + 14 pronosticados)
    largos = {len(r[k]) for k in ("graf_etiquetas", "graf_real", "graf_pred", "graf_min", "graf_max")}
    assert largos == {38}


def test_sarima_supera_a_la_regresion_lineal(serie_sintetica):
    r = modelos.proyeccion(serie_sintetica)
    assert r["mape_sarima"] < r["mape_lineal"]
    assert r["mape_sarima"] < 10


def test_pagina_de_proyeccion(client, serie_sintetica):
    login(client, "demo", USUARIOS["demo"][0])
    html = client.get("/prediccion-2026").get_data(as_text=True)
    assert "¿Qué viene en 2026?" in html and "Cierre estimado de 2025" in html
    assert "MAPE" in html and "SARIMA" in html


def test_datos_insuficientes_no_rompen_la_pagina(client, base_limpia):
    login(client, "demo", USUARIOS["demo"][0])
    r = client.get("/prediccion-2026")
    assert r.status_code == 200
    assert "No hay datos suficientes" in r.get_data(as_text=True)


def test_texto_de_proyeccion_para_la_ia(serie_sintetica):
    texto = modelos.texto_proyeccion(modelos.proyeccion(serie_sintetica))
    assert "Proyección 2026" in texto and "Cierre estimado de 2025" in texto and "MAPE" in texto
    assert "no disponible" in modelos.texto_proyeccion(None)


def test_proyeccion_con_datos_hasta_octubre_2026(base_limpia):
    """Con la versión 2026 del dataset, el objetivo pasa a 2027 y se estima el cierre de 2026."""
    col = base_limpia["denuncias"]
    rng = np.random.default_rng(1)
    docs = [{"ANIO": a, "MES": m, "cantidad": int(70000 + 200 * i + 5000 * np.sin(m) + rng.normal(0, 800))}
            for i, (a, m) in enumerate((a, m) for a in range(2018, 2027) for m in range(1, 13) if (a, m) <= (2026, 10))]
    col.insert_many(docs)
    r = modelos.proyeccion(col)
    assert (r["anio_actual"], r["anio_objetivo"], r["ultimo_mes"]) == (2026, 2027, "Oct 2026")
    assert len(r["pred_objetivo"]) == 12
