"""Pruebas del modelo de proyección 2026 (SARIMA) con una serie sintética."""
import numpy as np
import pytest
from conftest import USUARIOS, login

import app as sidpol
import ml_utils


@pytest.fixture
def serie_sintetica(base_limpia):
    """60 meses (2021-2025 hasta oct) con tendencia, estacionalidad y ruido conocidos."""
    col = base_limpia["denuncias"]
    col.delete_many({})
    rng = np.random.default_rng(0)
    docs, i = [], 0
    for anio in range(2018, 2026):
        for mes in range(1, 13):
            if anio == 2025 and mes > 10:
                break
            valor = 60000 + 300 * i + 6000 * np.sin(2 * np.pi * mes / 12) + rng.normal(0, 1000)
            if (anio == 2020 and mes >= 3) or (anio == 2021 and mes <= 2):
                valor *= 0.5  # caída tipo pandemia
            docs.append({"ANIO": anio, "MES": mes, "cantidad": int(valor)})
            i += 1
    col.insert_many(docs)
    ml_utils._cache_proyeccion.update(clave=None, resultado=None)
    yield col
    col.delete_many({})


def test_proyeccion_2026_estructura_y_coherencia(serie_sintetica):
    r = ml_utils.proyeccion_2026(serie_sintetica)
    assert r is not None
    assert len(r["pred_2026"]) == 12
    assert r["total_2026"] == sum(r["pred_2026"]) or abs(r["total_2026"] - sum(r["pred_2026"])) <= 12
    assert r["total_2026_min"] < r["total_2026"] < r["total_2026_max"]
    assert r["ultimo_mes"] == "Oct 2025"
    # Las series del gráfico tienen todas el mismo largo (24 reales + 14 pronosticados)
    largos = {len(r[k]) for k in ("graf_etiquetas", "graf_real", "graf_pred", "graf_min", "graf_max")}
    assert largos == {38}


def test_sarima_supera_a_la_regresion_lineal(serie_sintetica):
    r = ml_utils.proyeccion_2026(serie_sintetica)
    assert r["mape_sarima"] < r["mape_lineal"]
    assert r["mape_sarima"] < 10


def test_pagina_de_proyeccion(client, serie_sintetica):
    login(client, "demo", USUARIOS["demo"][0])
    html = client.get("/prediccion-2026").get_data(as_text=True)
    assert "¿Qué viene en 2026?" in html
    assert "MAPE" in html and "SARIMA" in html


def test_datos_insuficientes_no_rompen_la_pagina(client, base_limpia):
    base_limpia["denuncias"].delete_many({})
    ml_utils._cache_proyeccion.update(clave=None, resultado=None)
    login(client, "demo", USUARIOS["demo"][0])
    r = client.get("/prediccion-2026")
    assert r.status_code == 200
    assert "No hay datos suficientes" in r.get_data(as_text=True)
