"""Pruebas de la capa gold, los resultados precalculados y el job de precálculo."""
from sidpol import precalculo
from sidpol.servicios import analitica, resultados


def test_gold_reduce_documentos_y_conserva_los_totales(repo, datos_completos):
    # Un segundo distrito en el mismo departamento: en la capa gold se suman en un solo documento
    extra = [{**d, "DIST_HECHO": "ATE"} for d in datos_completos.find({"DPTO_HECHO_NEW": "LIMA METROPOLITANA"},
                                                                       {"_id": 0})]
    datos_completos.insert_many(extra)
    n = precalculo.construir_gold(repo)
    assert 0 < n < datos_completos.count_documents({})
    total = lambda col: next(col.aggregate([{"$group": {"_id": None, "t": {"$sum": "$cantidad"}}}]))["t"]  # noqa: E731
    assert total(repo.gold) == total(datos_completos)
    assert repo.agregada().name == "gold_mensual"


def test_analitica_da_lo_mismo_con_gold_y_con_crudos(repo, datos_completos):
    crudo = analitica.delitos(datos_completos)
    precalculo.construir_gold(repo)
    assert analitica.delitos(repo.gold) == crudo
    assert analitica.territorio(repo.gold, repo.denuncias) == analitica.territorio(datos_completos)


def test_resultado_se_guarda_y_sobrevive_a_un_reinicio(repo, datos_completos, monkeypatch):
    llamadas = []
    version, original = resultados.CALCULOS["panorama"]
    monkeypatch.setitem(resultados.CALCULOS, "panorama", (version, lambda r: llamadas.append(1) or original(r)))
    a = resultados.obtener(repo, "panorama")
    resultados.olvidar()                      # simula que el servidor se reinició
    b = resultados.obtener(repo, "panorama")
    assert a == b and len(llamadas) == 1      # el segundo se leyó de la colección `resultados`
    assert repo.resultados.find_one({"_id": "panorama"})["version_dataset"] == repo.version_dataset()


def test_nueva_carga_de_datos_invalida_los_resultados(repo, datos_completos):
    antes = resultados.obtener(repo, "panorama")["total"]
    datos_completos.insert_one({"ANIO": 2025, "MES": 1, "DPTO_HECHO_NEW": "CUSCO", "P_MODALIDADES": "Robo",
                                "cantidad": 1000})
    repo.registrar_version("v-nueva", datos_completos.count_documents({}))
    assert resultados.obtener(repo, "panorama")["total"] == antes + 1000


def test_cambio_de_version_del_calculo_lo_recalcula(repo, datos_completos, monkeypatch):
    resultados.obtener(repo, "panorama")
    resultados.olvidar()
    _, funcion = resultados.CALCULOS["panorama"]
    monkeypatch.setitem(resultados.CALCULOS, "panorama", (999, lambda r: {"nuevo": True}))
    assert resultados.obtener(repo, "panorama") == {"nuevo": True}


def test_precalculo_completo(repo, datos_completos):
    resumen = precalculo.ejecutar(repo, nueva_version=True)
    assert resumen["gold_documentos"] > 0
    assert set(resumen["tiempos"]) == set(resultados.CALCULOS)
    guardados = {d["_id"] for d in repo.resultados.find()}
    assert guardados == set(resultados.CALCULOS)
    assert repo.db["meta"].find_one({"_id": "dataset"})["version"] == resumen["version"]


def test_etl_valida_y_descarta_filas_invalidas(tmp_path):
    import etl_carga_mongo as etl
    csv = tmp_path / "datos.csv"
    csv.write_text(
        "ANIO;MES;DPTO_HECHO_NEW;PROV_HECHO;DIST_HECHO;UBIGEO_HECHO;P_MODALIDADES;cantidad\n"
        "2024;3;CUSCO;CUSCO;CUSCO;080101;Robo;5\n"
        "2024;13;CUSCO;CUSCO;CUSCO;080101;Robo;5\n"        # mes inválido
        "abc;1;CUSCO;CUSCO;CUSCO;080101;Robo;5\n"         # año inválido
        "2024;4; PUNO ;PUNO;PUNO;210101;Hurto;-1\n",      # cantidad negativa
        encoding="utf-8")
    df, descartadas = etl.validar(etl.leer_csv(csv))
    assert len(df) == 1 and descartadas == 3
    fila = df.iloc[0]
    assert fila["trimestre"] == "T1" and fila["anio_trimestre"] == "2024-T1" and fila["UBIGEO_HECHO"] == "080101"
