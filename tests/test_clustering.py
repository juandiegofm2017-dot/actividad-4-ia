import sys
from pathlib import Path

import numpy as np

# Permite importar los módulos ubicados en src
RUTA_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(RUTA_SRC))

from preparar_datos import cargar_dataset, preparar_datos
from clustering import (
    evaluar_clusters,
    entrenar_clustering,
    ejecutar_clustering
)


def test_dataset_cargado():
    """Verifica que el dataset tenga 1000 registros."""

    datos = cargar_dataset()

    assert len(datos) == 1000


def test_preparacion_datos():
    """Verifica las dimensiones y normalización."""

    datos = cargar_dataset()
    X, escalador = preparar_datos(datos)

    assert X.shape == (1000, 5)
    assert np.isfinite(X).all()
    assert np.allclose(X.mean(axis=0), 0, atol=1e-10)


def test_cantidad_clusters():
    """Verifica que K-Means genere dos grupos."""

    datos = cargar_dataset()
    X, _ = preparar_datos(datos)

    modelo, etiquetas = entrenar_clustering(X, 2)

    assert modelo.n_clusters == 2
    assert len(np.unique(etiquetas)) == 2


def test_silhouette_score():
    """Verifica que el coeficiente esté en su rango válido."""

    datos = cargar_dataset()
    X, _ = preparar_datos(datos)

    resultados = evaluar_clusters(X, minimo=2, maximo=2)

    assert len(resultados) == 1
    assert -1 <= resultados[0]["silueta"] <= 1


def test_asignacion_registros():
    """Verifica que todos los registros tengan un clúster."""

    resultados = ejecutar_clustering()

    datos = resultados["datos"]

    assert len(datos) == 1000
    assert datos["cluster"].notna().all()
    assert datos["cluster"].nunique() == resultados["mejor_k"]