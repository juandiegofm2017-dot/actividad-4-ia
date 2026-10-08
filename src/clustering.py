import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from preparar_datos import cargar_dataset, preparar_datos


def evaluar_clusters(X, minimo=2, maximo=6):
    """
    Evalúa diferentes cantidades de clusters
    utilizando el coeficiente de silueta.
    """

    resultados = []

    for k in range(minimo, maximo + 1):

        modelo = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        etiquetas = modelo.fit_predict(X)

        silueta = silhouette_score(
            X,
            etiquetas
        )

        resultados.append({
            "k": k,
            "silueta": silueta,
            "inercia": modelo.inertia_
        })

    return resultados


def entrenar_clustering(X, k):
    """
    Entrena el modelo K-Means con el número
    de clusters seleccionado.
    """

    modelo = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    etiquetas = modelo.fit_predict(X)

    return modelo, etiquetas


def ejecutar_clustering():

    datos = cargar_dataset()

    X, escalador = preparar_datos(datos)

    resultados = evaluar_clusters(X)

    mejor_resultado = max(
        resultados,
        key=lambda resultado: resultado["silueta"]
    )

    mejor_k = mejor_resultado["k"]

    modelo, etiquetas = entrenar_clustering(
        X,
        mejor_k
    )

    datos_resultado = datos.copy()

    datos_resultado["cluster"] = etiquetas

    return {
        "datos": datos_resultado,
        "modelo": modelo,
        "escalador": escalador,
        "X": X,
        "resultados_k": resultados,
        "mejor_k": mejor_k,
        "silueta": mejor_resultado["silueta"]
    }


if __name__ == "__main__":

    print("=" * 60)
    print("SISTEMA DE AGRUPAMIENTO DE ESTACIONES - K-MEANS")
    print("=" * 60)

    resultados = ejecutar_clustering()

    print("\nEVALUACIÓN DE CLUSTERS")
    print("-" * 45)

    for resultado in resultados["resultados_k"]:

        print(
            f"K={resultado['k']} | "
            f"Silueta={resultado['silueta']:.4f} | "
            f"Inercia={resultado['inercia']:.2f}"
        )

    print("\nMEJOR CONFIGURACIÓN")
    print("-" * 45)

    print(f"Número de clusters: {resultados['mejor_k']}")
    print(f"Silhouette Score: {resultados['silueta']:.4f}")

    print("\nDISTRIBUCIÓN DE REGISTROS")
    print("-" * 45)

    distribucion = (
        resultados["datos"]["cluster"]
        .value_counts()
        .sort_index()
    )

    for cluster, cantidad in distribucion.items():

        print(
            f"Cluster {cluster}: {cantidad} registros"
        )

    print("\nAGRUPAMIENTO COMPLETADO CORRECTAMENTE")