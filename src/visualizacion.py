from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from clustering import ejecutar_clustering


def generar_visualizaciones():
    resultados = ejecutar_clustering()

    datos = resultados["datos"]
    X = resultados["X"]
    evaluaciones = resultados["resultados_k"]

    ruta_proyecto = Path(__file__).resolve().parent.parent
    carpeta = ruta_proyecto / "evidencias"
    carpeta.mkdir(parents=True, exist_ok=True)

    # 1. Comparación del coeficiente de silueta
    valores_k = [r["k"] for r in evaluaciones]
    siluetas = [r["silueta"] for r in evaluaciones]

    plt.figure(figsize=(8, 5))
    plt.plot(valores_k, siluetas, marker="o")
    plt.title("Evaluación de K-Means - Silhouette Score")
    plt.xlabel("Número de clústeres (K)")
    plt.ylabel("Coeficiente de silueta")
    plt.xticks(valores_k)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(carpeta / "01_evaluacion_k.png", dpi=150)
    plt.close()

    # 2. Distribución de registros por clúster
    distribucion = datos["cluster"].value_counts().sort_index()

    plt.figure(figsize=(8, 5))
    barras = plt.bar(
        [f"Clúster {k}" for k in distribucion.index],
        distribucion.values
    )

    for barra in barras:
        altura = barra.get_height()
        plt.text(
            barra.get_x() + barra.get_width() / 2,
            altura,
            str(int(altura)),
            ha="center",
            va="bottom"
        )

    plt.title("Distribución de registros por clúster")
    plt.ylabel("Cantidad de registros")
    plt.tight_layout()
    plt.savefig(carpeta / "02_distribucion_clusters.png", dpi=150)
    plt.close()

    # 3. Proyección de los datos mediante PCA
    # PCA permite representar en 2 dimensiones los 5 atributos.
    pca = PCA(n_components=2)
    componentes = pca.fit_transform(X)

    plt.figure(figsize=(9, 6))
    grafico = plt.scatter(
        componentes[:, 0],
        componentes[:, 1],
        c=datos["cluster"],
        cmap="viridis",
        s=20,
        alpha=0.7
    )

    plt.title("Agrupamiento K-Means - Proyección PCA")
    plt.xlabel("Componente principal 1")
    plt.ylabel("Componente principal 2")
    plt.colorbar(grafico, label="Clúster")
    plt.tight_layout()
    plt.savefig(carpeta / "03_clusters_pca.png", dpi=150)
    plt.close()

    # 4. Exportación de los datos clasificados por clúster
    datos.to_csv(
        carpeta / "resultado_clustering.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 5. Resumen de las características de cada clúster
    columnas = [
        "hora",
        "es_festivo",
        "pasajeros",
        "frecuencia_buses",
        "incidente"
    ]

    resumen = datos.groupby("cluster")[columnas].mean().round(2)

    print("\nCARACTERÍSTICAS PROMEDIO POR CLÚSTER")
    print("-" * 60)
    print(resumen.to_string())

    print("\nARCHIVOS GENERADOS")
    print("-" * 60)

    for nombre in [
        "01_evaluacion_k.png",
        "02_distribucion_clusters.png",
        "03_clusters_pca.png",
        "resultado_clustering.csv"
    ]:
        print(f"Generado: evidencias/{nombre}")

    print("\nVISUALIZACIONES GENERADAS CORRECTAMENTE")


if __name__ == "__main__":
    generar_visualizaciones()