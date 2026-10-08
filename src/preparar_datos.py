from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler


def cargar_dataset():
    """
    Carga el dataset de transporte de la actividad anterior.
    """

    ruta_proyecto = Path(__file__).resolve().parent.parent
    archivo = ruta_proyecto / "data" / "demanda_transporte.csv"

    datos = pd.read_csv(archivo)

    return datos


def preparar_datos(datos):
    """
    Selecciona las variables para aprendizaje no supervisado
    y realiza su normalización.
    """

    variables = [
        "hora",
        "es_festivo",
        "pasajeros",
        "frecuencia_buses",
        "incidente"
    ]

    X = datos[variables].copy()

    escalador = StandardScaler()

    X_normalizado = escalador.fit_transform(X)

    return X_normalizado, escalador


if __name__ == "__main__":

    print("=" * 60)
    print("PREPARACIÓN DE DATOS - APRENDIZAJE NO SUPERVISADO")
    print("=" * 60)

    datos = cargar_dataset()

    X_normalizado, escalador = preparar_datos(datos)

    print(f"\nRegistros cargados: {len(datos)}")
    print(f"Variables utilizadas: {X_normalizado.shape[1]}")

    print("\nPrimeros cinco registros:")
    print(datos.head())

    print("\nDimensiones de los datos normalizados:")
    print(X_normalizado.shape)

    print("\nPreparación completada correctamente.")