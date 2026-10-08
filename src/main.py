from clustering import ejecutar_clustering


def mostrar_resumen(resultados):
    datos = resultados["datos"]

    print("\n" + "=" * 60)
    print("RESUMEN DEL AGRUPAMIENTO")
    print("=" * 60)

    print(f"\nTotal de registros: {len(datos)}")
    print(f"Clústeres identificados: {resultados['mejor_k']}")
    print(f"Silhouette Score: {resultados['silueta']:.4f}")

    print("\nCARACTERÍSTICAS DE LOS CLÚSTERES")
    print("-" * 60)

    columnas = [
        "hora",
        "es_festivo",
        "pasajeros",
        "frecuencia_buses",
        "incidente"
    ]

    resumen = datos.groupby("cluster")[columnas].mean()

    for cluster, valores in resumen.iterrows():

        cantidad = (datos["cluster"] == cluster).sum()

        print(f"\nCLÚSTER {cluster}")
        print(f"Registros: {cantidad}")
        print(f"Hora promedio: {valores['hora']:.2f}")
        print(f"Pasajeros promedio: {valores['pasajeros']:.2f}")
        print(
            f"Frecuencia de buses promedio: "
            f"{valores['frecuencia_buses']:.2f}"
        )
        print(f"Proporción de festivos: {valores['es_festivo']:.2%}")
        print(f"Proporción de incidentes: {valores['incidente']:.2%}")


def consultar_estacion(resultados):
    datos = resultados["datos"]

    estaciones = sorted(datos["estacion"].unique())

    print("\nESTACIONES DISPONIBLES")
    print("-" * 45)

    for indice, estacion in enumerate(estaciones, start=1):
        print(f"{indice}. {estacion}")

    try:
        seleccion = int(input("\nSeleccione una estación: "))

        if not 1 <= seleccion <= len(estaciones):
            print("La estación seleccionada no es válida.")
            return

    except ValueError:
        print("Debe ingresar un número válido.")
        return

    estacion = estaciones[seleccion - 1]

    registros = datos[datos["estacion"] == estacion]

    print("\n" + "=" * 60)
    print(f"RESULTADOS PARA: {estacion.upper()}")
    print("=" * 60)

    print(f"\nTotal de registros: {len(registros)}")

    distribucion = registros["cluster"].value_counts().sort_index()

    for cluster, cantidad in distribucion.items():
        porcentaje = cantidad / len(registros) * 100

        print(
            f"Clúster {cluster}: "
            f"{cantidad} registros ({porcentaje:.2f}%)"
        )


def main():
    print("=" * 60)
    print("SISTEMA INTELIGENTE DE AGRUPAMIENTO DE TRANSPORTE")
    print("APRENDIZAJE NO SUPERVISADO - K-MEANS")
    print("=" * 60)

    print("\nEntrenando modelo y evaluando clústeres...")

    resultados = ejecutar_clustering()

    while True:
        print("\nMENÚ PRINCIPAL")
        print("-" * 45)
        print("1. Consultar resumen del agrupamiento")
        print("2. Consultar resultados por estación")
        print("3. Salir")

        opcion = input("\nSeleccione una opción: ").strip()

        if opcion == "1":
            mostrar_resumen(resultados)

        elif opcion == "2":
            consultar_estacion(resultados)

        elif opcion == "3":
            print("\nPrograma finalizado correctamente.")
            break

        else:
            print("Opción no válida. Intente nuevamente.")


if __name__ == "__main__":
    main()