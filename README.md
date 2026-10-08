# Actividad 4 – Métodos de Aprendizaje No Supervisado

## Descripción

Proyecto académico desarrollado para la asignatura de Inteligencia Artificial, orientado al análisis de información de transporte masivo mediante técnicas de aprendizaje no supervisado.

Se implementa el algoritmo K-Means para identificar agrupaciones de registros con características operativas similares.

## Tecnologías utilizadas

- Python 3.10
- Pandas y NumPy
- Scikit-learn
- Matplotlib
- Pytest

## Dataset

Se utiliza un conjunto de 1.000 registros sintéticos relacionados con estaciones de transporte masivo.

Las variables utilizadas para el agrupamiento son hora, condición de día festivo, cantidad de pasajeros, frecuencia de buses e incidentes.

La variable `nivel_demanda` se excluye del entrenamiento, ya que corresponde a una etiqueta utilizada en la actividad anterior.

## Algoritmo

Se implementa K-Means, evaluando configuraciones de K=2 hasta K=6 mediante el coeficiente de silueta.

**Resultados obtenidos:**

- Mejor configuración evaluada: K=2
- Silhouette Score: 0.3688
- Clúster 0: 884 registros
- Clúster 1: 116 registros

El agrupamiento obtenido diferencia principalmente registros de días festivos y no festivos.

## Ejecución

Instalar dependencias:

```powershell
python -m pip install -r requirements.txt
```

Preparar datos:

```powershell
python src/preparar_datos.py
```

Entrenar K-Means:

```powershell
python src/clustering.py
```

Generar visualizaciones:

```powershell
python src/visualizacion.py
```

Ejecutar aplicación:

```powershell
python src/main.py
```

Ejecutar pruebas:

```powershell
python -m pytest -v
```

## Pruebas

Se implementaron cinco pruebas automatizadas para validar la carga de datos, la normalización, la generación de clústeres, el coeficiente de silueta y la asignación de registros.

**Resultado:** 5 pruebas aprobadas.

## Limitaciones

El dataset utilizado es sintético, por lo que los resultados no deben interpretarse como patrones comprobados de una red de transporte real.

Los clústeres representan agrupaciones de registros operativos y no una clasificación directa de estaciones.