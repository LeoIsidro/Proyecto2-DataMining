# Análisis de Grafos de Yelp: Rankings de Influencia y Detección de Comunidades

Este proyecto aplica técnicas de **Minería de Datos** para analizar interacciones entre usuarios y negocios de Yelp. A través de la construcción de un grafo bipartito y la implementación de algoritmos desde cero, identificamos comunidades, usuarios influyentes y generamos recomendaciones escalables.

📖 **Reporte final en formato [Wiki](https://github.com/LeoIsidro/Proyecto2-DataMining/wiki)**

---

## Estructura del Proyecto

El código está organizado de manera modular en las siguientes partes principales:

### 1. Implementaciones Principales (`implementaciones/`)
En esta carpeta se encuentran los módulos `.py` centrales donde se han implementado todos los algoritmos "desde cero":
* **`preprocessing.py`**: Limpieza de datos y técnicas de muestreo (Reservoir Sampling).
* **`graphs.py`**: Algoritmos de grafos (PageRank, HITS, Louvain).
* **`clustering.py`**: Modelos de agrupamiento (K-Means++, DBSCAN, CURE, BFR) y métricas.
* **`recommenders.py`**: Sistema de recomendación híbrido (Collaborative & Content-Based).
* **`dimensionality_reduction.py`**: Técnicas de reducción de dimensionalidad.
* **`streaming.py`**: Procesamiento de flujos de datos continuos.

### 2. Cuadernos de Experimentación
Estos cuadernos de Jupyter (`.ipynb`) ejecutan los algoritmos y permiten evaluar el código modularizado de forma interactiva:
* **[Parte_I_Preprocessing.ipynb](Parte_I_Preprocessing.ipynb)**: Análisis exploratorio, carga de datos con muestreo y construcción inicial.
* **[Parte_II_Graphs.ipynb](Parte_II_Graphs.ipynb)**: Análisis de PageRank, HITS y comunidades (Louvain).
* **[Parte_III_Clustering_VI_Reduction_Dimensionality.ipynb](Parte_III_Clustering_VI_Reduction_Dimensionality.ipynb)**: Evaluación de algoritmos de clustering y reducción de dimensionalidad.
* **[Parte_IV_Recommenders.ipynb](Parte_IV_Recommenders.ipynb)**: Construcción y evaluación del recomendador híbrido.
* **[parte_V_.ipynb](parte_V_.ipynb)**: Algoritmos para datos en streaming (ej. HyperLogLog, Bloom Filters).

---

## ¿Cómo ejecutar el proyecto?

1. Asegúrate de tener tu entorno activado con las dependencias necesarias.
2. Inicia el servidor de **Jupyter Notebook / JupyterLab**.
3. Abre el cuaderno interactivo de la parte que desees correr de forma independiente (ej. `Parte_I_Preprocessing.ipynb`).
4. Ejecuta las celdas de forma secuencial. Cada cuaderno se encarga de importar y utilizar los algoritmos de la carpeta `implementaciones/`.
