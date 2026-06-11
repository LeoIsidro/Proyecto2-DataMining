# Análisis de Grafos de Yelp: Rankings de Influencia y Detección de Comunidades

Este proyecto aplica técnicas de **Minería de Datos en Grafos** para analizar las interacciones entre usuarios y negocios de Yelp. A través de la construcción de un grafo bipartito y la implementación de algoritmos matemáticos desde cero (sin depender de librerías externas de redes como `networkx`), identificamos los actores más influyentes y descubrimos comunidades unidas por geografía o afinidad de consumo.

## ¿De qué trata este proyecto?
Yelp conecta a usuarios que escriben reseñas con los negocios locales que visitan. Esto forma de manera natural una red o **grafo bipartito** (donde los nodos son de dos tipos: Usuarios y Negocios, y las conexiones representan reseñas).
Analizar esta red nos permite:
1. **Identificar Influencia (PageRank):** Encontrar qué negocios acumulan la mayor visibilidad en la red.
2. **Modelar Roles de Consumo (HITS):** Descubrir los "Hubs" (usuarios expertos cuyas reseñas tienen gran peso) y las "Authorities" (negocios populares validados por esos hubs).
3. **Detectar Comunidades (Louvain):** Agrupar a los usuarios y negocios en comunidades cohesivas, revelando cómo se autoorganizan por ciudades (geografía) y por nichos de mercado (afinidad de brunch, comida gourmet, etc.).

---

## Estructura del Proyecto

El código está organizado de manera modular en los siguientes archivos:

* **[graphs.py](graphs.py):** Módulo central donde se definen las clases de grafos y se implementan desde cero todos los algoritmos core:
  * PageRank iterativo.
  * HITS (Hubs y Authorities).
  * Algoritmo de Louvain para modularidad.
  * Multi-sweep BFS para estimación del diámetro de la red.
* **[clustering.py](implementaciones/clustering.py):** Módulo para algoritmos de clustering (DBSCAN y CURE) y métricas de evaluación (Silhouette, Purity), programados vectorialmente.
* **[recommenders.py](implementaciones/recommenders.py):** Sistema de recomendación híbrido que combina Filtrado Colaborativo y Basado en Contenido con cálculo de métricas (Precision, Recall, NDCG, RMSE, MAE).
* **[preprocessing.py](preprocessing.py):** Módulo para la carga y limpieza eficiente de los datasets de Yelp. Utiliza técnicas de muestreo (*Reservoir Sampling*) para procesar gigabytes de datos en segundos sin agotar la memoria RAM.
* **[Proyecto2.ipynb](Proyecto2.ipynb):** Cuaderno interactivo de Jupyter que guía paso a paso a través de la construcción del grafo, cálculo de métricas, y ahora incluye secciones iterativas de Clustering (Parte III) y Sistemas de Recomendación (Parte IV).
* **[run_analysis.py](run_analysis.py):** Módulo de utilidades analíticas que provee las funciones necesarias para calcular métricas comparativas y caracterizar comunidades estructuradamente dentro del cuaderno.
* **[build_graph.py](build_graph.py):** Script preliminar de prueba para la construcción del grafo y validación de componentes conexas y diámetro.

---

## Conceptos Clave de los Algoritmos (Implementados desde Cero)

### 1. PageRank Iterativo
Modela el flujo de influencia en la red. Imagina a un usuario navegando de forma aleatoria por las reseñas. La probabilidad a largo plazo de que termine visitando un negocio representa su puntuación de PageRank. Cuenta con una constante de teleportación para evitar quedar atrapado en subgrafos aislados.

### 2. HITS (Hubs y Authorities)
Define una relación de beneficio mutuo en redes bipartitas:
* Un buen **Hub** es un usuario que reseña muchos negocios populares (Authorities).
* Una buena **Authority** es un negocio que recibe reseñas de usuarios muy activos (Hubs).

### 3. Detección de Comunidades (Louvain)
Agrupa nodos buscando maximizar la **Modularidad**, una métrica que compara cuántas conexiones internas existen dentro de los grupos frente a lo que ocurriría por puro azar. Funciona en dos fases: asignación óptima de comunidades locales (Fase 1) y colapso del grafo en metanodos para detectar estructuras jerárquicas más grandes (Fase 2).

### 4. Clustering y Segmentación Estratégica
Se implementaron algoritmos para agrupar usuarios o negocios basándose en su proximidad en un espacio dimensional:
* **K-Means++:** Partición con inicialización optimizada. Se evalúa el k óptimo a través de la inercia (WCSS) y la silueta.
* **DBSCAN:** Agrupa áreas de alta densidad y aísla el ruido. Estimación de vecindario vía K-Distance plot.
* **CURE:** Algoritmo aglomerativo basado en representantes múltiples para lidiar con formas complejas de clusters.
* **BFR:** Extensión escalable procesada en bloques (DS, CS, RS) que evalúa puntos mediante distancia de Mahalanobis.
*Evaluado mediante Silhouette, Purity y NMI.*

### 5. Recomendación Escalable e Híbrida
Un motor que mejora la precisión fusionando múltiples perspectivas, validado contra baselines de Aleatoriedad y Popularidad:
* **Filtrado Colaborativo:** K-NN ponderado User/Item-Based basado en similitud Coseno y Pearson. Manejo de cold-start.
* **Basado en Contenido:** Extracción de features (ej. TF-IDF) para mapear perfiles y encontrar similitud vectorial.
* **Híbrido:** Combinación ponderada (weighted average) evaluada por Precision@K, Recall@K, NDCG, RMSE y MAE.

---

## ¿Cómo ejecutar el proyecto?

Para ejecutar e interactuar con el análisis de grafos y rankings:

1. Asegúrate de tener activado tu entorno de Conda `spark310` que contiene todas las dependencias instaladas.
2. Inicia tu servidor de Jupyter y abre el cuaderno interactivo **[Proyecto2_ParteII_Grafos.ipynb](Proyecto2_ParteII_Grafos.ipynb)** (o **[analisis_grafo.ipynb](analisis_grafo.ipynb)**).
3. Ejecuta secuencialmente las celdas del cuaderno. Este cargará los datos, construirá el grafo bipartito y correrá de forma interactiva y en vivo las implementaciones nativas de PageRank, HITS y Louvain, mostrando gráficos y tablas comparativas dinámicamente en tu pantalla.
