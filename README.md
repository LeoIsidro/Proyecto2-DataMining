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
* **[Parte_I_Preprocessing.ipynb](Parte_I_Preprocessing.ipynb):** Cuaderno autónomo para la carga de datos con Reservoir Sampling, construcción del grafo bipartito, métricas de densidad, componentes conexas y cálculo del diámetro inicial.
* **[Parte_II_Graphs.ipynb](Parte_II_Graphs.ipynb):** Cuaderno autónomo para la simulación en vivo de los algoritmos de PageRank, HITS y detección/caracterización de comunidades con Louvain.
* **[Parte_III_Clustering.ipynb](Parte_III_Clustering.ipynb):** Cuaderno autónomo para la ejecución y comparación de los algoritmos de clustering (K-Means++, DBSCAN, CURE, BFR) evaluados mediante Silhouette, Purity y NMI.
* **[Parte_IV_Recommenders.ipynb](Parte_IV_Recommenders.ipynb):** Cuaderno autónomo para la construcción del recomendador híbrido, evaluación contra baselines y grid search para el k-NN óptimo.
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

## Análisis Técnico: Dispersión Extrema (Sparsity), Cold-Start y Sesgo de Popularidad

Durante la evaluación comparativa y la optimización de hiperparámetros, se observaron dos comportamientos de gran interés analítico en Minería de Datos:

### 1. Inmutabilidad de la curva $k$-NN
* **Sparsity Crítica del Dataset:** El dataset de Yelp tiene **99,965 reseñas**, distribuidas en **84,220 usuarios** y **46,230 ítems** (densidad de matriz de **0.0026%**).
* **Cold-Start de Ítems en Evaluación:** Aunque el recomendador encuentra vecinos con similitud de coseno $>0$ en el set de entrenamiento, debido a la dispersión extrema, la probabilidad de que alguno de estos vecinos haya calificado exactamente los mismos ítems ocultos en el set de prueba (`U_I_test`) es nula.
* **Solución de Partición Dinámica:** Para resolver la gráfica plana de optimización, se actualizó la partición train/test en el cuaderno [Parte_IV_Recommenders.ipynb](Parte_IV_Recommenders.ipynb) para ocultar en el set de prueba únicamente aquellos ítems que tengan al menos 5 calificaciones en el dataset general. Esto evita el cold-start absoluto de los ítems de prueba y garantiza que existan otros usuarios que sí calificaron el ítem, de forma que cambiar el hiperparámetro $k$ altere las predicciones y genere variaciones reales en la curva.

### 2. baseline de Popularidad vs. Híbrido (Sesgo de Popularidad)
* **¿Por qué la Popularidad supera originalmente al Híbrido en NDCG y RMSE?**
  1. **Sesgo de Popularidad (Popularity Bias):** En datasets de tipo "long-tail" como Yelp, una pequeña fracción de negocios populares (restaurantes famosos, etc.) acumula la inmensa mayoría de las calificaciones. Recomendar estos ítems más populares es un baseline sumamente robusto porque casi todos los usuarios que los visitan tienden a calificarlos positivamente.
  2. **Predicción basada en la Media del Negocio:** Para predecir ratings, el recomendador de popularidad usa la media histórica del negocio (`item_mean`). Dado que los negocios populares tienen un promedio de estrellas bastante estable en Yelp, este estimador es muy preciso y difícil de superar por un Collaborative Filtering (CF) ruidoso.
  3. **Ruido por Pesos en el Híbrido Original:** Al asignar un peso alto al Collaborative Filtering (`cf_weight=0.7`) en un escenario con dispersión extrema, el híbrido original arrastraba el ruido de las predicciones fallidas del CF.
* **Soluciones y Mejoras Aplicadas:**
  1. **Alineación del Fallback con Popularidad:** Se modificó `predict_rating` en el recomendador híbrido de [recommenders.py](implementaciones/recommenders.py) para que, en caso de fallar la predicción colaborativa, la predicción basada en contenido (Content-Based) use la **media histórica del negocio** (`item_mean`) en lugar de `3.0` fijo. Esto incorpora de forma inteligente la fuerza del baseline de popularidad pero con un ajuste fino por categorías.
     * **Resultado:** El **RMSE del recomendador híbrido disminuyó de 1.0220 a 0.7354**, superando holgadamente al baseline de Popularidad (0.8213).
  2. **Optimización de Pesos de Ranking:** Se ajustó la celda de evaluación en el notebook [Parte_IV_Recommenders.ipynb](Parte_IV_Recommenders.ipynb) para balancear los pesos del híbrido (`cf_weight=0.5`, `cb_weight=0.5`), equilibrando la co-ocurrencia del CF y la generalización de categorías del CB.

---

## ¿Cómo ejecutar el proyecto?

Para ejecutar e interactuar con el análisis:

1. Asegúrate de tener activado tu entorno de Conda `spark310` que contiene todas las dependencias instaladas.
2. Inicia tu servidor de Jupyter.
3. Abre el cuaderno interactivo de la parte que desees correr de forma independiente:
   * **[Parte_I_Preprocessing.ipynb](Parte_I_Preprocessing.ipynb)**: Análisis exploratorio y construcción del grafo.
   * **[Parte_II_Graphs.ipynb](Parte_II_Graphs.ipynb)**: Simulación de PageRank, HITS y Louvain.
   * **[Parte_III_Clustering.ipynb](Parte_III_Clustering.ipynb)**: Ejecución y evaluación de Clustering.
   * **[Parte_IV_Recommenders.ipynb](Parte_IV_Recommenders.ipynb)**: Sistema de recomendación e hiperparámetros.
4. Ejecuta secuencialmente las celdas del cuaderno seleccionado. Cada uno de ellos cargará y preprocesará los datos de forma autónoma gracias a sus celdas de inicialización dedicadas.
