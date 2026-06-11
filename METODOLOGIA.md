# Reporte de Metodología: Clustering y Recomendación

## 1. Justificación de la Metodología (Clustering - Parte III)
Para la segmentación estratégica de usuarios y productos, implementamos de forma nativa los siguientes enfoques:
* **K-Means++:** Empleado como algoritmo de partición fundamental. La inicialización "++" asegura centroides alejados, acelerando la convergencia y evitando mínimos locales. Utilizamos el *Método del Codo* (basado en Inercia/WCSS) y el *Coeficiente de Silueta* para identificar el $k$ óptimo. 
* **DBSCAN:** Fue seleccionado debido a su capacidad para manejar clusters de formas arbitrarias y detectar anomalías/outliers de forma automática. Usamos un *K-Distance Plot* para estimar visualmente el radio de vecindad óptimo ($eps$).
* **CURE:** Implementado para manejar clusters no globulares a través de múltiples representantes, brindando robustez ante outliers.
* **BFR (Bradley-Fayyad-Reina):** Extension de K-Means para minería masiva. Procesamos datos en bloques manejando *Discard Set (DS)*, *Compression Set (CS)* y *Retained Set (RS)*, evaluando la asignación a través de la Distancia de Mahalanobis.
* **Métricas:** Evaluamos rigurosamente empleando la *Silueta*, *Purity* y la Información Mutua Normalizada (*NMI*).

## 2. Justificación de la Metodología (Sistemas de Recomendación - Parte IV)
Construimos un entorno de recomendación escalable integrando múltiples perspectivas:
* **Filtrado Colaborativo (CF):** Implementado con base en Usuario o Ítem, empleando similitud del Coseno o Correlación de Pearson. Para la generación del Top-K se implementó un K-NN ponderado. Analizamos además el problema de *cold-start*.
* **Filtrado Basado en Contenido (CB):** Aprovecha características extraídas (ej. TF-IDF de descripciones) para ubicar similitudes en un espacio vectorial. Es ideal para ítems nuevos en la red.
* **Modelo Híbrido:** Combinamos predicciones mediante promedio ponderado (*weighted average*).
* **Baselines:** Contrastamos la calidad contra recomendaciones aleatorias (*RandomRecommender*) y de popularidad (*PopularityRecommender*).
* **Evaluación Matemática:** Usamos *Precision@K*, *Recall@K* y *NDCG* para la calidad del ranking, y *RMSE* / *MAE* para la precisión del rating predicho.

*Este documento sirve de base teórica para el reporte final.*
