# Reporte de Análisis Semántico y Comparativo de Clustering - Yelp Dataset

Este documento presenta un análisis en profundidad de los resultados de segmentación (clustering) obtenidos sobre el dataset de Yelp (muestreo denso K-Core, 1,500 negocios más calificados) en el espacio de características latentes (PCA + SVD). 

El objetivo es validar la coherencia semántica de los clústeres generados por cada algoritmo para determinar qué tipos de negocios agrupan y cuál es el más conveniente para el análisis estratégico del dataset.

---

## 1. Perfilado y Análisis Semántico de los Algoritmos

A partir del perfilado semántico (promedio de estrellas, volumen de reseñas y categorías dominantes), evaluamos el comportamiento de los cuatro algoritmos:

### A. K-Means++ ($K=5$)
Este algoritmo divide el dataset en 5 clústeres con una distribución muy equilibrada y con límites comerciales sumamente claros:
* **Clúster 0 (86 negocios):** *Estrellas: 4.05 | Reseñas: 756.1*. Dominado en un 100% por restaurantes **Vegetarianos y Veganos**. Representa el nicho de comida saludable/verde.
* **Clúster 1 (1,035 negocios):** *Estrellas: 3.98 | Reseñas: 800.2*. Dominado por **Restaurants, Nightlife y Bars**. Es el clúster principal de entretenimiento nocturno y gastronomía general.
* **Clúster 2 (95 negocios):** *Estrellas: 4.02 | Reseñas: 759.9*. Dominado por **Restaurants (91.6%), Food (29.5%) y Nightlife (28.4%)**.
* **Clúster 3 (146 negocios):** *Estrellas: 4.06 | Reseñas: 1,011.9*. Dominado en un 100% por **Restaurants y Sandwiches**. Es el sector de comida rápida de alta rotación (lunch spots) y es el que tiene el promedio de reseñas más alto (negocios sumamente populares).
* **Clúster 4 (108 negocios):** *Estrellas: 3.90 | Reseñas: 855.1*. Dominado por **Event Planning, Venues y Catering**. Agrupa servicios de logística y eventos en lugar de establecimientos puramente gastronómicos.

> [!NOTE]
> **K-Means++** es el mejor algoritmo para **macro-segmentación estratégica**, ya que genera perfiles humanos y comerciales muy fáciles de entender para la toma de decisiones empresariales.

---

### B. DBSCAN ($eps=6.5, min\_samples=5$)
DBSCAN detecta la densidad local de forma autónoma. En lugar de forzar un número de clústeres, descubre **118 clústeres de nicho** y aísla 23 negocios como ruido (`-1`):
* **Clúster 2 (43 negocios):** Coffee & Tea / Cafes (100% Restaurants). **Cafeterías y locales de café**.
* **Clúster 11 (24 negocios):** Vegetarian / Vegan. **Restaurantes vegetarianos**.
* **Clúster 12 (49 negocios):** Arts & Entertainment / Active Life. **Entretenimiento no gastronómico**.
* **Clúster 81 (8 negocios):** Italian / Pizza. **Pizzerías y trattorias italianas**.

> [!TIP]
> **DBSCAN** es ideal para **motores de recomendación (micro-segmentación)**. Si un usuario disfruta de una pizzería italiana, DBSCAN permite recomendar otros negocios en el clúster 81, garantizando una afinidad semántica casi perfecta. No obstante, es inútil para la toma de decisiones estratégicas globales debido a la enorme fragmentación (118 clústeres).

---

### C. CURE ($K=3$)
El algoritmo CURE (que utiliza múltiples puntos representativos por clúster) sufre severamente en este dataset por el **efecto de encadenamiento (chaining)**:
* **Clúster 0 (1,444 negocios):** Agrupa al **98.2% de los datos**.
* **Clúster 1 (19 negocios) y Clúster 2 (7 negocios):** Clústeres residuales y minúsculos.

> [!WARNING]
> Debido a que las categorías comerciales y las interacciones en Yelp tienen transiciones continuas (ej. Pizza se conecta con Italiano, Italiano con Americano, etc.), los representantes de CURE terminan fusionando toda la masa en un solo superclúster. **CURE queda descartado para este dataset**.

---

### D. BFR ($K=3$)
BFR (variante de K-Means para datos masivos) genera una segmentación híbrida debido a que conserva el Discard Set (DS) y crea micro-clústeres en el Compression Set (CS) y Retained Set (RS). El resultado consta de **52 clústeres**:
* **Clúster 1 (569 negocios) y Clúster 2 (367 negocios):** Concentran la gran masa de restaurantes y vida nocturna.
* **Micro-clústeres de nicho:** Genera grupos como el **Clúster 5 (Burgers)**, el **Clúster 20 (Vegetarian/Vegan)** o el **Clúster 29 (Desserts/Food)**.

> [!IMPORTANT]
> Aunque BFR tiene un comportamiento semántico interesante al aislar ciertos nichos en su Compression Set, el resultado final es ruidoso y complejo de interpretar para un analista, ya que no respeta el $K=3$ inicial en su salida de etiquetas (genera 52 clústeres debido a los CS/RS sub-clústeres).

---

## 2. Comparativa Teórica y de Métricas de Calidad

| Criterio | K-Means++ | DBSCAN | CURE | BFR |
| :--- | :---: | :---: | :---: | :---: |
| **Silhouette (Cohesión)** | **0.4244** (Estable) | 0.4511 (Excluyendo ruido) | **0.7689** (Engañoso) | -0.4143 (Malo por CS/RS) |
| **Purity (Calificaciones)**| 0.7700 | 0.8120 | 0.7707 | 0.8207 |
| **Número de Clústeres** | Fijo ($K=5$) | Descubierto (118) | Fijo ($K=3$) | Híbrido (52) |
| **Outliers / Ruido** | No detecta | **Detecta (1.5%)** | No detecta | **Detecta (1.2%)** |
| **Geometría** | Esférica | Basada en densidad | Arbitraria | Esférica (Gaussiana) |

* **Silhouette de CURE (0.7689) es engañoso:** Al tener un clúster con el 98% de los datos y otros dos extremadamente alejados, la distancia promedio interna parece muy baja en comparación con la externa, inflando el score matemáticamente, pero sin utilidad real.
* **NMI (Información Mutua) es cercano a 0 en todos:** Se valida que la calificación por estrellas (`stars`) no es una variable adecuada para medir la pureza de un clúster comercial, reforzando la necesidad de este análisis semántico cualitativo.

---

## 3. Conclusión y Recomendación

Para este dataset específico de Yelp, **el algoritmo que más conviene usar es K-Means++**:

1. **Interpretabilidad de Negocio:** Es el único que genera clústeres de macro-segmentación limpios y balanceados que corresponden a verdaderos segmentos de mercado (Comida Saludable, Vida Nocturna, Almuerzo Rápido y Eventos Corporativos).
2. **Eficiencia Computacional:** Su complejidad temporal lineal $O(I \cdot K \cdot N \cdot D)$ lo hace ideal para ejecutarse de forma rápida sobre datos enriquecidos.
3. **Recomendación Alternativa:** Si el fin del proyecto fuera diseñar un recomendador personalizado (ej. "dar opciones similares a este negocio"), se debe utilizar **DBSCAN**, pues sus microclústeres de nicho son de una precisión extraordinaria.

---

## 4. Verificación del Notebook (Conformidad con `experto-datamining`)

Revisamos el notebook [Parte_III_Clustering.ipynb](file:///home/leoisidro/CICLOS/X/DATA_MINING/Proyecto2/Proyecto2-DataMining/Parte_III_Clustering.ipynb) y se confirma el cumplimiento de cada requerimiento del proyecto:

* [x] **K-Means++:** Usa inicialización mejorada. Grafica el Método del Codo (WCSS) y calcula la silueta óptima en la Celda 8. Perfila las características en la Celda 9.
* [x] **DBSCAN:** Grafica el *K-Distance Plot* en la Celda 10 para estimar `eps` y reporta automáticamente el número de outliers detectados.
* [x] **CURE:** Implementa el uso de representantes de clúster. La Celda 14 incluye un **Análisis de Complejidad Temporal y Espacial** detallado comparándolo con K-Means y DBSCAN. La Celda 13 y 14 generan las métricas comparativas (Silhouette, Purity y NMI).
* [x] **BFR:** La clase `BFR` en `implementaciones/clustering.py` implementa el procesamiento en bloques (`chunk_size`), la distancia de Mahalanobis, y el mantenimiento del Discard Set (DS), Compression Set (CS) y Retained Set (RS).
* [x] **Análisis Semántico en Profundidad:** Con la Celda 14 insertada de Perfilado Semántico Comparativo, el notebook queda completamente equipado para generar el conocimiento de negocio extraído en este reporte.
