# Documentación del Sistema de Recomendación (Parte IV)

Este documento explica en detalle los componentes clave de la implementación del sistema de recomendación ubicada en `implementaciones/recommenders.py` y cómo interactúan en el notebook principal. Las explicaciones están orientadas a que comprendas el flujo matemático, lógico y algorítmico (estructuras de datos) detrás de los modelos para que puedas argumentarlos sólidamente en tus experimentos y en el reporte final.

---

## 1. Estructuras de Datos: El Manejo de Matrices Dispersas (Sparsity)

Uno de los cambios arquitectónicos más importantes frente a implementaciones básicas es cómo manejamos la memoria y las búsquedas. Las matrices de Usuario-Ítem en el mundo real (y en este dataset de Yelp) tienen más de un 99% de ceros (usuarios que no han visitado la vasta mayoría de restaurantes). 

Si usáramos un arreglo 2D tradicional (Ej: NumPy densos o Pandas DataFrames) de tamaño `N_usuarios x M_items`, **nos quedaríamos sin RAM (Out of Memory)**. 

### Implementación Nativa de Matriz Dispersa
En nuestro código, en lugar de importar herramientas complejas de `scipy.sparse`, hemos construido nuestra propia **Lista de Adyacencia usando Diccionarios Anidados** de Python:
```python
user_item_dict = {
    user_id_1: {item_id_A: 4.5, item_id_B: 3.0},
    user_id_2: {item_id_A: 5.0, item_id_C: 2.0}
}
```
**Ventajas Críticas que puedes mencionar en tu reporte:**
1. **Eficiencia Espacial:** O(E), donde E es el número de reviews. Ignora los ceros completamente.
2. **Eficiencia Temporal (Indexación en O(1)):** En Python, acceder a las claves de un diccionario toma O(1). 
3. **Similitud Rápida:** Para calcular la similitud entre el usuario 1 y 2, hacemos una **intersección de conjuntos nativa** (`set(u1_items) & set(u2_items)`). Las operaciones de conjuntos (Sets) en Python están implementadas en C usando Tablas Hash, haciendo que encontrar ítems en común entre dos usuarios tome tiempo O(min(len(u1), len(u2))) en lugar de O(M_items).

También construimos la matriz transpuesta en memoria, `item_user_dict`, permitiéndonos consultar en O(1) "qué usuarios calificaron a este ítem", acelerando drásticamente la búsqueda de vecinos candidatos. Además, pre-calculamos las **Normas (magnitud de los vectores)** y las **Medias** durante la fase `fit()` para no repetir sumatorias infinitas en el ciclo `predict()`.

---

## 2. Algoritmos Base (Baselines)
Los baselines sirven como punto de comparación. Si un modelo complejo (como el Híbrido) no supera a estos algoritmos simples, significa que algo anda mal o que los datos no son lo suficientemente ricos.

### 2.1 Random Recommender
* **Cómo funciona:** Recomienda ítems al azar de entre todos los ítems disponibles que el usuario aún no ha consumido.
* **Rating:** Para las predicciones de calificación, devuelve un valor uniforme al azar entre 1 y 5.
* **Propósito:** Actúa como el umbral de rendimiento mínimo absoluto. 

### 2.2 Popularity Recommender
* **Cómo funciona:** Cuenta cuántas interacciones totales tiene cada ítem (`item_counts`) en el entrenamiento. 
* **Ranking (`predict`):** Recomienda los ítems más populares a todos los usuarios.
* **Rating (`predict_rating`):** Predice basándose en la calificación promedio global del ítem.
* **Propósito:** Sus métricas de Precision y Recall suelen ser altas por puro sesgo de popularidad, pero fallan en personalización y novedad.

---

## 3. Modelos Individuales

### 3.1 Filtrado Colaborativo (CF)
* **Métrica de Similitud:** Usa Coseno o Pearson. Al usar nuestros diccionarios dispersos, el producto punto solo suma sobre los ítems comunes (`common_items`).
* **Predicción (`predict` / `predict_rating`):** Encuentra los $k$-NN (k vecinos más cercanos). Luego, pondera la calificación que esos vecinos le dieron al ítem.
* **Gestión del Cold-Start:** Si un usuario tiene muy pocas calificaciones, es difícil encontrarle vecinos. Por ello, si no puede predecir con los vecinos, retorna de forma inteligente la media histórica del usuario o del ítem en O(1).

### 3.2 Content-Based Filtering (CB)
* **Construcción del Perfil (`user_profile`):** El perfil del usuario es la suma ponderada del vector TF-IDF de los restaurantes que visitó. Se penalizan las calificaciones bajas centrando en 3.0 (ej. 1 estrella se vuelve -2.0 peso negativo).
* **Predicción:** Se calcula la **Similitud del Coseno** de forma vectorizada usando NumPy entre el perfil del usuario (array 1D) y la matriz `I_features` (array 2D).
* **Ventaja:** Es totalmente inmune al *cold-start* de nuevos restaurantes. 

---

## 4. Recomendador Híbrido
Fusiona las fortalezas del CF y del CB (un enfoque llamado *Weighted Hybrid*).

### Lógica de Combinación para el Ranking (`predict`)
Aquí hicimos cambios en los rangos numéricos para mezclar ambos mundos sin perder escala:
1. **Puntaje CF:** Originalmente en la escala de 1 a 5 estrellas.
2. **Min-Max Scaling:** El puntaje CF se normaliza al rango `[0, 1]`.
3. **Puntaje CB:** La similitud del Coseno devuelve un valor `[-1, 1]`. Se desplaza y escala a `[0, 1]` usando `(sim + 1) / 2`.
4. **Fusión Lineal:** 
   $$ \text{Score} = (W_{cf} \times \text{CF}_{norm}) + (W_{cb} \times \text{CB}_{norm}) $$

### Lógica de Combinación para el Rating (`predict_rating`)
Si el CF sufre un *cold-start* grave y devuelve `0.0`, el híbrido ignora su peso y **utiliza al 100% la predicción del Content-Based**. Si el CF pudo encontrar vecinos, se hace un promedio ponderado de las estrellas predichas.

---

## 5. Las Nuevas Métricas: Novedad y Diversidad

### 5.1 Novedad (Novelty)
Mide **qué tan poco convencionales o desconocidos** son los ítems recomendados. Penaliza fuertemente a los recomendadores de popularidad pura.
* **Fórmula:** Promedio de la Autoinformación de Shannon (o inversa logarítmica de la popularidad).
  $$ \text{Novelty} = \frac{1}{|R|} \sum_{i \in R} -\log_2(p_i) $$
  Donde $p_i$ es la probabilidad global del ítem (frecuencia / total usuarios). Ítems raros (con $p_i$ bajo) disparan el valor de Novedad porque $-\log(p)$ se vuelve muy alto.

### 5.2 Diversidad (Diversity)
Mide **qué tan variada** es la lista de recomendaciones.
* **Fórmula:** Distancia promedio intra-lista.
  $$ \text{Diversity} = \frac{2}{|R|(|R|-1)} \sum_{i=1}^{|R|} \sum_{j=i+1}^{|R|} (1 - \text{sim}(i, j)) $$
  Se compara el vector TF-IDF (características de NLP o metadata) de cada par de ítems recomendados. 

### ¿Cómo usar esto a tu favor?
En el reporte y la sustentación puedes afirmar: 
> *"A diferencia de implementaciones triviales usando scikit-learn o pandas que arrojan errores de Memoria RAM en datasets masivos, nuestro modelo indexa las interacciones en diccionarios anidados y sets de Hash de O(1), permitiendo escalar. Adicionalmente, demostramos mediante las métricas de Novedad y Diversidad que aunque el algoritmo Popular parece ser fuerte en Precision@K, se debe al **sesgo de popularidad**, mientras que el Híbrido no solo mitiga el cold-start, sino que enriquece el descubrimiento arrojando mayor novedad y listados semánticamente más diversos."*
