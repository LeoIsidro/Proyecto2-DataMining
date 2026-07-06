# Alertas Éticas, de Equidad y Escalabilidad

## 1. Sesgos (Bias) en Recomendaciones
* **Burbujas de Filtro:** El filtrado colaborativo puede reforzar los patrones de consumo existentes, aislando al usuario y evitando que descubra nuevos nichos de mercado (ej. favorecer exclusivamente los locales grandes y populares sobre pequeñas empresas emergentes).
* **Sesgo de Popularidad:** Las métricas de similitud (como el Coseno) tienden a sobrevalorar ítems con demasiadas reseñas. Esto se refleja directamente en las *Authorities* de los algoritmos de grafos y en el CF. Se recomienda penalizar los ítems "blockbuster" para mejorar la diversidad.
* **Cold-Start y Equidad Competitiva:** Los negocios recién abiertos, al no tener grafo de interacciones, serán penalizados. El módulo de Contenido (CB) mitiga esto, pero su peso ($cb\_weight$) debe calibrarse y ser transparente.

## 2. Escalabilidad de Algoritmos Nativos
* La matriz de similitud de usuarios ($N \times N$) puede alcanzar proporciones intratables para millones de nodos.
* **Alertas Computacionales:** Es vital aplicar muestreo jerárquico (*Reservoir Sampling*), o estructuras como **Count-Min Sketch** si se introducen flujos de datos en streaming continuos, para reducir el footprint de memoria (como se menciona en el módulo de Streaming).
* Las implementaciones matriciales actuales con NumPy escalarán bien hasta unos cientos de miles de registros dependiendo de la memoria RAM, pero deben particionarse o usar PySpark para el despliegue final sobre Big Data masivo real.

## 3. Equidad Demográfica
* Las agrupaciones generadas por **DBSCAN** y **CURE** (y las comunidades de **Louvain**) podrían discriminar indirectamente áreas geográficas de bajo nivel socioeconómico si hay un desbalance en la cantidad de reseñas por ciudad, reduciendo la visibilidad algorítmica de ciertas poblaciones.
* El análisis de resultados debe incluir una auditoría para garantizar que las recomendaciones cruzadas son balanceadas.
