# Explicación Detallada del Código (Línea por Línea)

A continuación, desglosamos qué hace exactamente el código a nivel de programación, enfocándonos en las partes más complejas (Collaborative Filtering, Híbrido y Métricas).

---

## 1. Collaborative Filtering (`CollaborativeFiltering`)

### Método `fit(self, user_item_dict)`
Este método "entrena" el modelo, preparando las estructuras de datos.

```python
def fit(self, user_item_dict):
    self.user_item_dict = user_item_dict  
    # Guarda la matriz principal: qué ítems calificó cada usuario.
    
    self.item_user_dict = defaultdict(dict) 
    # Prepara un diccionario inverso para saber qué usuarios calificaron cada ítem.
    
    # -----------------------------------------------------
    # PRIMER FOR: Construir la matriz transpuesta
    # -----------------------------------------------------
    for u, items in user_item_dict.items():
        for i, r in items.items():
            self.item_user_dict[i][u] = r
            # Asignación: iteramos sobre cada usuario (u) y sus ítems calificados.
            # Por cada ítem (i) calificado con rating (r), lo guardamos al revés.
            # Esto nos servirá luego para encontrar vecinos rápidamente.
```

```python
    # -----------------------------------------------------
    # SEGUNDO FOR: Calcular medias y normas de usuarios
    # -----------------------------------------------------
    for u, items in user_item_dict.items():
        ratings = list(items.values()) # Extrae solo las estrellitas dadas por el usuario
        if not ratings: # Si el usuario no calificó nada (manejo de errores)
            self.user_norms[u] = 1e-9 # Usamos 1e-9 en vez de 0 para evitar divisiones por cero
            self.user_means[u] = 0.0
            continue
            
        self.user_means[u] = sum(ratings) / len(ratings) # Promedio aritmético
        
        if self.similarity_metric == 'cosine':
            # La magnitud del vector de calificaciones del usuario (Raíz cuadrada de la suma de los cuadrados)
            self.user_norms[u] = math.sqrt(sum(r**2 for r in ratings)) or 1e-9
        elif self.similarity_metric == 'pearson':
            # Para Pearson, centramos el vector (rating - media) antes de sacar la magnitud
            centered = [r - self.user_means[u] for r in ratings]
            self.user_norms[u] = math.sqrt(sum(c**2 for c in centered)) or 1e-9
```
*Hace exactamente lo mismo en un TERCER FOR para los ítems (`item_user_dict`), calculando sus medias y normas.*

### Método `_get_user_similarity(self, u1, u2)`
Calcula qué tan parecidos son dos usuarios.

```python
def _get_user_similarity(self, u1, u2):
    u1_ratings = self.user_item_dict.get(u1, {})
    u2_ratings = self.user_item_dict.get(u2, {})
    
    # Intersección de Sets: ¡Clave para la velocidad!
    # common_items es el conjunto de ítems que ambos han calificado.
    common_items = set(u1_ratings.keys()) & set(u2_ratings.keys())
    
    if not common_items: return 0.0 # Si no tienen nada en común, similitud es 0.
        
    if self.similarity_metric == 'cosine':
        # Producto punto: Multiplica los ratings que ambos le dieron a los ítems en común y los suma
        dot = sum(u1_ratings[i] * u2_ratings[i] for i in common_items)
        # Retorna el producto punto dividido por la multiplicación de sus normas (fórmula de similitud del Coseno)
        return dot / (self.user_norms.get(u1, 1e-9) * self.user_norms.get(u2, 1e-9))
```

### Método `_get_predictions(self, user_id, k_nn=10)`
Calcula el puntaje estimado para todos los ítems posibles para un usuario.

```python
def _get_predictions(self, user_id, k_nn=10):
    user_items = self.user_item_dict.get(user_id, {})
    candidate_users = set()
    
    # -----------------------------------------------------
    # Encontrar usuarios candidatos (vecinos potenciales)
    # -----------------------------------------------------
    for item in user_items: # Iteramos sobre restaurantes que visité
        # Añado a mi lista de candidatos a todos los usuarios que también visitaron esos restaurantes
        candidate_users.update(self.item_user_dict.get(item, {}).keys())
    candidate_users.discard(user_id) # Me borro a mí mismo de la lista
```

```python
    sims = {} # Diccionario para guardar similitudes
    for u in candidate_users:
        sim = self._get_user_similarity(user_id, u)
        if sim > 0:
            sims[u] = sim # Asignamos: el usuario 'u' tiene similitud 'sim'
    
    # Ordenamos el diccionario 'sims' por valor de mayor a menor y tomamos los top 'k_nn' (ej: los 10 mejores vecinos)
    nearest_neighbors = sorted(sims.keys(), key=lambda x: sims[x], reverse=True)[:k_nn]
    
    weighted_sum = defaultdict(float) # Numerador de la predicción
    sum_sim = defaultdict(float)      # Denominador de la predicción
    
    for nn in nearest_neighbors:
        w = sims[nn] # Peso (w) es la similitud de este vecino
        for item, rating in self.user_item_dict[nn].items(): # Recorremos lo que este vecino ha calificado
            weighted_sum[item] += w * rating # Numerador: peso * rating
            sum_sim[item] += w               # Denominador: suma de pesos
            
    predictions = {}
    for item in weighted_sum:
        # División final de la media ponderada
        predictions[item] = weighted_sum[item] / sum_sim[item]
    return predictions # Retorna las predicciones de estrellas para los ítems
```

---

## 2. Recomendador Híbrido (`HybridRecommender`)

### Método `_get_hybrid_scores(self, user_id, user_profile, k_nn=10)`
```python
def _get_hybrid_scores(self, user_id, user_profile, k_nn=10):
    # 1. Traemos las predicciones (en escala de estrellas) del Collaborative Filtering
    cf_preds_dict = self.cf._get_predictions(user_id, k_nn)
    
    num_items = self.cb.item_profiles.shape[0] # Total de ítems en el dataset
    cf_preds = np.zeros(num_items) # Creamos un array de ceros
    
    for item, score in cf_preds_dict.items():
        if item < num_items:
            cf_preds[item] = score # Trasladamos del diccionario al array en su respectivo índice
            
    # -----------------------------------------------------
    # NORMALIZACIÓN MIN-MAX (Escala el CF al rango de 0 a 1)
    # -----------------------------------------------------
    # Encontramos la predicción más baja y la más alta que arrojó el CF para este usuario
    min_cf = np.min(cf_preds[cf_preds > 0]) if np.any(cf_preds > 0) else 0.0
    max_cf = np.max(cf_preds) if np.max(cf_preds) > 0 else 1.0
    range_cf = max_cf - min_cf # Distancia entre el máximo y mínimo
    
    # np.where: si el puntaje original es > 0, lo normalizamos: (valor - min) / rango
    # Si era 0 (no se pudo predecir), se queda en 0.0
    cf_preds_norm = np.where(cf_preds > 0, (cf_preds - min_cf) / (range_cf + 1e-9), 0.0)
    
    # -----------------------------------------------------
    # PREDICCIÓN Y NORMALIZACIÓN DEL CONTENT-BASED
    # -----------------------------------------------------
    # Retorna similitud coseno de todos los ítems contra el user_profile (en rango -1 a 1)
    cb_preds = self.cb._get_similarities(user_profile) 
    
    # Lo pasamos de rango [-1, 1] al rango [0, 1] sumando 1 y dividiendo entre 2
    cb_preds_norm = (cb_preds + 1.0) / 2.0
    
    # Finalmente multiplicamos los arreglos normalizados por sus pesos configurados (ej: 0.85 y 0.15)
    hybrid_preds = (self.cf_weight * cf_preds_norm) + (self.cb_weight * cb_preds_norm)
    
    return hybrid_preds, cf_preds
```

---

## 3. Nuevas Métricas: Novedad y Diversidad

### Función `novelty(predicted, item_counts, num_users)`
```python
def novelty(predicted, item_counts, num_users):
    if len(predicted) == 0: return 0.0 # Chequeo de seguridad si el modelo no recomendó nada
    nov = 0.0 # Inicializamos el acumulador de novedad
    
    # Bucle por cada ítem que el modelo decidió recomendar en el Top-K
    for item in predicted:
        # Probabilidad (frecuencia): Cuántos usuarios calificaron este ítem entre el total de usuarios
        # get(item, 0) evita un KeyError si el ítem no fue calificado por nadie (algo imposible, pero previene bugs)
        p_i = item_counts.get(item, 0) / num_users 
        
        if p_i > 0:
            # -math.log2(p_i): Inversa logarítmica de la probabilidad. 
            # Valores bajitos de p_i generan números grandes (alta novedad).
            nov += -math.log2(p_i) 
            
    # Retornamos el promedio dividiendo entre el total de ítems recomendados (k)
    return nov / len(predicted) 
```

### Función `diversity(predicted, item_features)`
```python
def diversity(predicted, item_features):
    if len(predicted) < 2: return 0.0 # Necesitamos al menos 2 recomendaciones para medir qué tan diferentes son
    div = 0.0
    count = 0
    
    # Estos DOS BUCLES FOR anidados comparan TODOS los pares posibles en la lista.
    # Ej: Si recomiendo [A, B, C], comparará (A-B), (A-C) y (B-C) sin repetir.
    for i in range(len(predicted)):
        for j in range(i + 1, len(predicted)):
            item_i = predicted[i]
            item_j = predicted[j]
            
            # Verificación de que el ítem exista dentro del rango de features
            if item_i < item_features.shape[0] and item_j < item_features.shape[0]:
                vec_i = item_features[item_i] # Vector TF-IDF del ítem 1
                vec_j = item_features[item_j] # Vector TF-IDF del ítem 2
                
                # Normas de los vectores para la similitud Coseno
                norm_i = np.linalg.norm(vec_i)
                norm_j = np.linalg.norm(vec_j)
                
                if norm_i > 0 and norm_j > 0:
                    sim = np.dot(vec_i, vec_j) / (norm_i * norm_j) # Producto punto / normas
                else:
                    sim = 0.0
                
                # La "distancia" (diversidad) es el opuesto a la similitud (1 - sim)
                div += (1.0 - sim)
                count += 1
                
    if count == 0: return 0.0
    return div / count # Se devuelve la distancia promedio
```
