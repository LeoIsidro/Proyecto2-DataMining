import numpy as np
import math
from collections import defaultdict

# ==========================================
# Métricas de Evaluación
# ==========================================

def compute_silhouette(X, labels):
    n = X.shape[0]
    if len(np.unique(labels)) < 2:
        return -1
    
    silhouettes = []
    for i in range(n):
        c = labels[i]
        if c == -1:
            continue
        
        cluster_pts = X[labels == c]
        if len(cluster_pts) > 1:
            a_i = np.mean(np.linalg.norm(cluster_pts - X[i], axis=1))
        else:
            a_i = 0
            
        b_i = np.inf
        for other_c in np.unique(labels):
            if other_c == c or other_c == -1:
                continue
            other_pts = X[labels == other_c]
            dist = np.mean(np.linalg.norm(other_pts - X[i], axis=1))
            if dist < b_i:
                b_i = dist
                
        s_i = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0
        silhouettes.append(s_i)
        
    return np.mean(silhouettes)

def purity_score(y_true, y_pred):
    clusters = np.unique(y_pred)
    total_correct = 0
    for c in clusters:
        if c == -1:
            continue
        idx = np.where(y_pred == c)[0]
        if len(idx) == 0: continue
        true_labels = y_true[idx]
        counts = np.bincount(true_labels)
        total_correct += np.max(counts)
    return total_correct / len(y_true)

def entropy(labels):
    counts = np.bincount(labels)
    probs = counts[counts > 0] / len(labels)
    return -np.sum(probs * np.log(probs))

def normalized_mutual_information(y_true, y_pred):
    # Ignoramos ruido para NMI
    mask = y_pred != -1
    if np.sum(mask) == 0:
        return 0.0
    y_t = y_true[mask]
    y_p = y_pred[mask]
    
    # Mutual Information
    n = len(y_t)
    classes = np.unique(y_t)
    clusters = np.unique(y_p)
    mi = 0.0
    
    for c in clusters:
        for cls in classes:
            p_c = np.sum(y_p == c) / n
            p_cls = np.sum(y_t == cls) / n
            p_c_cls = np.sum((y_p == c) & (y_t == cls)) / n
            if p_c_cls > 0:
                mi += p_c_cls * np.log(p_c_cls / (p_c * p_cls))
                
    h_true = entropy(y_t)
    h_pred = entropy(y_p)
    
    if h_true + h_pred == 0:
        return 0.0
    return 2.0 * mi / (h_true + h_pred)

def compute_k_distance(X, k):
    distances = []
    for i in range(X.shape[0]):
        dist = np.linalg.norm(X - X[i], axis=1)
        dist.sort()
        if len(dist) > k:
            distances.append(dist[k])
    distances.sort(reverse=True)
    return distances

# ==========================================
# Algoritmos de Clustering
# ==========================================

class KMeansPlusPlus:
    # 1. Agregamos n_init y random_state a la inicialización
    def __init__(self, k=3, max_iters=100, tol=1e-4, n_init=10, random_state=None):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None
        self.inertia_ = float('inf') # Inicializamos con infinito
        
    def _initialize_centroids(self, X):
        n_samples = X.shape[0]
        centroids = [X[np.random.randint(n_samples)]]
        
        for _ in range(1, self.k):
            dist_sq = np.array([min([np.inner(c-x, c-x) for c in centroids]) for x in X])
            probs = dist_sq / dist_sq.sum()
            cumulative_probs = probs.cumsum()
            r = np.random.rand()
            for j, p in enumerate(cumulative_probs):
                if r < p:
                    centroids.append(X[j])
                    break
        return np.array(centroids)

    def fit(self, X):
        # 2. Fijamos la semilla si fue proporcionada
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        best_centroids = None
        best_labels = None
        best_inertia = float('inf')

        # 3. Bucle externo para ejecutar el algoritmo n_init veces
        for init_run in range(self.n_init):
            centroids = self._initialize_centroids(X)
            
            for _ in range(self.max_iters):
                dists = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
                labels = np.argmin(dists, axis=1)
                
                new_centroids = np.array([X[labels == i].mean(axis=0) if np.sum(labels == i) > 0 else centroids[i] for i in range(self.k)])
                
                if np.all(np.linalg.norm(new_centroids - centroids, axis=1) < self.tol):
                    break
                centroids = new_centroids
                
            # Calcular inercia de esta corrida específica
            dists = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
            current_inertia = np.sum(np.min(dists, axis=1)**2)
            
            # 4. Guardar los resultados solo si mejoran la inercia anterior
            if current_inertia < best_inertia:
                best_inertia = current_inertia
                best_centroids = centroids
                best_labels = labels

        # Asignar los mejores resultados encontrados como los finales del modelo
        self.centroids = best_centroids
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        
        return self

class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None

    def fit(self, X):
        n = X.shape[0]
        self.labels_ = -np.ones(n, dtype=int)
        cluster_id = 0
        
        for i in range(n):
            if self.labels_[i] != -1:
                continue
            
            diff = X - X[i]
            dist = np.linalg.norm(diff, axis=1)
            neighbors = np.where(dist <= self.eps)[0]
            
            if len(neighbors) < self.min_samples:
                self.labels_[i] = -1 # Noise/Outlier
            else:
                self._expand_cluster(X, i, neighbors, cluster_id)
                cluster_id += 1
        return self

    def _expand_cluster(self, X, i, neighbors, cluster_id):
        self.labels_[i] = cluster_id
        queue = list(neighbors)
        
        while queue:
            p = queue.pop(0)
            if self.labels_[p] == -1:
                self.labels_[p] = cluster_id
                
            if self.labels_[p] >= 0:
                continue 
                
            self.labels_[p] = cluster_id
            
            diff = X - X[p]
            dist = np.linalg.norm(diff, axis=1)
            p_neighbors = np.where(dist <= self.eps)[0]
            
            if len(p_neighbors) >= self.min_samples:
                for n in p_neighbors:
                    if self.labels_[n] < 0:
                        queue.append(n)


import numpy as np

class ClusterCURE:
    def __init__(self, point, idx=None):
        self.points = np.array([point])
        self.indices = [idx] if idx is not None else []
        self.mean = np.array(point)
        self.rep = np.array([point])

    def merge(self, other_cluster, c, alpha):
        self.points = np.vstack((self.points, other_cluster.points))
        self.indices.extend(other_cluster.indices)
        self.mean = np.mean(self.points, axis=0)
        self._generate_representatives(c, alpha)

    def _generate_representatives(self, c, alpha):
        n_points = len(self.points)
        if n_points <= c:
            self.rep = self.points.copy()
        else:
            selected_reps = []
            dist_to_mean = np.linalg.norm(self.points - self.mean, axis=1)
            first_rep_idx = np.argmax(dist_to_mean)
            selected_reps.append(self.points[first_rep_idx])
            
            for _ in range(1, c):
                reps_arr = np.array(selected_reps)
                dist_to_reps = np.linalg.norm(self.points[:, np.newaxis] - reps_arr, axis=2)
                min_dists = np.min(dist_to_reps, axis=1)
                next_rep_idx = np.argmax(min_dists)
                selected_reps.append(self.points[next_rep_idx])
                
            self.rep = np.array(selected_reps)
            
        for i in range(len(self.rep)):
            self.rep[i] = self.rep[i] + alpha * (self.mean - self.rep[i])


class CURE:
    def __init__(self, k=2, c=3, alpha=0.3, sample_size=300):
        self.k = k
        self.c = c
        self.alpha = alpha
        self.sample_size = sample_size
        self.labels_ = None
        self.clusters_ = []

    def fit(self, X):
        n = X.shape[0]
        self.labels_ = -np.ones(n, dtype=int)
        
        use_sampling = n > self.sample_size
        if use_sampling:
            indices = np.random.choice(n, self.sample_size, replace=False)
        else:
            indices = np.arange(n)
            
        sample_data = X[indices]
        clusters = [ClusterCURE(pt, idx) for pt, idx in zip(sample_data, indices)]
        
        # OPTIMIZACIÓN 1: Matriz de distancias en caché (Reduce O(N^3) a O(N^2))
        n_c = len(clusters)
        dist_mat = np.full((n_c, n_c), np.inf)
        
        # Calcular matriz inicial
        for i in range(n_c):
            for j in range(i + 1, n_c):
                dists = np.linalg.norm(clusters[i].rep[:, np.newaxis] - clusters[j].rep, axis=2)
                min_dist = np.min(dists)
                dist_mat[i, j] = min_dist
                dist_mat[j, i] = min_dist
        
        while len(clusters) > self.k:
            # Encontrar el par más cercano en O(1) usando la matriz
            idx1, idx2 = np.unravel_index(np.argmin(dist_mat), dist_mat.shape)
            
            # Asegurar que idx1 < idx2 para sacar el mayor primero sin alterar el índice del menor
            if idx1 > idx2: idx1, idx2 = idx2, idx1
            
            cluster2 = clusters.pop(idx2)
            cluster1 = clusters[idx1]
            cluster1.merge(cluster2, self.c, self.alpha)
            
            # OPTIMIZACIÓN: Actualizar la matriz borrando la fila/columna de cluster2
            dist_mat = np.delete(dist_mat, idx2, axis=0)
            dist_mat = np.delete(dist_mat, idx2, axis=1)
            
            # Recalcular distancias SOLO para cluster1 (el fusionado)
            dist_mat[idx1, :] = np.inf
            dist_mat[:, idx1] = np.inf
            for j in range(len(clusters)):
                if idx1 == j: continue
                dists = np.linalg.norm(clusters[idx1].rep[:, np.newaxis] - clusters[j].rep, axis=2)
                min_d = np.min(dists)
                dist_mat[idx1, j] = min_d
                dist_mat[j, idx1] = min_d
                
        self.clusters_ = clusters
        
        for c_id, cluster in enumerate(self.clusters_):
            for idx in cluster.indices:
                self.labels_[idx] = c_id
                
        if use_sampling:
            remaining_indices = np.setdiff1d(np.arange(n), indices)
            for idx in remaining_indices:
                pt = X[idx]
                min_dist = float('inf')
                best_c_id = -1
                for c_id, cluster in enumerate(self.clusters_):
                    dist_to_reps = np.linalg.norm(cluster.rep - pt, axis=1)
                    dist = np.min(dist_to_reps)
                    if dist < min_dist:
                        min_dist = dist
                        best_c_id = c_id
                self.labels_[idx] = best_c_id
                
        return self


class BFR:
    def __init__(self, k=3, ds_threshold=2.0, cs_threshold=1.5, chunk_size=100, merge_threshold=0.1, promote_cs=True):
        self.k = k
        self.ds_threshold = ds_threshold
        self.cs_threshold = cs_threshold
        self.chunk_size = chunk_size
        self.merge_threshold = merge_threshold
        self.promote_cs = promote_cs
        self.ds_stats = {} 
        self.cs_stats = {} 
        self.rs = [] 
        self.labels_ = []
        self.cs_id_counter = 0
        
    def _mahalanobis_dist(self, point, stats):
        N = stats['N']
        if N <= 1: return float('inf')
        centroid = stats['SUM'] / N
        variance = (stats['SUMSQ'] / N) - (centroid ** 2)
        variance[variance <= 0] = 1e-6
        return np.sqrt(np.sum(((point - centroid) ** 2) / variance))
        
    def _centroid(self, stats):
        return stats['SUM'] / stats['N']
        
    def _add_point_to_stat(self, stats, point):
        stats['N'] += 1
        stats['SUM'] += point
        stats['SUMSQ'] += point**2
        
    def _create_stat(self, points):
        points = np.array(points)
        return {
            'N': len(points),
            'SUM': np.sum(points, axis=0),
            'SUMSQ': np.sum(points**2, axis=0)
        }

    def _merge_cs_stats(self, stat1, stat2):
        return {
            'N': stat1['N'] + stat2['N'],
            'SUM': stat1['SUM'] + stat2['SUM'],
            'SUMSQ': stat1['SUMSQ'] + stat2['SUMSQ']
        }
        
    def fit(self, X):
        n, dim = X.shape
        self.labels_ = -np.ones(n, dtype=int)
        
        sample_size = min(self.chunk_size, n)
        if sample_size < self.k:
            return self
            
        # IMPORTANTE: En tu script general asegúrate de importar tu KMeansPlusPlus
        from implementaciones.clustering import KMeansPlusPlus
        
        kmeans = KMeansPlusPlus(k=self.k)
        kmeans.fit(X[:sample_size])
        
        for i in range(self.k):
            cluster_idx = np.where(kmeans.labels_ == i)[0]
            cluster_pts = X[:sample_size][cluster_idx]
            if len(cluster_pts) > 0:
                self.ds_stats[i] = self._create_stat(cluster_pts)
                self.labels_[cluster_idx] = i
                
        # OPTIMIZACIÓN 2: Umbral de mantenimiento de RS mucho más permisivo
        rs_maintenance_threshold = max(50, self.chunk_size // 2) 
                
        for i in range(sample_size, n):
            point = X[i]
            
            best_ds = -1
            min_ds_dist = float('inf')
            for c_id, stats in self.ds_stats.items():
                dist = self._mahalanobis_dist(point, stats)
                if dist < min_ds_dist:
                    min_ds_dist = dist
                    best_ds = c_id
                    
            if min_ds_dist < self.ds_threshold:
                self._add_point_to_stat(self.ds_stats[best_ds], point)
                self.labels_[i] = best_ds
                continue
                
            best_cs = -1
            min_cs_dist = float('inf')
            for c_id, stats in self.cs_stats.items():
                dist = self._mahalanobis_dist(point, stats)
                if dist < min_cs_dist:
                    min_cs_dist = dist
                    best_cs = c_id
                    
            if best_cs != -1 and min_cs_dist < self.cs_threshold:
                self._add_point_to_stat(self.cs_stats[best_cs], point)
                self.labels_[i] = -2 
                continue
                
            self.rs.append(i)
            self.labels_[i] = -1
            
            # Ejecutamos KMeans en el ruido (RS) SOLAMENTE cuando hay un bloque significativo
            if len(self.rs) >= rs_maintenance_threshold:
                rs_pts = X[self.rs]
                k_rs = min(3, len(self.rs) // 10) # Limitar la creación excesiva de micro-clústeres
                
                # Prevenir fallo de K-Means si k_rs es 0 o muy pequeño
                if k_rs > 1:
                    km_rs = KMeansPlusPlus(k=k_rs)
                    km_rs.fit(rs_pts)
                    
                    new_rs = []
                    for c_id in range(k_rs):
                        cluster_idx = np.where(km_rs.labels_ == c_id)[0]
                        if len(cluster_idx) >= 3: # Promover si tiene al menos 3 puntos (más estricto)
                            cs_id = self.cs_id_counter
                            self.cs_id_counter += 1
                            global_idx = [self.rs[idx] for idx in cluster_idx]
                            self.cs_stats[cs_id] = self._create_stat(X[global_idx])
                            for idx in global_idx:
                                self.labels_[idx] = -2
                        else:
                            new_rs.extend([self.rs[idx] for idx in cluster_idx])
                    self.rs = new_rs
                
            merged = True
            while merged and len(self.cs_stats) > 1:
                merged = False
                cs_keys = list(self.cs_stats.keys())
                for idx1 in range(len(cs_keys)):
                    for idx2 in range(idx1 + 1, len(cs_keys)):
                        k1, k2 = cs_keys[idx1], cs_keys[idx2]
                        if k1 not in self.cs_stats or k2 not in self.cs_stats:
                            continue
                        c1 = self._centroid(self.cs_stats[k1])
                        c2 = self._centroid(self.cs_stats[k2])
                        dist = np.linalg.norm(c1 - c2)
                        if dist < self.merge_threshold:
                            self.cs_stats[k1] = self._merge_cs_stats(self.cs_stats[k1], self.cs_stats[k2])
                            del self.cs_stats[k2]
                            merged = True
                            break
                    if merged: break
                            
            cs_keys = list(self.cs_stats.keys())
            for c_id in cs_keys:
                c_centroid = self._centroid(self.cs_stats[c_id])
                best_ds = -1
                min_ds_dist = float('inf')
                for ds_id, ds_stat in self.ds_stats.items():
                    dist = self._mahalanobis_dist(c_centroid, ds_stat)
                    if dist < min_ds_dist:
                        min_ds_dist = dist
                        best_ds = ds_id
                if best_ds != -1 and min_ds_dist < self.ds_threshold:
                    self.ds_stats[best_ds] = self._merge_cs_stats(self.ds_stats[best_ds], self.cs_stats[c_id])
                    del self.cs_stats[c_id]
                    
        if self.promote_cs:
            cs_keys = list(self.cs_stats.keys())
            for c_id in cs_keys:
                stat = self.cs_stats[c_id]
                if stat['N'] >= 5:
                    c_centroid = self._centroid(stat)
                    min_ds_dist = min([self._mahalanobis_dist(c_centroid, ds_stat) for ds_stat in self.ds_stats.values()] + [float('inf')])
                    if min_ds_dist >= self.ds_threshold:
                        ds_id = len(self.ds_stats)
                        self.ds_stats[ds_id] = stat
                        del self.cs_stats[c_id]
                        
        cs_keys = list(self.cs_stats.keys())
        for c_id in cs_keys:
            c_centroid = self._centroid(self.cs_stats[c_id])
            best_ds = -1
            min_ds_dist = float('inf')
            for ds_id, ds_stat in self.ds_stats.items():
                dist = self._mahalanobis_dist(c_centroid, ds_stat)
                if dist < min_ds_dist:
                    min_ds_dist = dist
                    best_ds = ds_id
            if best_ds != -1:
                self.ds_stats[best_ds] = self._merge_cs_stats(self.ds_stats[best_ds], self.cs_stats[c_id])
                del self.cs_stats[c_id]
                
        for i in range(n):
            if self.labels_[i] != -1:
                best_ds = -1
                min_ds_dist = float('inf')
                for ds_id, ds_stat in self.ds_stats.items():
                    dist = self._mahalanobis_dist(X[i], ds_stat)
                    if dist < min_ds_dist:
                        min_ds_dist = dist
                        best_ds = ds_id
                self.labels_[i] = best_ds
                
        return self