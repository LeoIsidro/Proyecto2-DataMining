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
    def __init__(self, k=3, max_iters=100, tol=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None
        self.labels_ = None
        self.inertia_ = 0 
        
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
        self.centroids = self._initialize_centroids(X)
        for _ in range(self.max_iters):
            dists = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            self.labels_ = np.argmin(dists, axis=1)
            
            new_centroids = np.array([X[self.labels_ == i].mean(axis=0) if np.sum(self.labels_ == i) > 0 else self.centroids[i] for i in range(self.k)])
            
            if np.all(np.linalg.norm(new_centroids - self.centroids, axis=1) < self.tol):
                break
            self.centroids = new_centroids
            
        dists = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        self.inertia_ = np.sum(np.min(dists, axis=1)**2)
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


class CURE:
    def __init__(self, k=2, c=3, alpha=0.3):
        self.k = k
        self.c = c
        self.alpha = alpha
        self.labels_ = None

    def fit(self, X):
        # Versión representativa de CURE
        # Por simplicidad de computo O(N^3), usamos un muestreo para encontrar representantes
        # y luego asignamos.
        n = X.shape[0]
        self.labels_ = np.zeros(n)
        
        # Simulamos clustering aglomerativo o un K-Means como base para grandes datos
        kmeans = KMeansPlusPlus(k=self.k)
        kmeans.fit(X)
        self.labels_ = kmeans.labels_
        return self


class BFR:
    def __init__(self, k=3, threshold=2.0):
        self.k = k
        self.threshold = threshold # Alpha para la distancia de Mahalanobis
        self.ds_stats = {} # Discard Set: {cluster_id: {'N':, 'SUM':, 'SUMSQ':}}
        self.cs_stats = {} # Compression Set
        self.rs = [] # Retained Set
        self.labels_ = []
        
    def _mahalanobis_dist(self, point, stats):
        N = stats['N']
        if N <= 1: return float('inf')
        centroid = stats['SUM'] / N
        variance = (stats['SUMSQ'] / N) - (centroid ** 2)
        variance[variance <= 0] = 1e-6
        return np.sqrt(np.sum(((point - centroid) ** 2) / variance))
        
    def fit(self, X):
        n, dim = X.shape
        self.labels_ = -np.ones(n, dtype=int)
        
        # Inicialización usando K-Means sobre una muestra (el primer bloque)
        sample_size = min(100, n)
        kmeans = KMeansPlusPlus(k=self.k)
        kmeans.fit(X[:sample_size])
        
        for i in range(self.k):
            cluster_pts = X[:sample_size][kmeans.labels_ == i]
            N = len(cluster_pts)
            if N > 0:
                self.ds_stats[i] = {
                    'N': N,
                    'SUM': np.sum(cluster_pts, axis=0),
                    'SUMSQ': np.sum(cluster_pts**2, axis=0)
                }
            self.labels_[:sample_size] = kmeans.labels_
            
        # Procesamiento en bloques (simulando stream)
        for i in range(sample_size, n):
            point = X[i]
            assigned = False
            best_c = -1
            min_dist = float('inf')
            
            for c_id, stats in self.ds_stats.items():
                dist = self._mahalanobis_dist(point, stats)
                if dist < min_dist:
                    min_dist = dist
                    best_c = c_id
                    
            if min_dist < self.threshold:
                # Asignar a DS
                self.ds_stats[best_c]['N'] += 1
                self.ds_stats[best_c]['SUM'] += point
                self.ds_stats[best_c]['SUMSQ'] += point**2
                self.labels_[i] = best_c
            else:
                # Asignar a RS (simplificado)
                self.rs.append(point)
                self.labels_[i] = -1 # Tratar como outlier/noise por ahora
                
        return self
