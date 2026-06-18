import numpy as np
from collections import defaultdict
import math

# ==========================================
# Baselines
# ==========================================

class RandomRecommender:
    def __init__(self):
        self.all_items = []
        self.user_item_dict = {}

    def fit(self, user_item_dict):
        self.user_item_dict = user_item_dict
        items_set = set()
        for items in user_item_dict.values():
            items_set.update(items.keys())
        self.all_items = list(items_set)

    def predict(self, user_id, top_k=5, candidates=None):
        user_seen = self.user_item_dict.get(user_id, {})
        pool = candidates if candidates is not None else self.all_items
        unseen_items = [item for item in pool if item not in user_seen]
        if not unseen_items:
            return []
        return list(np.random.choice(unseen_items, min(top_k, len(unseen_items)), replace=False))
    
    def predict_rating(self, user_id, item_id):
        return np.random.uniform(1, 5)

class PopularityRecommender:
    def __init__(self):
        self.popular_items = []
        self.item_avg_ratings = {}
        self.user_item_dict = {}
        
    def fit(self, user_item_dict):
        self.user_item_dict = user_item_dict
        # Calculate item counts and sums
        item_counts = defaultdict(int)
        item_sums = defaultdict(float)
        for items in user_item_dict.values():
            for item, rating in items.items():
                item_counts[item] += 1
                item_sums[item] += rating
                
        # Sort items by count descending
        sorted_items = sorted(item_counts.keys(), key=lambda x: item_counts[x], reverse=True)
        self.popular_items = sorted_items
        
        # Average rating for RMSE/MAE
        for item in item_counts:
            self.item_avg_ratings[item] = item_sums[item] / item_counts[item]

    def predict(self, user_id, user_item_dict=None, top_k=5, candidates=None):
        # Manejar la firma anterior del notebook: predict(u_id, U_I_train, top_k=K)
        # donde user_item_dict recibe el diccionario y top_k recibe el entero
        if isinstance(user_item_dict, int):
            top_k = user_item_dict
            user_item_dict = None
            
        if user_item_dict is not None:
            user_seen = user_item_dict.get(user_id, {})
        else:
            user_seen = self.user_item_dict.get(user_id, {}) if hasattr(self, 'user_item_dict') else {}
        
        # Si pasamos candidatos, filtramos la lista de popularidad global
        pool_items = self.popular_items
        if candidates is not None:
            candidates_set = set(candidates)
            pool_items = [item for item in self.popular_items if item in candidates_set]
            
        recs = []
        for item in pool_items:
            if item not in user_seen:
                recs.append(item)
            if len(recs) == top_k:
                break
        return recs
    
    def predict_rating(self, user_id, item_id):
        return self.item_avg_ratings.get(item_id, 3.0)

# ==========================================
# Collaborative Filtering
# ==========================================

class CollaborativeFiltering:
    def __init__(self, based='user', similarity_metric='cosine'):
        self.similarity_metric = similarity_metric
        self.based = based # 'user' or 'item'
        self.user_item_dict = {}
        self.item_user_dict = {}
        self.user_norms = {}
        self.item_norms = {}
        self.user_means = {}
        self.item_means = {}

    def fit(self, user_item_dict):
        self.user_item_dict = user_item_dict
        
        # Build item_user_dict
        self.item_user_dict = defaultdict(dict)
        for u, items in user_item_dict.items():
            for i, r in items.items():
                self.item_user_dict[i][u] = r
                
        # Calculate norms and means for users
        for u, items in user_item_dict.items():
            ratings = list(items.values())
            if not ratings:
                self.user_norms[u] = 1e-9
                self.user_means[u] = 0.0
                continue
            self.user_means[u] = sum(ratings) / len(ratings)
            
            if self.similarity_metric == 'cosine':
                self.user_norms[u] = math.sqrt(sum(r**2 for r in ratings)) or 1e-9
            elif self.similarity_metric == 'pearson':
                centered = [r - self.user_means[u] for r in ratings]
                self.user_norms[u] = math.sqrt(sum(c**2 for c in centered)) or 1e-9
                
        # Calculate norms and means for items
        for i, users in self.item_user_dict.items():
            ratings = list(users.values())
            if not ratings:
                self.item_norms[i] = 1e-9
                self.item_means[i] = 0.0
                continue
            self.item_means[i] = sum(ratings) / len(ratings)
            
            if self.similarity_metric == 'cosine':
                self.item_norms[i] = math.sqrt(sum(r**2 for r in ratings)) or 1e-9
            elif self.similarity_metric == 'pearson':
                centered = [r - self.item_means[i] for r in ratings]
                self.item_norms[i] = math.sqrt(sum(c**2 for c in centered)) or 1e-9
                
        return self

    def _get_user_similarity(self, u1, u2):
        u1_ratings = self.user_item_dict.get(u1, {})
        u2_ratings = self.user_item_dict.get(u2, {})
        common_items = set(u1_ratings.keys()) & set(u2_ratings.keys())
        if not common_items:
            return 0.0
            
        if self.similarity_metric == 'cosine':
            dot = sum(u1_ratings[i] * u2_ratings[i] for i in common_items)
            return dot / (self.user_norms.get(u1, 1e-9) * self.user_norms.get(u2, 1e-9))
        elif self.similarity_metric == 'pearson':
            m1 = self.user_means.get(u1, 0.0)
            m2 = self.user_means.get(u2, 0.0)
            dot = sum((u1_ratings[i] - m1) * (u2_ratings[i] - m2) for i in common_items)
            return dot / (self.user_norms.get(u1, 1e-9) * self.user_norms.get(u2, 1e-9))
        return 0.0

    def _get_item_similarity(self, i1, i2):
        i1_users = self.item_user_dict.get(i1, {})
        i2_users = self.item_user_dict.get(i2, {})
        common_users = set(i1_users.keys()) & set(i2_users.keys())
        if not common_users:
            return 0.0
            
        if self.similarity_metric == 'cosine':
            dot = sum(i1_users[u] * i2_users[u] for u in common_users)
            return dot / (self.item_norms.get(i1, 1e-9) * self.item_norms.get(i2, 1e-9))
        elif self.similarity_metric == 'pearson':
            m1 = self.item_means.get(i1, 0.0)
            m2 = self.item_means.get(i2, 0.0)
            dot = sum((i1_users[u] - m1) * (i2_users[u] - m2) for u in common_users)
            return dot / (self.item_norms.get(i1, 1e-9) * self.item_norms.get(i2, 1e-9))
        return 0.0

    def _get_predictions(self, user_id, k_nn=10):
        if self.based == 'user':
            user_items = self.user_item_dict.get(user_id, {})
            candidate_users = set()
            for item in user_items:
                candidate_users.update(self.item_user_dict.get(item, {}).keys())
            candidate_users.discard(user_id)
            
            if not candidate_users:
                return {}
                
            sims = {}
            for u in candidate_users:
                sim = self._get_user_similarity(user_id, u)
                if sim > 0:
                    sims[u] = sim
            
            nearest_neighbors = sorted(sims.keys(), key=lambda x: sims[x], reverse=True)[:k_nn]
            if not nearest_neighbors:
                return {}
                
            weighted_sum = defaultdict(float)
            sum_sim = defaultdict(float)
            for nn in nearest_neighbors:
                w = sims[nn]
                for item, rating in self.user_item_dict[nn].items():
                    weighted_sum[item] += w * rating
                    sum_sim[item] += w
                    
            predictions = {}
            for item in weighted_sum:
                predictions[item] = weighted_sum[item] / sum_sim[item]
            return predictions
        else: # item-based
            user_ratings = self.user_item_dict.get(user_id, {})
            if not user_ratings:
                return {}
                
            rated_items = list(user_ratings.keys())
            candidate_items = set()
            for item in rated_items:
                for u in self.item_user_dict.get(item, {}):
                    candidate_items.update(self.user_item_dict.get(u, {}).keys())
            for item in rated_items:
                candidate_items.discard(item)
                
            predictions = {}
            for item in candidate_items:
                sims = {}
                for ri in rated_items:
                    sim = self._get_item_similarity(item, ri)
                    if sim > 0:
                        sims[ri] = sim
                if not sims:
                    continue
                nearest_rated = sorted(sims.keys(), key=lambda x: sims[x], reverse=True)[:k_nn]
                if not nearest_rated:
                    continue
                weighted_sum = sum(sims[ri] * user_ratings[ri] for ri in nearest_rated)
                sum_sim = sum(sims[ri] for ri in nearest_rated)
                if sum_sim > 0:
                    predictions[item] = weighted_sum / sum_sim
            return predictions

    def predict(self, target_id, top_k=5, k_nn=10):
        predictions = self._get_predictions(target_id, k_nn)
        user_seen = self.user_item_dict.get(target_id, {})
        preds_filtered = {item: val for item, val in predictions.items() if item not in user_seen}
        top_items = sorted(preds_filtered.keys(), key=lambda x: preds_filtered[x], reverse=True)[:top_k]
        return top_items

    def predict_rating(self, target_id, item_id, k_nn=10):
        predictions = self._get_predictions(target_id, k_nn)
        pred = predictions.get(item_id, 0.0)
        if pred > 0:
            return pred
        
        # Fallback dinámico e inteligente en caso de cold start o alta dispersión
        user_mean = self.user_means.get(target_id, 0.0)
        if user_mean > 0:
            return user_mean
            
        item_mean = self.item_means.get(item_id, 0.0)
        if item_mean > 0:
            return item_mean
            
        return 3.0

# ==========================================
# Content-Based & TF-IDF
# ==========================================

class TFIDF:
    def __init__(self):
        self.vocab = {}
        self.idf = {}
        
    def fit_transform(self, corpus):
        # Corpus is a list of lists of strings (tokens)
        N = len(corpus)
        df = defaultdict(int)
        
        # Build vocab and DF
        for doc in corpus:
            unique_tokens = set(doc)
            for token in unique_tokens:
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)
                df[token] += 1
                
        # Calculate IDF
        for token, freq in df.items():
            self.idf[token] = math.log((1 + N) / (1 + freq)) + 1
            
        # Calculate TF-IDF matrix
        tfidf_matrix = np.zeros((N, len(self.vocab)))
        for i, doc in enumerate(corpus):
            tf = defaultdict(int)
            for token in doc:
                tf[token] += 1
            for token, count in tf.items():
                if token in self.vocab:
                    tfidf_matrix[i, self.vocab[token]] = count * self.idf[token]
                    
        # Normalize
        norms = np.linalg.norm(tfidf_matrix, axis=1)
        norms[norms == 0] = 1e-9
        return tfidf_matrix / norms[:, np.newaxis]

class ContentBasedFiltering:
    def __init__(self):
        self.item_profiles = None
    
    def fit(self, item_features):
        self.item_profiles = item_features
        return self

    def predict(self, user_profile, top_k=5):
        sim = self._get_similarities(user_profile)
        top_items = np.argsort(sim)[::-1][:top_k]
        return top_items

    def _get_similarities(self, user_profile):
        norms_items = np.linalg.norm(self.item_profiles, axis=1)
        norms_items[norms_items == 0] = 1e-9
        norm_user = np.linalg.norm(user_profile)
        if norm_user == 0: norm_user = 1e-9
        return np.dot(self.item_profiles, user_profile) / (norms_items * norm_user)

    def predict_rating(self, user_profile, item_id):
        sim = self._get_similarities(user_profile)
        # Scale similarity [-1, 1] to a rating [1, 5] roughly
        s = sim[item_id]
        rating = 3.0 + (s * 2.0)
        return min(max(rating, 1.0), 5.0)

# ==========================================
# Hybrid Combination
# ==========================================

class HybridRecommender:
    def __init__(self, cf_weight=0.5, cb_weight=0.5,based='user', similarity_metric='pearson'):
        self.cf_weight = cf_weight
        self.cb_weight = cb_weight
        self.cf = CollaborativeFiltering(based=based, similarity_metric=similarity_metric)
        self.cb = ContentBasedFiltering()
        self.user_item_dict = {}

    def fit(self, user_item_dict, item_features):
        self.user_item_dict = user_item_dict
        self.cf.fit(user_item_dict)
        self.cb.fit(item_features)
        
    def _get_hybrid_scores(self, user_id, user_profile, k_nn=10):
        cf_preds_dict = self.cf._get_predictions(user_id, k_nn)
        num_items = self.cb.item_profiles.shape[0]
        cf_preds = np.zeros(num_items)
        
        for item, score in cf_preds_dict.items():
            if item < num_items:
                cf_preds[item] = score
        
        # --- CORRECCIÓN DE RANGOS (MIN-MAX SCALING REAL) ---
        # Escalamos CF al rango [0, 1] considerando solo valores mayores a cero
        min_cf = np.min(cf_preds[cf_preds > 0]) if np.any(cf_preds > 0) else 0.0
        max_cf = np.max(cf_preds) if np.max(cf_preds) > 0 else 1.0
        range_cf = max_cf - min_cf
        cf_preds_norm = np.where(cf_preds > 0, (cf_preds - min_cf) / (range_cf + 1e-9), 0.0)
        
        # Obtener Content-Based predictions (similitud coseno)
        cb_preds = self.cb._get_similarities(user_profile)
        # Escalamos CB de [-1, 1] al rango [0, 1] para que sea equivalente al CF normalizado
        cb_preds_norm = (cb_preds + 1.0) / 2.0
        
        # Combinación lineal idónea para ranking
        hybrid_preds = (self.cf_weight * cf_preds_norm) + (self.cb_weight * cb_preds_norm)
        return hybrid_preds, cf_preds

    def predict(self, user_id, user_profile, top_k=5, k_nn=10):
        hybrid_preds, _ = self._get_hybrid_scores(user_id, user_profile, k_nn)
        user_seen = self.user_item_dict.get(user_id, {})
        for item in user_seen:
            if item < len(hybrid_preds):
                hybrid_preds[item] = -np.inf
        return np.argsort(hybrid_preds)[::-1][:top_k]

    def predict_rating(self, user_id, user_profile, item_id, k_nn=10):
        # 1. Obtener predicción del Collaborative Filtering (Puntuación base de estrellas)
        _, cf_preds = self._get_hybrid_scores(user_id, user_profile, k_nn)
        cf_rating = cf_preds[item_id] if item_id < len(cf_preds) else 0.0
        
        # 2. Obtener predicción del Content Based (Ajustado al rango [1, 5])
        cb_rating = self.cb.predict_rating(user_profile, item_id)
        
        # --- COMBINACIÓN CORREGIDA PARA REGRESIÓN ---
        if cf_rating > 0:
            # Si hay vecinos, fusionamos linealmente ambas predicciones de estrellas reales
            pred = (self.cf_weight * cf_rating) + (self.cb_weight * cb_rating)
        else:
            # Si el CF sufre cold-start, el Content-Based salva el día (¡No más medias globales genéricas!)
            pred = cb_rating
            
        return min(max(pred, 1.0), 5.0)
# ==========================================
# Metrics
# ==========================================

def precision_at_k(actual, predicted, k):
    act_set = set(actual)
    pred_set = set(predicted[:k])
    if len(pred_set) == 0: return 0.0
    return len(act_set & pred_set) / float(k)

def recall_at_k(actual, predicted, k):
    act_set = set(actual)
    pred_set = set(predicted[:k])
    if len(act_set) == 0: return 0.0
    return len(act_set & pred_set) / float(len(act_set))

def ndcg_at_k(actual, predicted, k):
    dcg = 0.0
    idcg = 0.0
    for i, p in enumerate(predicted[:k]):
        if p in actual:
            dcg += 1.0 / math.log2(i + 2)
    for i in range(min(k, len(actual))):
        idcg += 1.0 / math.log2(i + 2)
    if idcg == 0.0: return 0.0
    return dcg / idcg

def rmse(actual_ratings, predicted_ratings):
    if len(actual_ratings) == 0: return 0.0
    return np.sqrt(np.mean((np.array(actual_ratings) - np.array(predicted_ratings))**2))

def mae(actual_ratings, predicted_ratings):
    if len(actual_ratings) == 0: return 0.0
    return np.mean(np.abs(np.array(actual_ratings) - np.array(predicted_ratings)))

def novelty(predicted, item_counts, num_users):
    if len(predicted) == 0: return 0.0
    nov = 0.0
    for item in predicted:
        p_i = item_counts.get(item, 0) / num_users
        if p_i > 0:
            nov += -math.log2(p_i)
    return nov / len(predicted)

def diversity(predicted, item_features):
    if len(predicted) < 2: return 0.0
    div = 0.0
    count = 0
    for i in range(len(predicted)):
        for j in range(i + 1, len(predicted)):
            item_i = predicted[i]
            item_j = predicted[j]
            if item_i < item_features.shape[0] and item_j < item_features.shape[0]:
                vec_i = item_features[item_i]
                vec_j = item_features[item_j]
                norm_i = np.linalg.norm(vec_i)
                norm_j = np.linalg.norm(vec_j)
                if norm_i > 0 and norm_j > 0:
                    sim = np.dot(vec_i, vec_j) / (norm_i * norm_j)
                else:
                    sim = 0.0
                div += (1.0 - sim)
                count += 1
    if count == 0: return 0.0
    return div / count
