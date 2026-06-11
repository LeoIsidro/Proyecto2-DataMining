import numpy as np
from collections import defaultdict
import math

# ==========================================
# Baselines
# ==========================================

class RandomRecommender:
    def __init__(self):
        self.n_items = 0
    def fit(self, user_item_matrix):
        self.n_items = user_item_matrix.shape[1]
    def predict(self, user_id, top_k=5):
        return np.random.choice(self.n_items, top_k, replace=False)

class PopularityRecommender:
    def __init__(self):
        self.popular_items = []
    def fit(self, user_item_matrix):
        # Items with most interactions
        counts = np.sum(user_item_matrix > 0, axis=0)
        self.popular_items = np.argsort(counts)[::-1]
    def predict(self, user_id, user_item_matrix, top_k=5):
        user_seen = user_item_matrix[user_id] > 0
        recs = []
        for item in self.popular_items:
            if not user_seen[item]:
                recs.append(item)
            if len(recs) == top_k:
                break
        return recs

# ==========================================
# Collaborative Filtering
# ==========================================

class CollaborativeFiltering:
    def __init__(self, based='user', similarity_metric='cosine'):
        self.similarity_metric = similarity_metric
        self.based = based # 'user' or 'item'
        self.matrix = None
        self.similarity = None

    def fit(self, user_item_matrix):
        self.matrix = user_item_matrix if self.based == 'user' else user_item_matrix.T
        
        if self.similarity_metric == 'cosine':
            norms = np.linalg.norm(self.matrix, axis=1)
            norms[norms == 0] = 1e-9
            normalized = self.matrix / norms[:, np.newaxis]
            self.similarity = np.dot(normalized, normalized.T)
        elif self.similarity_metric == 'pearson':
            means = np.mean(self.matrix, axis=1)
            centered = self.matrix - means[:, np.newaxis]
            norms = np.linalg.norm(centered, axis=1)
            norms[norms == 0] = 1e-9
            normalized = centered / norms[:, np.newaxis]
            self.similarity = np.dot(normalized, normalized.T)
        return self

    def predict(self, target_id, top_k=5, k_nn=10):
        # target_id is user_id si based=='user', sino item_id
        sim_scores = self.similarity[target_id]
        
        # k-NN ponderado
        nearest_neighbors = np.argsort(sim_scores)[::-1][1:k_nn+1] # exclude self
        
        if self.based == 'user':
            weighted_sum = np.zeros(self.matrix.shape[1])
            sum_sim = 0
            for nn in nearest_neighbors:
                if sim_scores[nn] > 0:
                    weighted_sum += sim_scores[nn] * self.matrix[nn]
                    sum_sim += sim_scores[nn]
                    
            predictions = weighted_sum / sum_sim if sum_sim > 0 else np.zeros_like(weighted_sum)
            # exclude seen
            predictions[self.matrix[target_id] > 0] = -np.inf
            
        else: # item-based
            pass # Simplification for this prototype
            predictions = np.random.rand(self.matrix.shape[0])

        top_items = np.argsort(predictions)[::-1][:top_k]
        return top_items

# ==========================================
# Content-Based
# ==========================================

class ContentBasedFiltering:
    def __init__(self):
        self.item_profiles = None
    
    def fit(self, item_features):
        self.item_profiles = item_features
        return self

    def predict(self, user_profile, top_k=5):
        norms_items = np.linalg.norm(self.item_profiles, axis=1)
        norms_items[norms_items == 0] = 1e-9
        
        norm_user = np.linalg.norm(user_profile)
        if norm_user == 0: norm_user = 1e-9
        
        sim = np.dot(self.item_profiles, user_profile) / (norms_items * norm_user)
        top_items = np.argsort(sim)[::-1][:top_k]
        return top_items

# ==========================================
# Hybrid Combination
# ==========================================

class HybridRecommender:
    def __init__(self, cf_weight=0.5, cb_weight=0.5):
        self.cf_weight = cf_weight
        self.cb_weight = cb_weight
        self.cf = CollaborativeFiltering(based='user', similarity_metric='cosine')
        self.cb = ContentBasedFiltering()
        self.user_item_matrix = None

    def fit(self, user_item_matrix, item_features):
        self.user_item_matrix = user_item_matrix
        self.cf.fit(user_item_matrix)
        self.cb.fit(item_features)
        
    def predict(self, user_id, user_profile, top_k=5, k_nn=10):
        # Get raw scores
        sim_scores_user = self.cf.similarity[user_id]
        nearest_users = np.argsort(sim_scores_user)[::-1][1:k_nn+1]
        
        cf_preds = np.zeros(self.user_item_matrix.shape[1])
        sum_sim = 0
        for nn in nearest_users:
            if sim_scores_user[nn] > 0:
                cf_preds += sim_scores_user[nn] * self.cf.matrix[nn]
                sum_sim += sim_scores_user[nn]
        if sum_sim > 0:
            cf_preds /= sum_sim
            
        norms_items = np.linalg.norm(self.cb.item_profiles, axis=1)
        norms_items[norms_items == 0] = 1e-9
        norm_user = np.linalg.norm(user_profile)
        if norm_user == 0: norm_user = 1e-9
        cb_preds = np.dot(self.cb.item_profiles, user_profile) / (norms_items * norm_user)
        
        # Combine
        hybrid_preds = (self.cf_weight * cf_preds) + (self.cb_weight * cb_preds)
        hybrid_preds[self.user_item_matrix[user_id] > 0] = -np.inf
        
        return np.argsort(hybrid_preds)[::-1][:top_k]

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
