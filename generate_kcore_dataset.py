import json
from collections import defaultdict
import time
import random
from tqdm import tqdm

REVIEW_PATH = 'Yelp-JSON/Yelp JSON/yelp_academic_dataset_review.json'
OUTPUT_PATH = 'Yelp-JSON/Yelp JSON/yelp_academic_dataset_review_10core.json'

print("Fase 1: Cargando adyacencias de la red completa...")
start_time = time.time()
user_adj = defaultdict(set)
item_adj = defaultdict(set)

with open(REVIEW_PATH, 'r', encoding='utf-8') as f:
    for line in tqdm(f, desc="Leyendo adyacencias"):
        obj = json.loads(line)
        u = obj['user_id']
        i = obj['business_id']
        user_adj[u].add(i)
        item_adj[i].add(u)

print("Fase 2: Ejecutando k-core (k=10) iterativo sobre los conjuntos...")
k = 10
while True:
    users_to_remove = [u for u, items in user_adj.items() if len(items) < k]
    items_to_remove = [i for i, users in item_adj.items() if len(users) < k]
    
    if not users_to_remove and not items_to_remove:
        break
        
    for u in users_to_remove:
        for item in user_adj[u]:
            item_adj[item].discard(u)
        del user_adj[u]
        
    for i in items_to_remove:
        for user in item_adj[i]:
            user_adj[user].discard(i)
        del item_adj[i]

print(f"k-core finalizado. Usuarios en core: {len(user_adj)} | Negocios en core: {len(item_adj)}")

print("\nFase 3: Filtrando y recolectando reseñas que pertenecen al 10-core...")
valid_users = set(user_adj.keys())
valid_items = set(item_adj.keys())

# Recolectar todas las reseñas que pertenezcan a la red densa, agrupándolas por usuario
reviews_by_user = defaultdict(list)
with open(REVIEW_PATH, 'r', encoding='utf-8') as f:
    for line in tqdm(f, desc="Recolectando reseñas del core"):
        obj = json.loads(line)
        u = obj['user_id']
        i = obj['business_id']
        if u in valid_users and i in valid_items:
            reviews_by_user[u].append({
                'user_id': u,
                'business_id': i,
                'stars': obj['stars']
            })

# Contar cuántas reseñas totales hay en el 10-core
total_core_reviews = sum(len(revs) for revs in reviews_by_user.values())
print(f"Total reseñas en el 10-core completo: {total_core_reviews}")

print("\nFase 4: Realizando Muestreo Centrado en Usuarios (User-Centric Sampling) para mantener la densidad...")
random.seed(42)
selected_users = list(reviews_by_user.keys())
random.shuffle(selected_users)

sample_reviews = []
num_users_selected = 0
for u in selected_users:
    sample_reviews.extend(reviews_by_user[u])
    num_users_selected += 1
    if len(sample_reviews) >= 800000:
        break

print(f"Muestreo finalizado: Seleccionamos {num_users_selected} usuarios y recolectamos {len(sample_reviews)} reseñas.")

print(f"Escribiendo {len(sample_reviews)} reseñas en {OUTPUT_PATH}...")
with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    for rev in sample_reviews:
        f.write(json.dumps(rev) + '\n')

print("¡Dataset densificado 10-core generado con éxito con muestreo denso!")
