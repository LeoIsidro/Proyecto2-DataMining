import json
import os
import numpy as np
import pandas as pd
from collections import Counter

def export_data_for_web(review_df, business_df, output_path='p2/presentation_data.js'):
    """
    Procesa y exporta las métricas calculadas del EDA en Python a un archivo
    JavaScript ligero, optimizado para renderizado dinámico en el navegador.
    """
    print(f"Iniciando exportación de datos para la presentación web...")
    
    # 1. Distribución de estrellas (Reviews vs Negocios)
    review_stars = review_df['stars'].value_counts().sort_index()
    business_stars = business_df['stars'].value_counts().sort_index()
    
    # 2. Distribución de número de reseñas (Log-scale Histogram precalculado)
    review_counts = business_df['review_count'].values
    log_review_counts = np.log10(review_counts)
    counts, bin_edges = np.histogram(log_review_counts, bins=50)
    # Convertir bin_edges de log10 de vuelta a valores reales para el tooltip en JS
    bin_centers = 10 ** ((bin_edges[:-1] + bin_edges[1:]) / 2)
    review_count_hist = {
        'x': bin_centers.tolist(),
        'y': counts.tolist(),
        'edges': (10 ** bin_edges).tolist()
    }
    
    # 3. Top 15 categorías de negocio
    categories_series = business_df['categories'].str.split(',').explode().str.strip()
    top_cats = categories_series.value_counts().head(15)
    top_categories = {
        'names': top_cats.index.tolist(),
        'counts': top_cats.values.tolist()
    }
    
    # 4. Relación Stars vs Review Count (Submuestreo de 3000 puntos para fluidez visual)
    scatter_sample = business_df[['stars', 'review_count']].sample(
        n=min(3000, len(business_df)), 
        random_state=42
    )
    scatter_data = {
        'stars': scatter_sample['stars'].tolist(),
        'review_count': scatter_sample['review_count'].tolist()
    }
    
    # 5. Análisis Geográfico (Top 5 Estados)
    top_states = business_df['state'].value_counts().head(5).index.tolist()
    state_counts = business_df['state'].value_counts().head(5)
    
    # Datos para Boxplot (submuestreo de 1000 puntos por estado para limitar peso del archivo)
    boxplot_data = {}
    for state in top_states:
        state_stars = business_df[business_df['state'] == state]['stars']
        state_stars_sample = state_stars.sample(
            n=min(1000, len(state_stars)),
            random_state=42
        ).tolist()
        boxplot_data[state] = state_stars_sample
        
    state_stats = {
        'states': top_states,
        'counts': state_counts.values.tolist(),
        'boxplot': boxplot_data
    }
    
    # Agrupar todo el payload
    payload = {
        'reviewStars': {
            'x': review_stars.index.tolist(),
            'y': review_stars.values.tolist()
        },
        'businessStars': {
            'x': business_stars.index.tolist(),
            'y': business_stars.values.tolist()
        },
        'reviewCountHist': review_count_hist,
        'topCategories': top_categories,
        'scatterStarsReviews': scatter_data,
        'stateStats': state_stats
    }
    
    # Crear directorio si no existe
    dir_name = os.path.dirname(output_path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)
        
    # Escribir como variable JavaScript global
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("// Datos de exploración de Yelp generados automáticamente por Python\n")
        f.write("const yelpEdaData = ")
        json.dump(payload, f, indent=2)
        f.write(";\n")
        
    print(f"¡Exportación exitosa! Datos guardados en: {output_path}")


def export_graph_for_web(g, partition, business_df=None, output_path='p2/graph_data.js'):
    """
    Exporta el subgrafo de las top comunidades de Louvain a un archivo JSON/JS
    optimizado para visualización con D3.js.
    """
    print("Iniciando exportación del grafo para D3.js...")
    
    # 1. Identificar las top 5 comunidades más grandes
    comm_counts = Counter(partition.values())
    top_comms = [c for c, _ in comm_counts.most_common(5)]
    
    # Mapa de nombres de negocios si está disponible
    biz_names = {}
    if business_df is not None:
        for _, row in business_df.iterrows():
            bid = row['business_id']
            name = row['name']
            biz_names[bid] = name
            biz_names[f"P_{bid}"] = name

    # 2. Seleccionar los top 50 nodos de cada una de las top 5 comunidades por grado
    nodes_to_plot = set()
    for c in top_comms:
        comm_nodes = [n for n, comm_id in partition.items() if comm_id == c]
        # Ordenar por grado
        comm_nodes.sort(key=lambda x: len(g.adj[x]), reverse=True)
        nodes_to_plot.update(comm_nodes[:50])
        
    # Calcular PageRank y HITS para dar tamaño y significancia a los nodos en D3
    print("Calculando PageRank para los tamaños de nodos...")
    try:
        pr_scores = g.pagerank(max_iter=30, tol=1e-4)
    except Exception as e:
        print(f"No se pudo calcular PageRank ({e}). Usando grados como tamaño.")
        pr_scores = {n: len(g.adj[n]) for n in g.adj}
        
    try:
        _, auths = g.hits(max_iter=30, tol=1e-4)
    except Exception as e:
        auths = {n: len(g.adj[n]) for n in g.adj}

    # 3. Construir lista de nodos
    nodes_list = []
    node_id_map = {}
    for idx, node in enumerate(nodes_to_plot):
        node_id_map[node] = idx
        is_user = node.startswith("U_")
        
        # Nombre para mostrar
        raw_id = node[2:]
        if is_user:
            label = f"Usuario {raw_id[:6]}"
        else:
            label = biz_names.get(node, biz_names.get(raw_id, f"Negocio {raw_id[:6]}"))
            
        nodes_list.append({
            "id": node,
            "label": label,
            "type": "user" if is_user else "business",
            "community": partition[node],
            "degree": len(g.adj[node]),
            "pagerank": pr_scores.get(node, 0.0),
            "authority": auths.get(node, 0.0) if not is_user else 0.0
        })
        
    # 4. Construir lista de enlaces
    links_list = []
    for u in nodes_to_plot:
        for v in g.adj[u]:
            if v in nodes_to_plot:
                # Evitar duplicados (arista no dirigida)
                if node_id_map[u] < node_id_map[v]:
                    links_list.append({
                        "source": u,
                        "target": v
                    })
                    
    payload = {
        "top_communities": top_comms,
        "nodes": nodes_list,
        "links": links_list
    }
    
    # Guardar como JS global
    dir_name = os.path.dirname(output_path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)
        
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("// Subgrafo de comunidades de Louvain exportado para D3.js\n")
        f.write("const yelpGraphData = ")
        json.dump(payload, f, indent=2)
        f.write(";\n")
        
    print(f"¡Grafo exportado con éxito a {output_path}!")


def export_elbow_data(k_values, users_pct, reviews_pct, output_path='p2/elbow_data.js'):
    """
    Exporta las curvas de retención de usuarios y reseñas (método del codo k-core)
    a un archivo JS para renderizado interactivo en la presentación.
    """
    print("Iniciando exportación de datos del codo k-core...")
    
    # Calcular el codo matemáticamente de nuevo en el script para verificar
    x_arr = np.array(k_values)
    y_arr = np.array(reviews_pct)
    p1 = np.array([x_arr[0], y_arr[0]])
    p2 = np.array([x_arr[-1], y_arr[-1]])
    line_vec = p2 - p1
    line_vec_norm = line_vec / np.linalg.norm(line_vec)
    distances = []
    for i in range(len(x_arr)):
        p = np.array([x_arr[i], y_arr[i]])
        start_to_p = p - p1
        proj = np.dot(start_to_p, line_vec_norm) * line_vec_norm
        perp_vec = start_to_p - proj
        distances.append(np.linalg.norm(perp_vec))
    best_idx = np.argmax(distances)
    k_elbow = int(x_arr[best_idx])
    pct_elbow = float(y_arr[best_idx])
    
    payload = {
        "k_values": list(k_values),
        "users_pct": list(users_pct),
        "reviews_pct": list(reviews_pct),
        "elbow": {
            "k": k_elbow,
            "pct": pct_elbow,
            "idx": int(best_idx),
            "secant_start": [int(x_arr[0]), float(y_arr[0])],
            "secant_end": [int(x_arr[-1]), float(y_arr[-1])]
        }
    }
    
    dir_name = os.path.dirname(output_path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)
        
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("// Datos del análisis del codo K-Core exportados para la web\n")
        f.write("const yelpElbowData = ")
        json.dump(payload, f, indent=2)
        f.write(";\n")
        
    print(f"¡Datos del codo exportados con éxito a {output_path}!")

if __name__ == '__main__':
    print("Script de exportación listo para importar.")
