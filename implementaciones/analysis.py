import pandas as pd

def compare_rankings(g, pr_scores, hubs, authorities):
    """
    Compara cuantitativamente los rankings de PageRank e HITS.
    Retorna la correlación de Spearman y los solapamientos del Top 50.
    """
    # 1. Extraer puntuaciones de negocios
    business_nodes = list(g.product_nodes)
    b_pr_ranks = {node: pr_scores[node] for node in business_nodes if node in pr_scores}
    b_auth_ranks = {node: authorities[node] for node in business_nodes if node in authorities}
    
    # Asignar rangos para Spearman
    def get_ranks(d):
        sorted_nodes = sorted(d.keys(), key=lambda x: d[x], reverse=True)
        return {node: rank for rank, node in enumerate(sorted_nodes, 1)}
    
    b_pr_rankings = get_ranks(b_pr_ranks)
    b_auth_rankings = get_ranks(b_auth_ranks)
    
    diff_sq = 0
    n_b = len(business_nodes)
    for node in business_nodes:
        r1 = b_pr_rankings.get(node, n_b / 2)
        r2 = b_auth_rankings.get(node, n_b / 2)
        diff_sq += (r1 - r2) ** 2
    
    spearman_business = 1.0 - (6.0 * diff_sq) / (n_b * (n_b**2 - 1)) if n_b > 1 else 0.0
    
    # 2. Extraer puntuaciones de usuarios
    user_nodes = list(g.user_nodes)
    u_pr_ranks = {node: pr_scores[node] for node in user_nodes if node in pr_scores}
    u_hub_ranks = {node: hubs[node] for node in user_nodes if node in hubs}
    
    u_pr_rankings = get_ranks(u_pr_ranks)
    u_hub_rankings = get_ranks(u_hub_ranks)
    
    diff_sq_u = 0
    n_u = len(user_nodes)
    for node in user_nodes:
        r1 = u_pr_rankings.get(node, n_u / 2)
        r2 = u_hub_rankings.get(node, n_u / 2)
        diff_sq_u += (r1 - r2) ** 2
        
    spearman_users = 1.0 - (6.0 * diff_sq_u) / (n_u * (n_u**2 - 1)) if n_u > 1 else 0.0
    
    # 3. Solapamientos Top 50
    def top_k_overlap(d1, d2, k=50):
        top1 = set(sorted(d1.keys(), key=lambda x: d1[x], reverse=True)[:k])
        top2 = set(sorted(d2.keys(), key=lambda x: d2[x], reverse=True)[:k])
        return len(top1.intersection(top2)) / k
        
    overlap_business = top_k_overlap(b_pr_ranks, b_auth_ranks, k=50)
    overlap_users = top_k_overlap(u_pr_ranks, u_hub_ranks, k=50)
    
    return {
        'spearman_business': spearman_business,
        'spearman_users': spearman_users,
        'overlap_business_top50': overlap_business,
        'overlap_users_top50': overlap_users
    }

def get_top_k_table(scores_dict, g, business_map=None, k=10):
    """
    Formatea las puntuaciones de un ranking en un DataFrame de pandas para visualización.
    """
    top_items = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)[:k]
    data = []
    for node, score in top_items:
        node_type = 'Usuario' if node.startswith('U_') else 'Negocio'
        name = ''
        city = ''
        if node_type == 'Negocio' and business_map:
            b_id = node[2:]
            name = business_map.get(b_id, {}).get('name', 'Desconocido')
            city = business_map.get(b_id, {}).get('city', 'Desconocido')
        data.append({
            'node_id': node,
            'type': node_type,
            'score': score,
            'name': name,
            'city': city
        })
    return pd.DataFrame(data)

def characterize_communities(g, partition, business_map, top_k=5):
    """
    Caracteriza las top_k comunidades más grandes del grafo g.
    """
    community_nodes = {}
    for node, comm in partition.items():
        if comm not in community_nodes:
            community_nodes[comm] = []
        community_nodes[comm].append(node)
        
    sorted_comms = sorted(community_nodes.items(), key=lambda x: len(x[1]), reverse=True)
    comms_summary = []
    
    for comm_id, nodes in sorted_comms[:top_k]:
        users = [n for n in nodes if n.startswith('U_')]
        products = [n for n in nodes if n.startswith('P_')]
        
        # Conexiones internas y externas
        internal_edges = 0
        inter_edges = 0
        for u in nodes:
            for neighbor in g.adj[u]:
                if partition[neighbor] == comm_id:
                    internal_edges += 1
                else:
                    inter_edges += 1
        internal_edges //= 2
        
        num_users = len(users)
        num_products = len(products)
        
        # Densidad bipartita interna
        density = 0.0
        if num_users > 0 and num_products > 0:
            density = internal_edges / (num_users * num_products)
            
        # Grados internos
        node_int_degrees = {}
        for u in nodes:
            node_int_degrees[u] = sum(1 for neighbor in g.adj[u] if partition[neighbor] == comm_id)
            
        top_u_nodes = sorted(users, key=lambda x: node_int_degrees.get(x, 0), reverse=True)[:3]
        top_p_nodes = sorted(products, key=lambda x: node_int_degrees.get(x, 0), reverse=True)[:3]
        
        top_p_mapped = []
        for p in top_p_nodes:
            b_id = p[2:]
            b_name = business_map.get(b_id, {}).get('name', 'Desconocido')
            b_city = business_map.get(b_id, {}).get('city', 'Desconocido')
            top_p_mapped.append({
                'node_id': p,
                'name': b_name,
                'city': b_city,
                'internal_degree': node_int_degrees[p],
                'global_degree': len(g.adj[p])
            })
            
        top_u_mapped = []
        for u in top_u_nodes:
            top_u_mapped.append({
                'node_id': u,
                'internal_degree': node_int_degrees[u],
                'global_degree': len(g.adj[u])
            })
            
        # Ciudades y categorías
        cities = {}
        categories = {}
        for p in products:
            b_id = p[2:]
            b_info = business_map.get(b_id, {})
            city = b_info.get('city', 'Desconocido')
            cities[city] = cities.get(city, 0) + 1
            
            cats = b_info.get('categories', '')
            if cats:
                for cat in cats.split(','):
                    cat = cat.strip()
                    categories[cat] = categories.get(cat, 0) + 1
                    
        top_cities = sorted(cities.items(), key=lambda x: x[1], reverse=True)[:3]
        top_cats = sorted(categories.items(), key=lambda x: x[1], reverse=True)[:5]
        
        comms_summary.append({
            'comm_id': int(comm_id),
            'total_size': len(nodes),
            'num_users': num_users,
            'num_products': num_products,
            'internal_edges': internal_edges,
            'inter_edges': inter_edges,
            'density': density,
            'top_users': top_u_mapped,
            'top_products': top_p_mapped,
            'top_cities': top_cities,
            'top_categories': top_cats
        })
        
    return comms_summary
