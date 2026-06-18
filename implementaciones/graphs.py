import random
import time

class Graph:
    """
    Clase Grafo no dirigido implementada desde cero (sin dependencias de networkx).
    Representa el grafo usando listas/sets de adyacencia.
    """
    def __init__(self):
        self.adj = {}  # Diccionario de adyacencia: nodo -> set de vecinos
        self.num_edges_count = 0

    def add_node(self, node):
        if node not in self.adj:
            self.adj[node] = set()

    def add_edge(self, u, v):
        self.add_node(u)
        self.add_node(v)
        if v not in self.adj[u]:
            self.adj[u].add(v)
            self.adj[v].add(u)
            self.num_edges_count += 1

    def number_of_nodes(self):
        return len(self.adj)

    def number_of_edges(self):
        return self.num_edges_count

    def density(self):
        """
        Calcula la densidad de un grafo simple no dirigido:
        D = 2 * |E| / (|V| * (|V| - 1))
        """
        v = self.number_of_nodes()
        if v <= 1:
            return 0.0
        e = self.number_of_edges()
        return (2.0 * e) / (v * (v - 1))

    def connected_components(self):
        """
        Encuentra las componentes conexas del grafo utilizando BFS iterativo.
        Retorna una lista de conjuntos, donde cada conjunto contiene los nodos de una componente.
        """
        visited = set()
        components = []
        
        # Iterar sobre todos los nodos del grafo
        for node in self.adj:
            if node not in visited:
                component = set()
                queue = [node]
                visited.add(node)
                
                # BFS iterativo usando puntero de lectura para evitar pop(0) que es O(N)
                head = 0
                while head < len(queue):
                    curr = queue[head]
                    head += 1
                    component.add(curr)
                    
                    for neighbor in self.adj[curr]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                
                components.append(component)
        return components

    def _bfs_farthest_node(self, start_node, component_set):
        """
        Realiza un BFS desde start_node dentro del conjunto de la componente conexa
        para encontrar el nodo más lejano y la distancia máxima (excentricidad).
        """
        visited = {start_node: 0}
        queue = [start_node]
        head = 0
        max_dist = 0
        farthest_node = start_node
        
        while head < len(queue):
            curr = queue[head]
            dist = visited[curr]
            if dist > max_dist:
                max_dist = dist
                farthest_node = curr
            head += 1
            
            for neighbor in self.adj[curr]:
                if neighbor not in visited:
                    visited[neighbor] = dist + 1
                    queue.append(neighbor)
                    
        return farthest_node, max_dist

    def diameter(self, component_nodes=None, method='auto', max_sweeps=10):
        """
        Calcula el diámetro de una componente conexa del grafo.
        
        Parámetros:
        - component_nodes: Conjunto de nodos que conforman la componente. Si es None,
                           se toma la componente conexa más grande.
        - method: 'exact' (calcula excentricidad para cada nodo, O(|V|*(|V|+|E|))),
                  'approximate' (método double-sweep / multi-sweep, O(k*(|V|+|E|))),
                  'auto' (exacto si |V| <= 1000, aproximado si |V| > 1000).
        - max_sweeps: Número de barridos para el método aproximado (a mayor número, mayor precisión).
        """
        if component_nodes is None:
            components = self.connected_components()
            if not components:
                return 0
            # Tomar la componente conexa más grande (Gcc)
            component_nodes = max(components, key=len)
            
        n = len(component_nodes)
        if n <= 1:
            return 0
            
        if method == 'auto':
            method = 'exact' if n <= 1000 else 'approximate'
            
        if method == 'exact':
            # Diámetro exacto: BFS desde cada nodo
            max_diameter = 0
            component_nodes_list = list(component_nodes)
            for idx, node in enumerate(component_nodes_list):
                _, ecc = self._bfs_farthest_node(node, component_nodes)
                if ecc > max_diameter:
                    max_diameter = ecc
            return max_diameter
            
        elif method == 'approximate':
            # Diámetro aproximado usando multi-sweep BFS (cota inferior muy ajustada)
            max_diameter = 0
            nodes_list = list(component_nodes)
            
            for _ in range(max_sweeps):
                # Seleccionar un nodo inicial aleatorio
                start_node = random.choice(nodes_list)
                
                # Barrido 1: Encontrar el nodo más lejano 'u' desde start_node
                u, _ = self._bfs_farthest_node(start_node, component_nodes)
                
                # Barrido 2: Encontrar el nodo más lejano 'v' desde 'u'
                v, dist = self._bfs_farthest_node(u, component_nodes)
                
                if dist > max_diameter:
                    max_diameter = dist
                    
            return max_diameter
        else:
            raise ValueError("El método debe ser 'exact', 'approximate' o 'auto'")

    def pagerank(self, damping=0.85, max_iter=100, tol=1e-6):
        """
        Calcula el PageRank de cada nodo de forma iterativa.
        Retorna un diccionario: nodo -> puntuación de PageRank.
        """
        nodes = list(self.adj.keys())
        N = len(nodes)
        if N == 0:
            return {}
        
        # Inicializar PageRank de manera uniforme
        pr = {node: 1.0 / N for node in nodes}
        
        # Precalcular grados de los nodos
        degrees = {node: len(self.adj[node]) for node in nodes}
        
        # Identificar sumideros (nodos de grado 0)
        dangling_nodes = [node for node in nodes if degrees[node] == 0]
        
        for iteration in range(max_iter):
            next_pr = {node: 0.0 for node in nodes}
            
            # Calcular la suma de PageRank de los nodos colgantes (sumideros)
            dangling_sum = sum(pr[node] for node in dangling_nodes)
            
            # Para cada nodo, distribuir su PageRank a sus vecinos
            for node in nodes:
                deg = degrees[node]
                if deg > 0:
                    share = damping * pr[node] / deg
                    for neighbor in self.adj[node]:
                        next_pr[neighbor] += share
            
            # Término constante para todos los nodos (teleportación + dangling)
            const_contribution = (1.0 - damping) / N + (damping * dangling_sum / N)
            for node in nodes:
                next_pr[node] += const_contribution
                
            # Verificar convergencia (norma L1)
            err = sum(abs(next_pr[node] - pr[node]) for node in nodes)
            pr = next_pr
            if err < tol:
                break
                
        return pr

    def hits(self, max_iter=100, tol=1e-6):
        """
        Calcula las puntuaciones de Hubs y Authorities (HITS) de cada nodo.
        Retorna dos diccionarios: (hubs, authorities).
        """
        nodes = list(self.adj.keys())
        N = len(nodes)
        if N == 0:
            return {}, {}
        
        # Inicializar hubs y authorities con 1.0
        hubs = {node: 1.0 for node in nodes}
        authorities = {node: 1.0 for node in nodes}
        
        for iteration in range(max_iter):
            next_authorities = {node: 0.0 for node in nodes}
            next_hubs = {node: 0.0 for node in nodes}
            
            # 1. Actualizar autoridades: a(p) = suma de h(q) de los vecinos q
            for node in nodes:
                next_authorities[node] = sum(hubs[neighbor] for neighbor in self.adj[node])
                
            # 2. Actualizar hubs: h(p) = suma de a(q) de los vecinos q
            for node in nodes:
                next_hubs[node] = sum(authorities[neighbor] for neighbor in self.adj[node])
                
            # Normalizar autoridades (norma L2)
            norm_auth = sum(val**2 for val in next_authorities.values())**0.5
            if norm_auth > 0:
                for node in nodes:
                    next_authorities[node] /= norm_auth
                    
            # Normalizar hubs (norma L2)
            norm_hub = sum(val**2 for val in next_hubs.values())**0.5
            if norm_hub > 0:
                for node in nodes:
                    next_hubs[node] /= norm_hub
                    
            # Verificar convergencia
            err_auth = sum(abs(next_authorities[node] - authorities[node]) for node in nodes)
            err_hub = sum(abs(next_hubs[node] - hubs[node]) for node in nodes)
            
            authorities = next_authorities
            hubs = next_hubs
            
            if err_auth < tol and err_hub < tol:
                break
                
        return hubs, authorities

    def louvain_communities(self, max_levels=10, tol=1e-6):
        """
        Algoritmo de Louvain completo para la detección de comunidades por maximización de modularidad.
        Retorna:
        - partition: diccionario nodo -> community_id
        - modularity: valor de la modularidad obtenida
        """
        # Construir el grafo ponderado inicial
        adj_weighted = {}
        for u in self.adj:
            adj_weighted[u] = {}
            for v in self.adj[u]:
                adj_weighted[u][v] = 1.0
                
        # Mapeo de nodos originales a sus comunidades
        partition = {u: u for u in self.adj}
        
        current_adj = adj_weighted
        level = 0
        modularity = -1.0
        
        while level < max_levels:
            # 1. Ejecutar Fase 1
            temp_partition, q = _louvain_phase1(current_adj, tol=tol)
            
            # Si la ganancia en modularidad es muy pequeña o nula, salir
            if q - modularity <= tol:
                break
                
            modularity = q
            
            # 2. Actualizar la partición de los nodos originales
            for u in partition:
                partition[u] = temp_partition[partition[u]]
                
            # 3. Condensar el grafo
            current_adj = _louvain_phase2(current_adj, temp_partition)
            level += 1
            
        # Renombrar las comunidades para tener IDs correlativos de 0 a C-1
        unique_comms = list(set(partition.values()))
        comm_map = {comm: idx for idx, comm in enumerate(unique_comms)}
        for u in partition:
            partition[u] = comm_map[partition[u]]
            
        return partition, modularity


class BipartiteGraph(Graph):
    """
    Subclase especializada para grafos bipartitos Usuario-Producto.
    Mantiene conjuntos separados para ambos tipos de nodos.
    """
    def __init__(self):
        super().__init__()
        self.user_nodes = set()
        self.product_nodes = set()

    def add_bipartite_edge(self, user, product):
        # Asegurar prefijos para evitar colisiones si los IDs se solapan
        u_node = f"U_{user}" if not str(user).startswith("U_") else user
        p_node = f"P_{product}" if not str(product).startswith("P_") else product
        
        self.user_nodes.add(u_node)
        self.product_nodes.add(p_node)
        self.add_edge(u_node, p_node)

    def bipartite_density(self):
        """
        Calcula la densidad específica para un grafo bipartito:
        D_bip = |E| / (|V_user| * |V_product|)
        """
        u = len(self.user_nodes)
        p = len(self.product_nodes)
        if u == 0 or p == 0:
            return 0.0
        return self.number_of_edges() / (u * p)

import random
import time

class DiGraph:
    """
    Clase base de Grafo Dirigido optimizada.
    """
    def __init__(self):
        self.adj = {}  # Enlaces salientes: nodo -> set de nodos a los que apunta
        self.in_adj = {} # Enlaces entrantes (útil para HITS/PageRank): nodo -> set de nodos que lo apuntan
        self.num_edges_count = 0

    def add_node(self, node):
        if node not in self.adj:
            self.adj[node] = set()
            self.in_adj[node] = set()

    def add_edge(self, u, v):
        """Añade una arista dirigida u -> v"""
        self.add_node(u)
        self.add_node(v)
        if v not in self.adj[u]:
            self.adj[u].add(v)
            self.in_adj[v].add(u)
            self.num_edges_count += 1

    def number_of_nodes(self):
        return len(self.adj)

    def number_of_edges(self):
        return self.num_edges_count

    def density(self):
        v = self.number_of_nodes()
        if v <= 1:
            return 0.0
        return float(self.num_edges_count) / (v * (v - 1))

    # --- ALGORITMOS AJUSTADOS A DIRIGIDO ---

    def pagerank(self, damping=0.85, max_iter=100, tol=1e-6):
        nodes = list(self.adj.keys())
        N = len(nodes)
        if N == 0:
            return {}
        
        pr = {node: 1.0 / N for node in nodes}
        out_degree = {node: len(self.adj[node]) for node in nodes}
        dangling_nodes = [node for node in nodes if out_degree[node] == 0]
        
        for iteration in range(max_iter):
            next_pr = {node: 0.0 for node in nodes}
            dangling_sum = sum(pr[node] for node in dangling_nodes)
            
            # Distribuir a través de los enlaces salientes
            for node in nodes:
                deg = out_degree[node]
                if deg > 0:
                    share = pr[node] / deg
                    for neighbor in self.adj[node]:
                        next_pr[neighbor] += share
            
            # Teleportación + Distribución de sumideros (Fórmula de tu diapositiva)
            teleport = (1.0 - damping) / N
            dangling_allocation = (damping * dangling_sum) / N
            
            for node in nodes:
                next_pr[node] = (damping * next_pr[node]) + teleport + dangling_allocation
            
            err = sum(abs(next_pr[node] - pr[node]) for node in nodes)
            pr = next_pr
            if err < tol:
                break
                
        return pr

    def hits(self, max_iter=100, tol=1e-6):
        nodes = list(self.adj.keys())
        if len(nodes) == 0:
            return {}, {}
        
        hubs = {node: 1.0 for node in nodes}
        authorities = {node: 1.0 for node in nodes}
        
        for iteration in range(max_iter):
            next_authorities = {node: 0.0 for node in nodes}
            next_hubs = {node: 0.0 for node in nodes}
            
            # 1. Autoridad: se recibe de los hubs que te apuntan (in_adj)
            for node in nodes:
                next_authorities[node] = sum(hubs[parent] for parent in self.in_adj[node])
                
            # 2. Hubs: apuntan a nodos con alta autoridad (adj)
            for node in nodes:
                next_hubs[node] = sum(authorities[child] for child in self.adj[node])
                
            # Normalización L2
            norm_auth = sum(val**2 for val in next_authorities.values())**0.5
            if norm_auth > 0:
                for node in nodes: next_authorities[node] /= norm_auth
                    
            norm_hub = sum(val**2 for val in next_hubs.values())**0.5
            if norm_hub > 0:
                for node in nodes: next_hubs[node] /= norm_hub
            
            err_auth = sum(abs(next_authorities[node] - authorities[node]) for node in nodes)
            err_hub = sum(abs(next_hubs[node] - hubs[node]) for node in nodes)
            
            authorities = next_authorities
            hubs = next_hubs
            
            if err_auth < tol and err_hub < tol:
                break
                
        return hubs, authorities

class BipartiteDiGraph(DiGraph):
    """
    Subclase especializada para grafos bipartitos DIRIGIDOS (Usuario -> Producto/Negocio).
    """
    def __init__(self):
        super().__init__()
        self.user_nodes = set()
        self.product_nodes = set()

    def add_bipartite_edge(self, user, product):
        u_node = f"U_{user}" if not str(user).startswith("U_") else user
        p_node = f"P_{product}" if not str(product).startswith("P_") else product
        
        self.user_nodes.add(u_node)
        self.product_nodes.add(p_node)
        
        # Enlace estrictamente dirigido: Usuario apunta a Negocio
        self.add_edge(u_node, p_node)

    def bipartite_density(self):
        u = len(self.user_nodes)
        p = len(self.product_nodes)
        if u == 0 or p == 0:
            return 0.0
        return self.number_of_edges() / (u * p)
    
def _louvain_phase1(adj_weighted, tol=1e-6):
    # Calcular grados ponderados k
    k = {u: sum(neighbors.values()) for u, neighbors in adj_weighted.items()}
    m2 = sum(k.values())
    if m2 == 0:
        return {u: u for u in adj_weighted}, 0.0
        
    # Inicializar partición: cada nodo en su propia comunidad
    partition = {u: u for u in adj_weighted}
    community_tot = {u: k[u] for u in adj_weighted}
    
    # community_in mantiene la suma de los pesos de las aristas internas de cada comunidad.
    # Inicialmente, para cada nodo u, community_in[u] es el peso de su self-loop (si tiene).
    community_in = {u: adj_weighted[u].get(u, 0.0) for u in adj_weighted}
    
    nodes = list(adj_weighted.keys())
    improved = True
    
    # Calcular modularidad inicial
    q_init = 0.0
    for c in community_tot:
        q_init += (community_in.get(c, 0.0) / m2) - (community_tot[c] / m2) ** 2
        
    q_current = q_init
    
    while improved:
        improved = False
        # Mezclar nodos para evitar sesgos de ordenamiento
        random.shuffle(nodes)
        
        for u in nodes:
            c_old = partition[u]
            k_u = k[u]
            
            # Si el nodo no tiene vecinos, no hay nada que hacer
            if k_u == 0:
                continue
                
            # Encontrar comunidades vecinas de u
            neighbor_communities = {}
            for v, weight in adj_weighted[u].items():
                if v == u:
                    continue
                c_v = partition[v]
                neighbor_communities[c_v] = neighbor_communities.get(c_v, 0.0) + weight
            
            # Calcular k_u,in en su comunidad actual
            k_u_in_old = neighbor_communities.get(c_old, 0.0)
            
            best_community = c_old
            best_gain = 0.0
            
            for c_new, k_u_in_new in neighbor_communities.items():
                if c_new == c_old:
                    continue
                
                # delta_q es la diferencia en modularidad al mover u de c_old a c_new
                delta_q = (k_u_in_new - k_u_in_old) * 2.0 / m2 - k_u * (community_tot[c_new] - community_tot[c_old] + k_u) * 2.0 / (m2 ** 2)
                
                if delta_q > best_gain:
                    best_gain = delta_q
                    best_community = c_new
            
            if best_community != c_old and best_gain > tol:
                # Mover u a best_community
                partition[u] = best_community
                
                # Actualizar community_tot y community_in
                k_u_in_new = neighbor_communities[best_community]
                
                community_tot[c_old] -= k_u
                community_tot[best_community] += k_u
                
                # El peso interno de c_old disminuye en 2 * k_u_in_old + self_loop(u)
                self_loop = adj_weighted[u].get(u, 0.0)
                community_in[c_old] -= (2 * k_u_in_old + self_loop)
                # El peso interno de best_community aumenta en 2 * k_u_in_new + self_loop(u)
                community_in[best_community] += (2 * k_u_in_new + self_loop)
                
                improved = True
                
        # Calcular nueva modularidad
        q_new = 0.0
        for c in community_tot:
            q_new += (community_in.get(c, 0.0) / m2) - (community_tot[c] / m2) ** 2
            
        if q_new - q_current < tol:
            break
        q_current = q_new
        
    return partition, q_current

def _louvain_phase2(adj_weighted, partition):
    new_adj = {}
    for u, neighbors in adj_weighted.items():
        c_u = partition[u]
        if c_u not in new_adj:
            new_adj[c_u] = {}
        for v, weight in neighbors.items():
            c_v = partition[v]
            new_adj[c_u][c_v] = new_adj[c_u].get(c_v, 0.0) + weight
    return new_adj

