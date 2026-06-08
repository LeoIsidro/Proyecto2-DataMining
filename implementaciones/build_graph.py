import time
from preprocessing import load_reviews_efficiently
from implementaciones.graphs import BipartiteGraph

def main():
    print("=== Construcción y Análisis del Grafo Usuario-Producto ===")
    
    # Rutas a los datasets de Yelp en la carpeta local
    REVIEW_PATH = 'Yelp-JSON/Yelp JSON/yelp_academic_dataset_review.json'
    
    start_time = time.time()
    
    # Paso 1: Cargar de forma eficiente una muestra de 100,000 reseñas a partir de las primeras 300,001 filas
    # Esto replica el comportamiento del notebook original que tenía un 'review_sample.json' de 300k filas.
    print("\n1. Cargando y procesando el dataset...")
    # Cargamos las primeras 300,001 filas
    df_temp = load_reviews_efficiently(REVIEW_PATH, sample_size=300001, use_reservoir=False)
    print(f"Filas leídas del dataset original: {len(df_temp)}")
    
    # Tomamos la muestra aleatoria de 100,000 filas (con random_state=42 para reproducibilidad)
    print("Seleccionando muestra aleatoria de 100,000 reseñas...")
    sample_df = df_temp.sample(min(100000, len(df_temp)), random_state=42)
    print(f"Tamaño de la muestra final: {len(sample_df)}")
    
    # Paso 2: Construir el grafo bipartito
    print("\n2. Construyendo el grafo bipartito Usuario-Producto...")
    g = BipartiteGraph()
    for _, row in sample_df.iterrows():
        # Añadimos la arista entre el usuario y el producto/negocio
        g.add_bipartite_edge(row['user_id'], row['business_id'])
        
    print(f"Grafo construido en {time.time() - start_time:.2f} segundos.")
    
    # Paso 3: Calcular métricas básicas
    print("\n3. Calculando métricas básicas del grafo...")
    nodes = g.number_of_nodes()
    edges = g.number_of_edges()
    users = len(g.user_nodes)
    products = len(g.product_nodes)
    density_std = g.density()
    density_bip = g.bipartite_density()
    
    print(f" - Total de Nodos (|V|): {nodes} (Usuarios: {users}, Productos/Negocios: {products})")
    print(f" - Total de Aristas (|E|): {edges}")
    print(f" - Densidad Estándar (General): {density_std:.8f}")
    print(f" - Densidad Bipartita: {density_bip:.8f}")
    
    # Paso 4: Componentes conexas
    print("\n4. Calculando componentes conexas...")
    cc_start = time.time()
    components = g.connected_components()
    largest_cc = max(components, key=len)
    print(f" - Total de componentes conexas: {len(components)}")
    print(f" - Tamaño de la componente más grande (Gcc): {len(largest_cc)} nodos ({(len(largest_cc)/nodes)*100:.2f}% del grafo)")
    print(f"Componentes calculadas en {time.time() - cc_start:.4f} segundos.")
    
    # Paso 5: Cálculo de diámetro (inicial / Gcc)
    print("\n5. Calculando el diámetro de la componente más grande (Gcc)...")
    print("Nota: El cálculo exacto en un componente de este tamaño puede tomar horas en Python.")
    print("Utilizaremos nuestro algoritmo optimizado de barridos múltiples (Multi-sweep BFS).")
    
    dia_start = time.time()
    # Ejecutamos con 10 barridos múltiples para garantizar una aproximación del diámetro extremadamente precisa (usualmente exacta)
    approx_dia = g.diameter(largest_cc, method='approximate', max_sweeps=10)
    dia_end = time.time()
    
    print(f" - Diámetro aproximado (Multi-sweep BFS): {approx_dia}")
    print(f"Cálculo del diámetro completado en {dia_end - dia_start:.4f} segundos (¡en lugar de horas!).")
    
    print(f"\nTiempo total de ejecución: {time.time() - start_time:.2f} segundos.")

if __name__ == '__main__':
    main()
