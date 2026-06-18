import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from implementaciones.graphs import BipartiteGraph

from tqdm import tqdm
import random

# Alias para mantener compatibilidad
ujson = json

def load_json_lines(filepath, max_rows=None):
    """
    Carga líneas de un archivo JSONL (JSON Lines) de manera convencional.
    Permite limitar el número de filas cargadas con max_rows.
    """
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, desc=f"Cargando {filepath}")):
            data.append(ujson.loads(line))
            if max_rows and i + 1 >= max_rows:
                break
    return pd.DataFrame(data)

def load_reviews_efficiently(filepath, sample_size=100000, use_reservoir=True, random_state=42):
    """
    Carga de forma eficiente los datos de reseñas de Yelp.
    Para evitar problemas de memoria con el archivo de 5.34 GB, solo extrae
    los campos necesarios ('user_id', 'business_id', 'stars') y aplica
    muestreo (ya sea cargando las primeras N líneas o mediante Reservoir Sampling).
    """
    random.seed(random_state)
    data = []
    
    if use_reservoir and sample_size is not None:
        # Reservoir Sampling para obtener una muestra aleatoria y no sesgada en una sola pasada de memoria.
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(tqdm(f, desc="Reservoir Sampling de reseñas")):
                if len(data) < sample_size:
                    obj = ujson.loads(line)
                    data.append({
                        'user_id': obj['user_id'],
                        'business_id': obj['business_id'],
                        'stars': obj['stars']
                    })
                else:
                    j = random.randint(0, i)
                    if j < sample_size:
                        obj = ujson.loads(line)
                        data[j] = {
                            'user_id': obj['user_id'],
                            'business_id': obj['business_id'],
                            'stars': obj['stars']
                        }
    else:
        # Carga secuencial normal con límite de filas
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(tqdm(f, desc="Carga secuencial de reseñas")):
                if sample_size and i >= sample_size:
                    break
                obj = ujson.loads(line)
                data.append({
                    'user_id': obj['user_id'],
                    'business_id': obj['business_id'],
                    'stars': obj['stars']
                })
                
    df = pd.DataFrame(data)
    # Limpieza básica
    df.drop_duplicates(inplace=True)
    df.dropna(subset=['user_id', 'business_id', 'stars'], inplace=True)
    return df

def load_business_efficiently(filepath):
    """
    Carga de forma eficiente el archivo de negocios de Yelp, seleccionando
    únicamente las columnas de interés y eliminando duplicados.
    """
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Cargando negocios"):
            obj = ujson.loads(line)
            data.append({
                'business_id': obj['business_id'],
                'name': obj['name'],
                'city': obj['city'],
                'state': obj['state'],
                'stars': obj['stars'],
                'review_count': obj['review_count'],
                'categories': obj.get('categories', '')
            })
            
    df = pd.DataFrame(data)
    df.drop_duplicates(inplace=True)
    return df


# FUNCIONES DE LIMPIEZA, EDA Y ANÁLISIS



def perform_exhaustive_cleaning(review_df, business_df):
    """
    Realiza una limpieza exhaustiva del dataset de Yelp.
    Detecta y trata valores faltantes, duplicados, inconsistencias y outliers.
    Retorna los DataFrames limpios y un diccionario con el reporte del proceso.
    """
    report = {}
    
    # 1. VALORES FALTANTES
    # Reviews
    rev_initial_shape = review_df.shape[0]
    rev_nulls = review_df[['user_id', 'business_id', 'stars']].isnull().sum()
    report['reviews_missing_values'] = rev_nulls.to_dict()
    # Dropping critical NaNs in reviews
    review_df = review_df.dropna(subset=['user_id', 'business_id', 'stars'])
    report['reviews_dropped_nan'] = rev_initial_shape - review_df.shape[0]
    
    # Businesses
    bus_initial_shape = business_df.shape[0]
    bus_nulls = business_df[['business_id', 'name', 'stars', 'review_count']].isnull().sum()
    report['business_missing_values'] = bus_nulls.to_dict()
    # Dropping critical NaNs in business
    business_df = business_df.dropna(subset=['business_id', 'name', 'stars'])
    # Handle missing categories by imputing 'Uncategorized' instead of dropping
    bus_null_categories = business_df['categories'].isnull().sum()
    business_df['categories'] = business_df['categories'].fillna('Uncategorized')
    report['business_dropped_nan'] = bus_initial_shape - business_df.shape[0]
    report['business_imputed_categories'] = bus_null_categories

    # 2. DUPLICADOS
    # Reviews (mismo usuario califica mismo negocio)
    rev_before_dup = review_df.shape[0]
    review_df = review_df.drop_duplicates(subset=['user_id', 'business_id'])
    report['reviews_dropped_duplicates'] = rev_before_dup - review_df.shape[0]
    
    # Businesses
    bus_before_dup = business_df.shape[0]
    business_df = business_df.drop_duplicates(subset=['business_id'])
    report['business_dropped_duplicates'] = bus_before_dup - business_df.shape[0]

    # 3. INCONSISTENCIAS
    # Rango de estrellas [1, 5]
    rev_invalid_stars = review_df[(review_df['stars'] < 1) | (review_df['stars'] > 5)].shape[0]
    review_df = review_df[(review_df['stars'] >= 1) & (review_df['stars'] <= 5)]
    report['reviews_invalid_stars'] = rev_invalid_stars
    
    bus_invalid_stars = business_df[(business_df['stars'] < 1) | (business_df['stars'] > 5)].shape[0]
    business_df = business_df[(business_df['stars'] >= 1) & (business_df['stars'] <= 5)]
    report['business_invalid_stars'] = bus_invalid_stars
    
    # Consistencia cruzada (orphan reviews): reseñas de negocios que no existen en el catálogo
    valid_business_ids = set(business_df['business_id'])
    rev_orphan = review_df[~review_df['business_id'].isin(valid_business_ids)].shape[0]
    review_df = review_df[review_df['business_id'].isin(valid_business_ids)]
    report['reviews_orphan_removed'] = rev_orphan

    # 4. OUTLIERS (Identificación)
    # Número de reseñas por negocio en catálogo
    bus_reviews_stats = business_df['review_count'].describe()
    report['business_review_count_stats'] = bus_reviews_stats.to_dict()
    
    # Determinar umbral de outliers usando rango intercuartílico (IQR)
    Q1 = bus_reviews_stats['25%']
    Q3 = bus_reviews_stats['75%']
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    report['outlier_threshold_business_reviews'] = upper_bound
    
    # Contar negocios sobre el umbral (outliers)
    num_outliers_bus = business_df[business_df['review_count'] > upper_bound].shape[0]
    report['business_outliers_count'] = num_outliers_bus
    
    return review_df, business_df, report

def get_descriptive_statistics(review_df, business_df):
    """
    Calcula estadísticas descriptivas detalladas para las calificaciones y conteos.
    """
    stats = {}
    stats['review_stars_summary'] = review_df['stars'].describe().to_dict()
    stats['business_stars_summary'] = business_df['stars'].describe().to_dict()
    stats['business_review_count_summary'] = business_df['review_count'].describe().to_dict()
    
    # Estadísticas por segmento (Top 5 Estados con más negocios)
    top_states = business_df['state'].value_counts().head(5).index
    state_segments = []
    for state in top_states:
        state_df = business_df[business_df['state'] == state]
        state_segments.append({
            'state': state,
            'count': state_df.shape[0],
            'mean_stars': state_df['stars'].mean(),
            'median_stars': state_df['stars'].median(),
            'mean_review_count': state_df['review_count'].mean()
        })
    stats['state_segments'] = state_segments
    
    return stats

def plot_exploratory_charts(review_df, business_df, save_prefix=''):
    """
    Genera y guarda gráficos exploratorios clave en formato de imagen.
    """
    sns.set_theme(style="whitegrid")
    
    # Gráfico 1: Distribución de estrellas (Reviews vs Negocios)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.countplot(data=review_df, x='stars', ax=axes[0], hue='stars', palette='viridis', legend=False)
    axes[0].set_title('Distribución de Estrellas en Reseñas (Reviews)')
    axes[0].set_xlabel('Estrellas')
    axes[0].set_ylabel('Cantidad')
    
    sns.histplot(data=business_df, x='stars', bins=9, kde=True, ax=axes[1], color='coral')
    axes[1].set_title('Distribución de Estrellas Promedio en Negocios')
    axes[1].set_xlabel('Estrellas Promedio')
    axes[1].set_ylabel('Frecuencia')
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}distribucion_estrellas.png', dpi=300)
    plt.close()
    
    # Gráfico 2: Distribución del número de reseñas (Escala Logarítmica)
    plt.figure(figsize=(10, 5))
    sns.histplot(data=business_df, x='review_count', bins=50, log_scale=True, color='purple', kde=True)
    plt.title('Distribución del Conteo de Reseñas por Negocio (Escala Logarítmica)')
    plt.xlabel('Conteo de Reseñas (log)')
    plt.ylabel('Frecuencia')
    plt.tight_layout()
    plt.savefig(f'{save_prefix}distribucion_review_count.png', dpi=300)
    plt.close()
    
    # Gráfico 3: Top 15 categorías de negocio
    categories_series = business_df['categories'].str.split(',').explode().str.strip()
    top_cats = categories_series.value_counts().head(15)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_cats.values, y=top_cats.index, hue=top_cats.index, palette='magma', legend=False)
    plt.title('Top 15 Categorías de Negocios en Yelp')
    plt.xlabel('Cantidad de Negocios')
    plt.ylabel('Categoría')
    plt.tight_layout()
    plt.savefig(f'{save_prefix}top_categorias.png', dpi=300)
    plt.close()
    
    # Gráfico 4: Correlación entre estrellas promedio y review count
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=business_df, x='stars', y='review_count', alpha=0.3, color='teal')
    plt.yscale('log')
    plt.title('Relación entre Estrellas Promedio y Conteo de Reseñas')
    plt.xlabel('Estrellas Promedio')
    plt.ylabel('Conteo de Reseñas (Escala Log)')
    plt.tight_layout()
    plt.savefig(f'{save_prefix}correlacion_stars_reviews.png', dpi=300)
    plt.close()

    # Gráfico 5: Análisis por Segmento Geográfico (Top 5 Estados)
    top_states = business_df['state'].value_counts().head(5).index
    state_df = business_df[business_df['state'].isin(top_states)]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Subgráfico 5A: Cantidad de negocios por estado
    sns.countplot(data=state_df, x='state', order=top_states, ax=axes[0], hue='state', palette='coolwarm', legend=False)
    axes[0].set_title('Número de Negocios por Estado (Top 5)')
    axes[0].set_xlabel('Estado')
    axes[0].set_ylabel('Cantidad de Negocios')
    
    # Subgráfico 5B: Distribución de estrellas por estado (Boxplot)
    sns.boxplot(data=state_df, x='state', y='stars', order=top_states, ax=axes[1], hue='state', palette='coolwarm', legend=False)
    axes[1].set_title('Distribución de Estrellas por Estado (Top 5)')
    axes[1].set_xlabel('Estado')
    axes[1].set_ylabel('Estrellas Promedio')
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}analisis_segmentos_estados.png', dpi=300)
    plt.close()


def build_user_product_graph(sample_df):
    """
    Construye el grafo bipartito Usuario-Producto a partir de la muestra.
    Retorna el grafo construido y un diccionario con sus métricas básicas.
    """
    g = BipartiteGraph()
    for _, row in sample_df.iterrows():
        g.add_bipartite_edge(row['user_id'], row['business_id'])
        
    metrics = {}
    metrics['num_nodes'] = g.number_of_nodes()
    metrics['num_edges'] = g.number_of_edges()
    metrics['num_users'] = len(g.user_nodes)
    metrics['num_products'] = len(g.product_nodes)
    metrics['density'] = g.density()
    metrics['bipartite_density'] = g.bipartite_density()
    
    # Componentes conexas
    components = g.connected_components()
    metrics['num_components'] = len(components)
    
    largest_cc = max(components, key=len)
    metrics['largest_cc_size'] = len(largest_cc)
    metrics['largest_cc_percentage'] = (len(largest_cc) / metrics['num_nodes']) * 100
    
    # Calcular diámetro de la componente conexa más grande
    # Usando Multi-sweep BFS con 10 barridos para máxima rapidez y precisión
    approx_dia = g.diameter(largest_cc, method='approximate', max_sweeps=10)
    metrics['initial_diameter'] = approx_dia
    
    return g, metrics
