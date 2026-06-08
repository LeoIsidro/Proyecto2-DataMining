import json
import pandas as pd
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
