import json
import base64
import os

def extract_images_from_notebook(notebook_path, prefix):
    if not os.path.exists(notebook_path):
        return
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    img_counter = 1
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            for output in cell.get('outputs', []):
                if output.get('output_type') == 'display_data':
                    data = output.get('data', {})
                    if 'image/png' in data:
                        img_data = data['image/png']
                        filename = f"p2/{prefix}_{img_counter}.png"
                        with open(filename, "wb") as fh:
                            fh.write(base64.b64decode(img_data))
                        print(f"Extracted {filename}")
                        img_counter += 1

os.makedirs('p2', exist_ok=True)
extract_images_from_notebook('Parte_I_Preprocessing.ipynb', 'parte1')
extract_images_from_notebook('Parte_II_Graphs.ipynb', 'parte2')
extract_images_from_notebook('Parte_IV_Recommenders.ipynb', 'parte4')
