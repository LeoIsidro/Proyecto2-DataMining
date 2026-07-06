import json
import sys
import os

def parse_notebook(filepath):
    print(f"\n{'='*80}\nAnalizando: {os.path.basename(filepath)}\n{'='*80}")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for cell in data.get('cells', []):
        if cell['cell_type'] == 'markdown':
            source = "".join(cell.get('source', []))
            if source.strip():
                print("\n[MARKDOWN]:")
                print(source[:500] + ("..." if len(source) > 500 else ""))
        elif cell['cell_type'] == 'code':
            outputs = cell.get('outputs', [])
            for output in outputs:
                if output.get('output_type') == 'stream':
                    text = "".join(output.get('text', []))
                    if text.strip():
                        print("\n[OUTPUT TEXT]:")
                        print(text[:500] + ("..." if len(text) > 500 else ""))
                elif output.get('output_type') == 'execute_result' or output.get('output_type') == 'display_data':
                    data_out = output.get('data', {})
                    if 'text/plain' in data_out:
                        text = "".join(data_out['text/plain'])
                        if text.strip():
                            print("\n[OUTPUT RESULT]:")
                            print(text[:500] + ("..." if len(text) > 500 else ""))

if __name__ == '__main__':
    notebooks = [
        "Parte_I_Preprocessing.ipynb",
        "Parte_II_Graphs.ipynb",
        "Parte_III_Clustering_VI_Reduction_Dimensionality.ipynb",
        "Parte_IV_Recommenders.ipynb",
        "Parte_V.ipynb"
    ]
    for nb in notebooks:
        try:
            parse_notebook(nb)
        except Exception as e:
            print(f"Error procesando {nb}: {e}")
