import json
import sys

def inspect_notebook(path):
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    for i, c in enumerate(nb['cells']):
        cell_type = c['cell_type']
        source = "".join(c['source'])
        print(f"{i}: {cell_type} - {source[:100].replace('\n', ' ')}")

if __name__ == '__main__':
    inspect_notebook('c:/Users/esmae/Documents/Educx Neuronale Netze/NN_Projekt_Workspace/01_Neural_Network_Basics/notebooks/Anfaenger.ipynb')
