import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
import os

def py_to_nb(py_path, nb_path):
    with open(py_path, 'r', encoding='utf-8') as f:
        code = f.read()
    
    nb = new_notebook()
    # we just put everything in one big code cell for now, or split by some marker
    # The python code has '# ── ' as markers. Let's split by '# ── ' to make separate cells.
    parts = code.split('\n# ── ')
    
    # First part
    if parts[0].strip():
        nb.cells.append(new_code_cell(parts[0]))
        
    for part in parts[1:]:
        cell_text = '# ── ' + part
        nb.cells.append(new_code_cell(cell_text))
        
    with open(nb_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print(f"Created clean notebook at {nb_path}")

py_to_nb(r'c:\Users\esmae\Documents\Educx Neuronale Netze\NN_Projekt_Workspace\Experte_code.py', 
         r'c:\Users\esmae\Documents\Educx Neuronale Netze\NN_Projekt_Workspace\Tag_06_Keras_TensorFlow\notebooks\Experte_Clean.ipynb')

py_to_nb(r'c:\Users\esmae\Documents\Educx Neuronale Netze\NN_Projekt_Workspace\Fortgeschrittene_code.py', 
         r'c:\Users\esmae\Documents\Educx Neuronale Netze\NN_Projekt_Workspace\Tag_06_Keras_TensorFlow\notebooks\Fortgeschrittene_Clean.ipynb')
