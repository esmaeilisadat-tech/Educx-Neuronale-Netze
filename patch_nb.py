import nbformat
import os

paths = [
    r'c:\Users\esmae\Documents\Educx Neuronale Netze\NN_Projekt_Workspace\Tag_06_Keras_TensorFlow\notebooks\Fortgeschrittene.ipynb',
    r'c:\Users\esmae\Documents\Educx Neuronale Netze\NN_Projekt_Workspace\Tag_06_Keras_TensorFlow\notebooks\Fortgeschrittene_Clean.ipynb'
]

old_code = '''for layer in model.layers:
    config = layer.get_config()
    print(f"  [{layer.name}] Typ: {type(layer).__name__}, "
          f"Ausgabe-Shape: {layer.output_shape}")'''

new_code = '''for layer in model.layers:
    config = layer.get_config()
    try:
        shape = layer.output_shape
    except AttributeError:
        shape = getattr(layer, 'output', None)
        shape = shape.shape if shape is not None else "N/A"
    print(f"  [{layer.name}] Typ: {type(layer).__name__}, "
          f"Ausgabe-Shape: {shape}")'''

for p in paths:
    if not os.path.exists(p): continue
    with open(p, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
        
    for cell in nb.cells:
        if cell.cell_type == 'code':
            if 'layer.output_shape' in cell.source:
                cell.source = cell.source.replace(old_code, new_code)
                
    with open(p, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print(f"Patched {os.path.basename(p)}")
