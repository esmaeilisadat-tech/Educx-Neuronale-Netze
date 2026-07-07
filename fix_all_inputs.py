import nbformat
import glob

def fix_all_inputs(directory):
    for filepath in glob.glob(directory + '/*.ipynb'):
        with open(filepath, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)

        modified = False
        for cell in nb.cells:
            if cell.cell_type == 'code':
                # Remove input_shape=(...) and insert tf.keras.Input
                # Since replacing exactly is hard, we can use regex or string replace.
                # Let's replace 'input_shape=(32, 32, 3),' with '' and prepend Input
                if 'input_shape=(32, 32, 3)' in cell.source:
                    cell.source = cell.source.replace('input_shape=(32, 32, 3), ', '')
                    cell.source = cell.source.replace('input_shape=(32, 32, 3)', '')
                    cell.source = cell.source.replace('tf.keras.Sequential([\n', 'tf.keras.Sequential([\n    tf.keras.Input(shape=(32, 32, 3)),\n')
                    modified = True
                
                if 'input_shape=(28, 28, 1)' in cell.source:
                    cell.source = cell.source.replace('input_shape=(28, 28, 1), ', '')
                    cell.source = cell.source.replace('input_shape=(28, 28, 1)', '')
                    cell.source = cell.source.replace('tf.keras.Sequential([\n', 'tf.keras.Sequential([\n    tf.keras.Input(shape=(28, 28, 1)),\n')
                    modified = True

                if 'input_shape=(48, 48, 3)' in cell.source:
                    cell.source = cell.source.replace('input_shape=(48, 48, 3), ', '')
                    cell.source = cell.source.replace('input_shape=(48, 48, 3)', '')
                    cell.source = cell.source.replace('tf.keras.Sequential([\n', 'tf.keras.Sequential([\n    tf.keras.Input(shape=(48, 48, 3)),\n')
                    modified = True

        if modified:
            with open(filepath, 'w', encoding='utf-8') as f:
                nbformat.write(nb, f)

fix_all_inputs(r'c:\Users\esmae\Documents\Educx Neuronale Netze\NN_Projekt_Workspace\Tag_07_CNN\notebooks')
