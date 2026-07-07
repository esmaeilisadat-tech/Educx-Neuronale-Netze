import nbformat

def fix_keras_input(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    for cell in nb.cells:
        if cell.cell_type == 'code':
            if 'tf.keras.layers.Conv2D(' in cell.source and 'input_shape=' in cell.source:
                # We need to replace input_shape=(..., ..., ...) with an explicit Input layer
                # Or just add `tf.keras.Input(shape=(28, 28, 1)),` right after `Sequential([`
                if 'Feature_Map_CNN' in cell.source:
                    cell.source = cell.source.replace(
                        'tf.keras.Sequential([\n    tf.keras.layers.Conv2D(16, (3, 3), activation="relu",\n                           input_shape=(28, 28, 1), padding="same", name="conv_1"),',
                        'tf.keras.Sequential([\n    tf.keras.Input(shape=(28, 28, 1)),\n    tf.keras.layers.Conv2D(16, (3, 3), activation="relu",\n                           padding="same", name="conv_1"),'
                    )
                elif 'MNIST_CNN' in cell.source:
                    cell.source = cell.source.replace(
                        'tf.keras.Sequential([\n    # Erster Faltungsblock: 32 Filter, 3×3 Kernel, ReLU\n    tf.keras.layers.Conv2D(32, (3, 3), activation="relu",\n                           input_shape=(28, 28, 1), name="conv_1"),',
                        'tf.keras.Sequential([\n    tf.keras.Input(shape=(28, 28, 1)),\n    # Erster Faltungsblock: 32 Filter, 3×3 Kernel, ReLU\n    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", name="conv_1"),'
                    )

    with open(filepath, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

fix_keras_input(r'c:\Users\esmae\Documents\Educx Neuronale Netze\NN_Projekt_Workspace\Tag_07_CNN\notebooks\Anfaenger.ipynb')
