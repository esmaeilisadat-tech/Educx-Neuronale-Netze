import nbformat

new_code = """%matplotlib inline
# ============================================================
# educx GmbH – Neuronale Netze | Modul 2
# Lerntag 7: Convolutional Neural Networks (CNN)
# Niveau: Anfänger
# Aufgabe 3 von 3
# ============================================================
# Musterlösung – lauffähig in Spyder (tf_arm conda env)
# Python-Pfad: /Users/solusprime/opt/anaconda3/envs/tf_arm/bin/python
# ============================================================

import tensorflow as tf
import numpy as np
import matplotlib

import matplotlib.pyplot as plt
import math

print("TensorFlow Version:", tf.__version__)

# ── 1. MNIST laden und CNN trainieren ─────────────────────────────────────────
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train[:5000].astype("float32")[..., np.newaxis] / 255.0
x_test  = x_test.astype("float32")[..., np.newaxis]         / 255.0

# Wir nutzen hier die Functional API, damit get_layer().output in Keras 3 sicher funktioniert
inputs = tf.keras.Input(shape=(28, 28, 1), name="eingabe")
x = tf.keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same", name="conv_1")(inputs)
x = tf.keras.layers.MaxPooling2D((2, 2), name="pool_1")(x)
x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="conv_2")(x)
x = tf.keras.layers.MaxPooling2D((2, 2), name="pool_2")(x)
x = tf.keras.layers.Flatten(name="flatten")(x)
x = tf.keras.layers.Dense(64, activation="relu")(x)
outputs = tf.keras.layers.Dense(10, activation="softmax")(x)

modell = tf.keras.Model(inputs=inputs, outputs=outputs, name="Feature_Map_CNN")

modell.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
print("Trainiere kleines CNN...")
modell.fit(x_train, y_train[:5000], epochs=3, batch_size=64,
           validation_split=0.1, verbose=1)
print("Training abgeschlossen.")

# ── 2. Modell für Feature-Map-Extraktion erstellen ────────────────────────────
# Zwischenmodell: gibt Ausgabe der ersten Conv-Schicht zurück
feature_map_modell = tf.keras.Model(
    inputs=modell.input,
    outputs=modell.get_layer("conv_1").output,
    name="Feature_Map_Extraktor"
)

# ── 3. Feature Maps für ein Beispielbild extrahieren ─────────────────────────
# Erstes Testbild
probe_bild = x_test[0:1]           # Shape: (1, 28, 28, 1)
feature_maps = feature_map_modell.predict(probe_bild, verbose=0)
print(f"\\nFeature-Map-Shape (conv_1): {feature_maps.shape}")
# → (1, 28, 28, 16): 16 verschiedene Feature Maps für das eine Bild

# ── 4. Feature Maps visualisieren ────────────────────────────────────────────
n_filter = feature_maps.shape[-1]     # = 16
n_spalten = 4
n_zeilen  = math.ceil(n_filter / n_spalten)

fig, axes = plt.subplots(n_zeilen + 1, n_spalten, figsize=(n_spalten * 3, (n_zeilen + 1) * 3))

# Originalbild in der ersten Zeile
for col in range(n_spalten):
    if col == 0:
        axes[0, col].imshow(probe_bild[0, :, :, 0], cmap="gray")
        axes[0, col].set_title(f"Originalbild\\n(Ziffer: {y_test[0]})", fontsize=9)
    else:
        axes[0, col].axis("off")
axes[0, 0].axis("off")
axes[0, 0].imshow(probe_bild[0, :, :, 0], cmap="gray")

# Feature Maps in den weiteren Zeilen
for i in range(n_filter):
    zeile  = i // n_spalten + 1
    spalte = i  % n_spalten
    fm = feature_maps[0, :, :, i]
    axes[zeile, spalte].imshow(fm, cmap="viridis")
    axes[zeile, spalte].set_title(f"Filter {i+1}", fontsize=8)
    axes[zeile, spalte].axis("off")

# Leere Subplots ausblenden
for i in range(n_filter, n_zeilen * n_spalten):
    zeile  = i // n_spalten + 1
    spalte = i  % n_spalten
    axes[zeile, spalte].axis("off")

plt.suptitle("Feature Maps der ersten Conv2D-Schicht (16 Filter auf MNIST)", fontsize=13)
plt.tight_layout()
plt.savefig("A7_3_feature_maps.png", dpi=100)
plt.show()
print("Diagramm gespeichert: A7_3_feature_maps.png")

# ── 5. Statistiken der Feature Maps ──────────────────────────────────────────
print(f"\\nStatistiken der 16 Feature Maps:")
for i in range(n_filter):
    fm = feature_maps[0, :, :, i]
    print(f"  Filter {i+1:2d}: min={fm.min():.3f}, max={fm.max():.3f}, "
          f"mittelwert={fm.mean():.3f}")
"""

filepath = r'c:\Users\esmae\Documents\Educx Neuronale Netze\NN_Projekt_Workspace\Tag_07_CNN\notebooks\Anfaenger.ipynb'

with open(filepath, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

modified = False
for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'code':
        if 'Feature_Map_CNN' in cell.source:
            print(f"Modifying cell {i}")
            nb.cells[i].source = new_code
            modified = True

if modified:
    with open(filepath, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print("File saved.")
else:
    print("Could not find cell to modify.")
