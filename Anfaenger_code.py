%matplotlib inline
# ============================================================
# educx GmbH – Neuronale Netze | Modul 2
# Lerntag 6: Keras & TensorFlow Grundlagen
# Niveau: Anfänger
# Aufgabe 1 von 3
# ============================================================
# Musterlösung – lauffähig in Spyder (tf_arm conda env)
# Python-Pfad: /Users/solusprime/opt/anaconda3/envs/tf_arm/bin/python
# ============================================================

import tensorflow as tf
import numpy as np
import matplotlib

import matplotlib.pyplot as plt

print("TensorFlow Version:", tf.__version__)

# ── 1. MNIST-Daten laden und vorbereiten ─────────────────────────────────────
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalisierung: Pixelwerte von [0,255] auf [0,1] skalieren
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32")  / 255.0

print(f"Trainingsdaten:  {x_train.shape}  Labels: {y_train.shape}")
print(f"Testdaten:       {x_test.shape}   Labels: {y_test.shape}")

# ── 2. Sequential-Modell aufbauen ─────────────────────────────────────────────
model = tf.keras.Sequential([
    # Flatten: 28x28-Bild → 784-Vektor
    tf.keras.layers.Flatten(input_shape=(28, 28), name="flatten"),
    # Erste Dense-Schicht mit ReLU-Aktivierung
    tf.keras.layers.Dense(128, activation="relu", name="dense_1"),
    # Zweite Dense-Schicht (Ausgabe) mit Softmax für 10 Klassen
    tf.keras.layers.Dense(10,  activation="softmax", name="ausgabe"),
], name="MNIST_Sequential")

# ── 3. Modell kompilieren ─────────────────────────────────────────────────────
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Modell-Zusammenfassung ausgeben
model.summary()

# ── 4. Training ───────────────────────────────────────────────────────────────
print("\nStarte Training...")
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.1,   # 10 % der Trainingsdaten als Validierung
    verbose=1
)

# ── 5. Evaluation auf Testdaten ───────────────────────────────────────────────
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
print(f"\nTest-Verlust:     {test_loss:.4f}")
print(f"Test-Genauigkeit: {test_acc:.4f}")

# ── 6. Trainingsverlauf visualisieren ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Verlust (Loss)
axes[0].plot(history.history["loss"],     label="Training-Loss")
axes[0].plot(history.history["val_loss"], label="Validierungs-Loss")
axes[0].set_title("Modell-Verlust (Loss)")
axes[0].set_xlabel("Epoche")
axes[0].set_ylabel("Verlust")
axes[0].legend()
axes[0].grid(True)

# Genauigkeit (Accuracy)
axes[1].plot(history.history["accuracy"],     label="Training-Genauigkeit")
axes[1].plot(history.history["val_accuracy"], label="Validierungs-Genauigkeit")
axes[1].set_title("Modell-Genauigkeit (Accuracy)")
axes[1].set_xlabel("Epoche")
axes[1].set_ylabel("Genauigkeit")
axes[1].legend()
axes[1].grid(True)

plt.suptitle("Keras Sequential API – MNIST Training", fontsize=14)
plt.tight_layout()
plt.savefig("A6_1_training_history.png", dpi=100)
plt.show()
print("Diagramm gespeichert: A6_1_training_history.png")

%matplotlib inline
# ============================================================
# educx GmbH – Neuronale Netze | Modul 2
# Lerntag 6: Keras & TensorFlow Grundlagen
# Niveau: Anfänger
# Aufgabe 2 von 3
# ============================================================
# Musterlösung – lauffähig in Spyder (tf_arm conda env)
# Python-Pfad: /Users/solusprime/opt/anaconda3/envs/tf_arm/bin/python
# ============================================================

import tensorflow as tf
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib

import matplotlib.pyplot as plt

print("TensorFlow Version:", tf.__version__)

# ── 1. Iris-Datensatz laden ───────────────────────────────────────────────────
iris = load_iris()
X, y = iris.data.astype("float32"), iris.target

# Daten normalisieren (Mittelwert=0, Std=1)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-/Test-Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Trainingsdaten: {X_train.shape}, Testdaten: {X_test.shape}")

# ── 2. Kleines Modell aufbauen ────────────────────────────────────────────────
def modell_erstellen():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation="relu", input_shape=(4,), name="dense_1"),
        tf.keras.layers.Dense(8,  activation="relu", name="dense_2"),
        tf.keras.layers.Dense(3,  activation="softmax", name="ausgabe"),
    ], name="Iris_Modell")
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

model = modell_erstellen()
model.summary()

# ── 3. Modell trainieren ──────────────────────────────────────────────────────
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.15,
    verbose=1   # kein detaillierter Output
)
print("Training abgeschlossen.")

# Vorhersagen VOR dem Speichern
vorhersagen_vor = model.predict(X_test, verbose=1)
klassen_vor = np.argmax(vorhersagen_vor, axis=1)
print(f"\nVorhersagen VOR dem Speichern (erste 5): {klassen_vor[:5]}")
print(f"Wahre Labels                (erste 5): {y_test[:5]}")

# ── 4. Modell speichern ───────────────────────────────────────────────────────
speicherpfad = "iris_model.keras"
model.save(speicherpfad)
print(f"\nModell gespeichert unter: {speicherpfad}")

# ── 5. Modell laden ───────────────────────────────────────────────────────────
geladenes_modell = tf.keras.models.load_model(speicherpfad)
print("Modell erfolgreich geladen!")
geladenes_modell.summary()

# Vorhersagen NACH dem Laden
vorhersagen_nach = geladenes_modell.predict(X_test, verbose=1)
klassen_nach = np.argmax(vorhersagen_nach, axis=1)
print(f"\nVorhersagen NACH dem Laden (erste 5): {klassen_nach[:5]}")

# ── 6. Vergleich: Identische Vorhersagen? ─────────────────────────────────────
unterschiede = np.sum(klassen_vor != klassen_nach)
print(f"\nAnzahl unterschiedlicher Vorhersagen: {unterschiede}")
if unterschiede == 0:
    print("Perfekt: Vorhersagen sind identisch – Modell korrekt gespeichert!")

# ── 7. Trainingsverlauf plotten ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history["loss"],     label="Training-Loss")
axes[0].plot(history.history["val_loss"], label="Validierungs-Loss")
axes[0].set_title("Verlust (Loss)")
axes[0].set_xlabel("Epoche")
axes[0].set_ylabel("Verlust")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(history.history["accuracy"],     label="Training")
axes[1].plot(history.history["val_accuracy"], label="Validierung")
axes[1].set_title("Genauigkeit (Accuracy)")
axes[1].set_xlabel("Epoche")
axes[1].set_ylabel("Genauigkeit")
axes[1].legend()
axes[1].grid(True)

plt.suptitle("Iris-Modell: Speichern und Laden", fontsize=14)
plt.tight_layout()
plt.savefig("A6_2_iris_training.png", dpi=100)
plt.show()
print("Diagramm gespeichert: A6_2_iris_training.png")

print("\nModell erfolgreich gespeichert und geladen!")

%matplotlib inline
# ============================================================
# educx GmbH – Neuronale Netze | Modul 2
# Lerntag 6: Keras & TensorFlow Grundlagen
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

print("=" * 60)
print("TensorFlow Tensoren – Grundlagen")
print("TF-Version:", tf.__version__)
print("=" * 60)

# ── 1. tf.constant – unveränderlicher Tensor ──────────────────────────────────
print("\n── 1. tf.constant ──")
skalar    = tf.constant(42, dtype=tf.int32)
vektor    = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
matrix    = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)
tensor_3d = tf.constant(np.ones((2, 3, 4)), dtype=tf.float32)

print(f"Skalar:    Wert={skalar.numpy()}, Shape={skalar.shape}, Dtype={skalar.dtype}")
print(f"Vektor:    Wert={vektor.numpy()}, Shape={vektor.shape}, Dtype={vektor.dtype}")
print(f"Matrix:    Shape={matrix.shape}, Dtype={matrix.dtype}")
print(f"3D-Tensor: Shape={tensor_3d.shape}, Dtype={tensor_3d.dtype}")

# ── 2. tf.Variable – veränderbarer Tensor ─────────────────────────────────────
print("\n── 2. tf.Variable ──")
variable = tf.Variable([1.0, 2.0, 3.0], name="meine_variable")
print(f"Vor Zuweisung:  {variable.numpy()}")
variable.assign([10.0, 20.0, 30.0])        # Wert komplett ersetzen
print(f"Nach assign():  {variable.numpy()}")
variable.assign_add([1.0, 1.0, 1.0])       # Addition
print(f"Nach assign_add(): {variable.numpy()}")

# ── 3. Grundlegende Tensor-Operationen ───────────────────────────────────────
print("\n── 3. Tensor-Operationen ──")
a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
b = tf.constant([[5.0, 6.0], [7.0, 8.0]])

# Addition
summe = tf.add(a, b)
print(f"Addition:\n{summe.numpy()}")

# Elementweise Multiplikation
produkt = tf.multiply(a, b)
print(f"Elementweise Multiplikation:\n{produkt.numpy()}")

# Matrix-Multiplikation
matmul = tf.matmul(a, b)
print(f"Matrix-Multiplikation:\n{matmul.numpy()}")

# Transponierung
transponiert = tf.transpose(a)
print(f"Transponiert:\n{transponiert.numpy()}")

# ── 4. Reshape und Umformen ───────────────────────────────────────────────────
print("\n── 4. Reshape ──")
flach   = tf.constant(list(range(12)), dtype=tf.float32)
reshaped = tf.reshape(flach, (3, 4))
print(f"Original (flach): Shape={flach.shape}")
print(f"Umgeformt (3×4):  Shape={reshaped.shape}")
print(reshaped.numpy())

# ── 5. Mathematische Funktionen ───────────────────────────────────────────────
print("\n── 5. Mathematische Funktionen ──")
x = tf.constant([-2.0, -1.0, 0.0, 1.0, 2.0])
print(f"ReLU:    {tf.nn.relu(x).numpy()}")
print(f"Sigmoid: {tf.sigmoid(x).numpy()}")
print(f"Tanh:    {tf.tanh(x).numpy()}")
print(f"Softmax: {tf.nn.softmax(x).numpy()}")

# ── 6. Geräteplatzierung (CPU/GPU) ────────────────────────────────────────────
print("\n── 6. Geräteplatzierung ──")
verfuegbare_geraete = tf.config.list_physical_devices()
print("Verfügbare Geräte:")
for g in verfuegbare_geraete:
    print(f"  {g}")

# Tensor auf einem bestimmten Gerät erstellen
with tf.device('/CPU:0'):
    cpu_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
print(f"CPU-Tensor Gerät: {cpu_tensor.device}")

# ── 7. Visualisierung ─────────────────────────────────────────────────────────
x_vals = np.linspace(-4, 4, 200)
relu_vals    = tf.nn.relu(x_vals).numpy()
sigmoid_vals = tf.sigmoid(tf.cast(x_vals, tf.float32)).numpy()
tanh_vals    = tf.tanh(tf.cast(x_vals, tf.float32)).numpy()

plt.figure(figsize=(10, 5))
plt.plot(x_vals, relu_vals,    label="ReLU",    linewidth=2)
plt.plot(x_vals, sigmoid_vals, label="Sigmoid", linewidth=2)
plt.plot(x_vals, tanh_vals,    label="Tanh",    linewidth=2)
plt.axhline(0, color="black", linewidth=0.5)
plt.axvline(0, color="black", linewidth=0.5)
plt.title("TensorFlow Aktivierungsfunktionen")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("A6_3_aktivierungsfunktionen.png", dpi=100)
plt.show()
print("\nDiagramm gespeichert: A6_3_aktivierungsfunktionen.png")
print("\nAlle Tensor-Grundlagen erfolgreich demonstriert!")

