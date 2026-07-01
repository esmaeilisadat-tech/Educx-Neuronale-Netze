#%matplotlib inline
# ============================================================
# educx GmbH – Neuronale Netze | Modul 2
# Lerntag 6: Keras & TensorFlow Grundlagen
# Niveau: Fortgeschrittene
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

# ── 1. MNIST laden ────────────────────────────────────────────────────────────
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32")  / 255.0

# ── 2. Functional API – Verzweigtes Modell ────────────────────────────────────
# Eingabe-Layer
eingabe = tf.keras.Input(shape=(28, 28), name="bild_eingabe")

# Gemeinsame (Shared) Schichten
flatten     = tf.keras.layers.Flatten(name="flatten")(eingabe)
gemeinsam_1 = tf.keras.layers.Dense(128, activation="relu", name="gemeinsam_1")(flatten)

# Zweig A: für niedrige Ziffern (0-4)
zweig_a = tf.keras.layers.Dense(64, activation="relu", name="zweig_a")(gemeinsam_1)
zweig_a = tf.keras.layers.Dropout(0.3, name="dropout_a")(zweig_a)

# Zweig B: für hohe Ziffern (5-9)
zweig_b = tf.keras.layers.Dense(64, activation="relu", name="zweig_b")(gemeinsam_1)
zweig_b = tf.keras.layers.Dropout(0.3, name="dropout_b")(zweig_b)

# Zusammenführen beider Zweige (Concatenate)
zusammen = tf.keras.layers.Concatenate(name="merge")([zweig_a, zweig_b])

# Ausgabe-Schicht
ausgabe = tf.keras.layers.Dense(10, activation="softmax", name="ausgabe")(zusammen)

# Modell mit tf.keras.Model(inputs=..., outputs=...) erstellen
model = tf.keras.Model(inputs=eingabe, outputs=ausgabe, name="Verzweigtes_MNIST_Modell")

# ── 3. Modell kompilieren ─────────────────────────────────────────────────────
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()

# ── 4. Text-Darstellung der Modellarchitektur (Fallback ohne graphviz) ────────
print("\n── Modellarchitektur (Textbeschreibung) ──")
print("Eingabe: (28, 28) → Flatten → Dense(128)")
print("         ├─ Zweig A: Dense(64) → Dropout(0.3)")
print("         └─ Zweig B: Dense(64) → Dropout(0.3)")
print("                └── Concatenate → Dense(10, softmax)")

# Schichten und deren Verbindungen ausgeben
print("\nSchichten-Details:")
for layer in model.layers:
    config = layer.get_config()
    print(f"  [{layer.name}] Typ: {type(layer).__name__}, "
          f"Ausgabe-Shape: {layer.output_shape}")

# ── 5. Training ───────────────────────────────────────────────────────────────
print("\nStarte Training...")
history = model.fit(
    x_train, y_train,
    epochs=8,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)

# ── 6. Evaluation ─────────────────────────────────────────────────────────────
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
print(f"\nTest-Verlust:     {test_loss:.4f}")
print(f"Test-Genauigkeit: {test_acc:.4f}")

# ── 7. Shared Layers demonstrieren ───────────────────────────────────────────
print("\n── Gemeinsame Schichten (Shared Layers) ──")
# Dieselbe Dense-Schicht auf zwei verschiedene Eingaben anwenden
gemeinsame_schicht = tf.keras.layers.Dense(32, activation="relu", name="shared_dense")

eingang_1 = tf.keras.Input(shape=(16,), name="eingang_1")
eingang_2 = tf.keras.Input(shape=(16,), name="eingang_2")
ausgang_1 = gemeinsame_schicht(eingang_1)
ausgang_2 = gemeinsame_schicht(eingang_2)  # gleiche Gewichte!

diff      = tf.keras.layers.Subtract(name="differenz")([ausgang_1, ausgang_2])
siamese   = tf.keras.layers.Dense(1, activation="sigmoid", name="aehnlichkeit")(diff)
siamese_modell = tf.keras.Model(
    inputs=[eingang_1, eingang_2], outputs=siamese, name="Siamese_Demo"
)
siamese_modell.summary()
print(f"\nAnzahl Gewichte in 'shared_dense': {len(gemeinsame_schicht.weights)}")
print("Beide Eingaben teilen dieselben Gewichte!")

# ── 8. Trainingsverlauf plotten ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history["loss"],     label="Training-Loss")
axes[0].plot(history.history["val_loss"], label="Validierungs-Loss")
axes[0].set_title("Verlust – Functional API Modell")
axes[0].set_xlabel("Epoche")
axes[0].set_ylabel("Verlust")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(history.history["accuracy"],     label="Training")
axes[1].plot(history.history["val_accuracy"], label="Validierung")
axes[1].set_title("Genauigkeit – Functional API Modell")
axes[1].set_xlabel("Epoche")
axes[1].set_ylabel("Genauigkeit")
axes[1].legend()
axes[1].grid(True)

plt.suptitle("Keras Functional API – Verzweigtes MNIST-Modell", fontsize=14)
plt.tight_layout()
plt.savefig("F6_1_functional_api.png", dpi=100)
plt.show()
print("Diagramm gespeichert: F6_1_functional_api.png")

#%matplotlib inline
# ============================================================
# educx GmbH – Neuronale Netze | Modul 2
# Lerntag 6: Keras & TensorFlow Grundlagen
# Niveau: Fortgeschrittene
# Aufgabe 2 von 3
# ============================================================
# Musterlösung – lauffähig in Spyder (tf_arm conda env)
# Python-Pfad: /Users/solusprime/opt/anaconda3/envs/tf_arm/bin/python
# ============================================================

import tensorflow as tf
import numpy as np
import matplotlib

import matplotlib.pyplot as plt

print("TensorFlow Version:", tf.__version__)

# ── 1. Eigener Keras-Layer: Lineare Projektion mit Skalierung ─────────────────
class SkalierteDense(tf.keras.layers.Layer):
    """
    Eigener Layer: Führt eine gewichtete lineare Projektion durch
    und multipliziert das Ergebnis mit einem lernbaren Skalierungsfaktor.
    y = activation(x @ W + b) * scale
    """

    def __init__(self, einheiten, aktivierung=None, **kwargs):
        super(SkalierteDense, self).__init__(**kwargs)
        self.einheiten   = einheiten
        self.aktivierung = tf.keras.activations.get(aktivierung)

    def build(self, eingabe_shape):
        """Gewichte werden erst beim ersten Aufruf erstellt (lazy building)."""
        # Gewichtsmatrix W
        self.W = self.add_weight(
            shape=(eingabe_shape[-1], self.einheiten),
            initializer="glorot_uniform",
            trainable=True,
            name="gewichtsmatrix"
        )
        # Bias-Vektor b
        self.b = self.add_weight(
            shape=(self.einheiten,),
            initializer="zeros",
            trainable=True,
            name="bias"
        )
        # Lernbarer Skalierungsfaktor (ein Skalar pro Einheit)
        self.skala = self.add_weight(
            shape=(self.einheiten,),
            initializer="ones",
            trainable=True,
            name="skalierungsfaktor"
        )
        # Wichtig: build() am Ende aufrufen
        super(SkalierteDense, self).build(eingabe_shape)

    def call(self, eingabe):
        """Forward-Pass: lineare Projektion + Skalierung."""
        z = tf.matmul(eingabe, self.W) + self.b
        if self.aktivierung is not None:
            z = self.aktivierung(z)
        return z * self.skala

    def get_config(self):
        """Konfiguration für Serialisierung."""
        config = super().get_config()
        config.update({
            "einheiten":   self.einheiten,
            "aktivierung": tf.keras.activations.serialize(self.aktivierung),
        })
        return config


# ── 2. Zweiter eigener Layer: Residual-Verbindung ─────────────────────────────
class ResidualBlock(tf.keras.layers.Layer):
    """
    Residual-Block: y = x + Dense(Dense(x))
    Eingabe und Ausgabe müssen dieselbe Dimension haben.
    """

    def __init__(self, einheiten, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.dense_1 = tf.keras.layers.Dense(einheiten, activation="relu")
        self.dense_2 = tf.keras.layers.Dense(einheiten)
        self.relu    = tf.keras.layers.Activation("relu")

    def call(self, eingabe):
        x = self.dense_1(eingabe)
        x = self.dense_2(x)
        return self.relu(x + eingabe)  # Residual-Verbindung


# ── 3. Synthetischen Datensatz erzeugen ───────────────────────────────────────
np.random.seed(42)
n = 2000
X = np.random.randn(n, 16).astype("float32")
# Binäre Klassifikation: y = 1 wenn Summe der Quadrate > Schwellwert
y = (np.sum(X**2, axis=1) > 16).astype("float32")
print(f"Datensatz: {X.shape}, Klassenverteilung: "
      f"{np.mean(y):.2%} positiv")

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 4. Modell mit eigenen Layern aufbauen ─────────────────────────────────────
modell = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(16,), name="eingabe"),
    SkalierteDense(64, aktivierung="relu", name="skaliert_1"),
    ResidualBlock(64, name="residual_1"),
    SkalierteDense(32, aktivierung="relu", name="skaliert_2"),
    tf.keras.layers.Dense(1, activation="sigmoid", name="ausgabe"),
], name="Modell_mit_eigenen_Layern")

modell.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
modell.summary()

# ── 5. Training ───────────────────────────────────────────────────────────────
print("\nStarte Training mit eigenen Layern...")
history = modell.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.15,
    verbose=1
)

# ── 6. Evaluation ─────────────────────────────────────────────────────────────
verlust, genauigkeit = modell.evaluate(X_test, y_test, verbose=1)
print(f"\nTest-Verlust:     {verlust:.4f}")
print(f"Test-Genauigkeit: {genauigkeit:.4f}")

# Skalierungsfaktoren ausgeben
print("\nGelernte Skalierungsfaktoren (skaliert_1):")
for layer in modell.layers:
    if isinstance(layer, SkalierteDense):
        print(f"  [{layer.name}] skala = {layer.skala.numpy()[:5]} ...")

# ── 7. Trainingsverlauf plotten ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history["loss"],     label="Training-Loss")
axes[0].plot(history.history["val_loss"], label="Validierungs-Loss")
axes[0].set_title("Verlust – Eigener Keras-Layer")
axes[0].set_xlabel("Epoche")
axes[0].set_ylabel("Verlust")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(history.history["accuracy"],     label="Training")
axes[1].plot(history.history["val_accuracy"], label="Validierung")
axes[1].set_title("Genauigkeit – Eigener Keras-Layer")
axes[1].set_xlabel("Epoche")
axes[1].set_ylabel("Genauigkeit")
axes[1].legend()
axes[1].grid(True)

plt.suptitle("Custom Keras Layer: SkalierteDense + ResidualBlock", fontsize=14)
plt.tight_layout()
plt.savefig("F6_2_custom_layer.png", dpi=100)
plt.show()
print("Diagramm gespeichert: F6_2_custom_layer.png")
print("\nEigener Keras-Layer funktioniert korrekt!")

#%matplotlib inline
# ============================================================
# educx GmbH – Neuronale Netze | Modul 2
# Lerntag 6: Keras & TensorFlow Grundlagen
# Niveau: Fortgeschrittene
# Aufgabe 3 von 3
# ============================================================
# Musterlösung – lauffähig in Spyder (tf_arm conda env)
# Python-Pfad: /Users/solusprime/opt/anaconda3/envs/tf_arm/bin/python
# ============================================================

import tensorflow as tf
import numpy as np
import matplotlib

import matplotlib.pyplot as plt
import os

print("TensorFlow Version:", tf.__version__)

# ── 1. MNIST-Teilmenge laden ──────────────────────────────────────────────────
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32")  / 255.0

# Nur 10.000 Trainingsbeispiele für schnelleres Training
x_train = x_train[:10000]
y_train = y_train[:10000]
print(f"Trainingsdaten (Teilmenge): {x_train.shape}")

# ── 2. Modell-Fabrik ──────────────────────────────────────────────────────────
def modell_erstellen():
    return tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10,  activation="softmax"),
    ], name="MNIST_Callbacks_Modell")

# ── 3. Callbacks definieren ───────────────────────────────────────────────────

# (a) EarlyStopping: Beendet Training wenn val_loss nicht mehr sinkt
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",   # überwachte Metrik
    patience=3,           # Epoche warten bevor Abbruch
    restore_best_weights=True,  # beste Gewichte wiederherstellen
    verbose=1
)

# (b) ModelCheckpoint: Bestes Modell automatisch speichern
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath="bestes_modell.keras",
    monitor="val_accuracy",
    save_best_only=True,    # nur bei Verbesserung speichern
    verbose=1
)

# (c) ReduceLROnPlateau: Lernrate reduzieren wenn kein Fortschritt
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,       # neue LR = alte LR * factor
    patience=2,       # Epochen warten
    min_lr=1e-6,      # Mindestwert der Lernrate
    verbose=1
)

# (d) LambdaCallback für Logging
epochen_log = []
lr_verlauf  = []

logging_callback = tf.keras.callbacks.LambdaCallback(
    on_epoch_end=lambda epoch, logs: (
        epochen_log.append(epoch),
        lr_verlauf.append(float(tf.keras.backend.get_value(modell.optimizer.learning_rate)))
    )
)

# ── 4. Modell kompilieren und trainieren ──────────────────────────────────────
modell = modell_erstellen()
modell.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
modell.summary()

print("\nStarte Training mit Callbacks (EarlyStopping, ModelCheckpoint, ReduceLR)...")
history = modell.fit(
    x_train, y_train,
    epochs=50,         # viele Epochen – EarlyStopping beendet früher
    batch_size=128,
    validation_split=0.15,
    callbacks=[early_stopping, checkpoint, reduce_lr, logging_callback],
    verbose=1
)

# ── 5. Ergebnisse ausgeben ────────────────────────────────────────────────────
tatsaechliche_epochen = len(history.history["loss"])
print(f"\nTraining gestoppt nach {tatsaechliche_epochen} Epochen "
      f"(EarlyStopping patience=3)")

test_loss, test_acc = modell.evaluate(x_test, y_test, verbose=1)
print(f"Test-Verlust:     {test_loss:.4f}")
print(f"Test-Genauigkeit: {test_acc:.4f}")

# ── 6. Visualisierung ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Verlust
axes[0].plot(history.history["loss"],     label="Training-Loss")
axes[0].plot(history.history["val_loss"], label="Validierungs-Loss")
axes[0].set_title("Verlust (Loss)")
axes[0].set_xlabel("Epoche")
axes[0].set_ylabel("Verlust")
axes[0].legend()
axes[0].grid(True)

# Genauigkeit
axes[1].plot(history.history["accuracy"],     label="Training")
axes[1].plot(history.history["val_accuracy"], label="Validierung")
axes[1].axvline(x=tatsaechliche_epochen - 1, color="red",
                linestyle="--", label="EarlyStopping")
axes[1].set_title("Genauigkeit")
axes[1].set_xlabel("Epoche")
axes[1].set_ylabel("Genauigkeit")
axes[1].legend()
axes[1].grid(True)

# Lernrate
axes[2].plot(epochen_log, lr_verlauf, color="green", marker="o", markersize=4)
axes[2].set_title("Lernrate (ReduceLROnPlateau)")
axes[2].set_xlabel("Epoche")
axes[2].set_ylabel("Lernrate")
axes[2].set_yscale("log")
axes[2].grid(True)

plt.suptitle("Keras Callbacks: EarlyStopping + ModelCheckpoint + ReduceLR", fontsize=13)
plt.tight_layout()
plt.savefig("F6_3_callbacks.png", dpi=100)
plt.show()
print("Diagramm gespeichert: F6_3_callbacks.png")

# Callback-Erklärungen ausgeben
print("\n── Callback-Erklärungen ──")
print("EarlyStopping: Stoppt Training wenn val_loss sich nicht verbessert "
      "(patience=3 Epochen). Verhindert Overfitting.")
print("ModelCheckpoint: Speichert das Modell automatisch wenn val_accuracy "
      "sich verbessert. Sichert das beste Modell.")
print("ReduceLROnPlateau: Halbiert die Lernrate wenn val_loss stagniert "
      "(patience=2). Ermöglicht feinere Konvergenz.")

